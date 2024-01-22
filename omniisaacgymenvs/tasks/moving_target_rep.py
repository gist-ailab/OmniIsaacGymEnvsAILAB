import torch
from torch import linalg as LA
import numpy as np
import cv2
from gym import spaces
 
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")   # required by PytorchListener
# enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface
import omni.replicator.core as rep
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ur5e_tool import UR5eTool
from omniisaacgymenvs.robots.articulations.views.ur5e_view import UR5eView
from omniisaacgymenvs.tasks.utils.pcd_writer import PointcloudWriter
from omniisaacgymenvs.tasks.utils.pcd_listener import PointcloudListener
from omni.isaac.core.utils.semantics import add_update_semantics

import omni
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.objects import DynamicCylinder, DynamicSphere, DynamicCuboid
# from omni.isaac.core.materials import PhysicsMaterial
# from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
# from omni.isaac.sensor import Camera, LidarRtx, RotatingLidarPhysX
from omni.kit.viewport.utility import get_active_viewport

from skrl.utils import omniverse_isaacgym_utils
from pxr import Usd, UsdGeom, Gf, UsdPhysics, Semantics              # pxr usd imports used to create cube

from typing import Optional, Tuple
import asyncio

import open3d as o3d
import point_cloud_utils as pcu
import copy


# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()    



class MovingTargetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        #################### BSH
        self.rep = rep
        self.camera_width = 640
        self.camera_height = 640
        #################### BSH

        self.update_config(sim_config)

        self.step_num = 0
        self.dt = 1 / 120.0
        self._env = env      

        self.previous_tool_target_distance = None
        self.current_tool_target_distance = None
        self.previous_target_goal_distance = None
        self.current_target_goal_distance = None
        self.previous_target_position = None
        self.current_target_position = None
        self.target_moving_distance = None
        
        self.alpha = 0.15
        self.beta = 0.5
        self.gamma = 0
        self.zeta = 0.1
        self.eta = 0.25
        self.punishment = -1

        self.stage = omni.usd.get_context().get_stage()        

        self._pcd_sampling_num = self._task_cfg["sim"]["point_cloud_samples"]
        
        # observation and action space
        pcd_observations = self._pcd_sampling_num * 2 * 3     # 2 is a number of point cloud masks and 3 is a cartesian coordinate
        self._num_observations = pcd_observations + 6 + 6 + 3 + 4 + 3 + 3
        '''
        refer to observations in get_observations()
        dof_pos_scaled,                               # [NE, 6]
        dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
        flange_pos,                                   # [NE, 3]
        flange_rot,                                   # [NE, 4]
        target_pos,                                   # [NE, 3]
        goal_pos,                                     # [NE, 3]
        
        '''
        # TODO: point cloud의 sub mask 개수를 state 또는 config에서 받아올 것.

        if self._control_space == "joint":
            self._num_actions = 6
        elif self._control_space == "cartesian":
            self._num_actions = 3
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))

        self._flange_link = "flange_tool_rot_x"

        # ############################################################ BSH
        # # use multi-dimensional observation for camera RGB => 이거 쓰면 오류남
        # self.observation_space = spaces.Box(
        #     np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
        #     np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)
        # ############################################################ BSH
        
        self.PointcloudWriter = PointcloudWriter
        self.PointcloudListener = PointcloudListener
        
        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self._action_scale = self._task_cfg["env"]["actionScale"]
        # self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        # self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]

        self._dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]

        self._control_space = self._task_cfg["env"]["controlSpace"]
        self._goal_position = self._task_cfg["env"]["goal"]
        self._pcd_normalization = self._task_cfg["sim"]["point_cloud_normalization"]

        
    def set_up_scene(self, scene) -> None:
        self.get_robot()
        self.get_target()
        self.get_goal()
        # self.get_lidar(idx=0, scene=scene)

        # RLTask.set_up_scene(self, scene)
        super().set_up_scene(scene)

        # robot view
        self._robots = UR5eView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        # flanges view
        self._flanges = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/{self._flange_link}", name="end_effector_view")
        # target view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)        
        # goal view
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)

        scene.add(self._goals)
        scene.add(self._robots)
        scene.add(self._flanges)
        scene.add(self._targets)
        
        
        # point cloud view
        self.render_products = []
        camera_positions = {0: [2, 1.5, 0.5],
                            1: [2, -1.3, 0.5],
                            2: [-1.5, -2.2, 0.5]}
        camera_rotations = {0: [0, -10, 60],
                            1: [0, -10, -45],
                            2: [0, -10, -130]}
        env_pos = self._env_pos.cpu()

        # Used to get depth data from the camera
        for i in range(self._num_envs):
            self.rep.get.camera()
            # Get the parameters of the camera via links below
            # https://learn.microsoft.com/en-us/answers/questions/201906/pixel-size-of-rgb-and-tof-camera#:~:text=Pixel%20Size%20for,is%20~2.3%20mm
            # https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html#calibrated-camera-sensors
            '''
            Azure Kinect DK
            [Depth Camera]
            focal_length = 0.18 (cm)
            focus_distance = 3.86 (m) 그런데 clipping range가 있는데 이게 어떻게 작용할지 불확실함
            horizontal_aperture = 0.224 (cm)
            vertical_aperture = 0.2016 (cm)
            clipping_range = 0.5 ~ 3.86 (m)
            위 값을 RGB로 가시화 하면 이상하긴 한데, depth camera 니까 우선 해보자.
            
            여기 보면 또 mm임..
            https://docs.omniverse.nvidia.com/isaacsim/latest/manual_replicator_composer_parameter_list.html#camera-lens
            

            [RGB Camera]
            focal_length = 0.23 (cm)
            focus_distance = 5 (m) 그런데 clipping range가 있는데 이게 어떻게 작용할지 불확실함
            horizontal_aperture = 0.24 (cm)
            vertical_aperture = 0.135 (cm)
            clipping_range = 0.01 ~ 10 (m)
            '''
            # pos: (2 -1.3 0.5), ori: (0, -10, -45)
            # pos: (-1.5 -2.2 0.5), ori: (0, -10, -130)
            # 

            for j in range(3):
                locals()[f"camera_{j}"] = self.rep.create.camera(
                                                                 position = (env_pos[i][0] + camera_positions[j][0],
                                                                             env_pos[i][1] + camera_positions[j][1],
                                                                             env_pos[i][2] + camera_positions[j][2]),
                                                                 rotation=(camera_rotations[j][0],
                                                                           camera_rotations[j][1],
                                                                           camera_rotations[j][2]), 
                                                                 focal_length=1.8,
                                                                 focus_distance=3.86,
                                                                 horizontal_aperture=2.24,
                                                                 #  vertical_aperture=0.2016,
                                                                 clipping_range=(0.5, 3.86),
                                                                 # TODO: clipping range 조절해서 환경이 서로 안 겹치게 하자.
                                                                 )
                
                render_product = self.rep.create.render_product(locals()[f"camera_{j}"], resolution=(self.camera_width, self.camera_height))
                self.render_products.append(render_product)

            # camera_position = (2 + env_pos[i][0], 1.5 + env_pos[i][1], 0.5)
            # self.camera_positions[i] = camera_position
            # self.camera_1_orientations = (0, -10, 60)
            # # self.camera_orientations = (0, 0, 60)
            # camera = self.rep.create.camera(
            #     # position=(0.5 + env_pos[i][0], 1.5 + env_pos[i][1], 0.4),
            #     # position=(2 + env_pos[i][0], 1.5 + env_pos[i][1], 0.5),
            #     position=self.camera_positions,
            #     rotation=self.camera_orientations,

            #     focal_length=1.8,
            #     focus_distance=3.86,
            #     horizontal_aperture=2.24,
            #     # vertical_aperture=0.2016,
            #     clipping_range=(0.5, 3.86),
                
            #     )
            # render_product = self.rep.create.render_product(camera, resolution=(self.camera_width, self.camera_height))
            # distance_to_camera = self.rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            # distance_to_image_plane = self.rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            # pointcloud = self.rep.AnnotatorRegistry.get_annotator("pointcloud")

            # self.render_products.append(render_product)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # initialize pytorch writer for vectorized collection
        self.pointcloud_listener = self.PointcloudListener()
        self.pointcloud_writer = self.rep.WriterRegistry.get("PointcloudWriter")
        self.pointcloud_writer.initialize(listener=self.pointcloud_listener,
                                          pcd_sampling_num=self._pcd_sampling_num,
                                          pcd_normalize = self._pcd_normalization,
                                          env_pos = self._env_pos.cpu(),
                                          camera_positions=camera_positions,
                                          camera_orientations=camera_rotations,
                                          device=self.device,
                                          )
        self.pointcloud_writer.attach(self.render_products)
        ################################################################################## 231121 added BSH
            
        # get robot semantic data
        # 그런데 어짜피 로봇 point cloud는 필요 없기 때문에 안 받아도 될듯
        self._robot_semantics = {}
        for i in range(self._num_envs):
            robot_prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/robot")
            self._robot_semantics[i] = Semantics.SemanticsAPI.Apply(robot_prim, "Semantics")
            self._robot_semantics[i].CreateSemanticTypeAttr()
            self._robot_semantics[i].CreateSemanticDataAttr()
            self._robot_semantics[i].GetSemanticTypeAttr().Set("class")
            self._robot_semantics[i].GetSemanticDataAttr().Set(f"robot_{i}")
            add_update_semantics(robot_prim, '0')


        # get tool semantic data
        self._tool_semantics = {}
        for i in range(self._num_envs):
            tool_prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/robot/tool")
            self._tool_semantics[i] = Semantics.SemanticsAPI.Apply(tool_prim, "Semantics")
            self._tool_semantics[i].CreateSemanticTypeAttr()
            self._tool_semantics[i].CreateSemanticDataAttr()
            self._tool_semantics[i].GetSemanticTypeAttr().Set("class")
            self._tool_semantics[i].GetSemanticDataAttr().Set(f"tool_{i}")
            add_update_semantics(tool_prim, '1')

        # get target object semantic data
        self._target_semantics = {}
        for i in range(self._num_envs):
            target_prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/target")
            self._target_semantics[i] = Semantics.SemanticsAPI.Apply(target_prim, "Semantics")
            self._target_semantics[i].CreateSemanticTypeAttr()
            self._target_semantics[i].CreateSemanticDataAttr()
            self._target_semantics[i].GetSemanticTypeAttr().Set("class")
            self._target_semantics[i].GetSemanticDataAttr().Set(f"target_{i}")
            add_update_semantics(target_prim, '2')

        self.init_data()
        # return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exist("robot_view"):
            scene.remove("robot_view", registry_only=True)
        if scene.object_exist("end_effector_view"):
            scene.remove("end_effector_view", registry_only=True)
        if scene.object_exist("target_view"):
            scene.remove("target_view", registry_only=True)
        if scene.object_exist("goal_view"):
            scene.remove("goal_view", registry_only=True)

        self._robots = UR5eView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        self._flanges = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/{self._flange_link}", name="end_effector_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)

        scene.add(self._robots)
        scene.add(self._flanges)
        scene.add(self._targets)
        scene.add(self._goals)
        
        self.init_data()

    def get_robot(self):
        robot = UR5eTool(prim_path=self.default_zero_env_path + "/robot",
                         translation=torch.tensor([0.0, 0.0, 0.0]),
                         orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                         name="robot")
        self._sim_config.apply_articulation_settings("robot",
                                                     get_prim_at_path(robot.prim_path),
                                                     self._sim_config.parse_actor_config("robot"))

    def get_target(self):
        target = DynamicCuboid(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               size=0.07,
                               density=1,
                               color=torch.tensor([255, 0, 0]),
                            #    physics_material=PhysicsMaterial(prim_path=physics_material_path,
                            #                                     static_friction=0.1, dynamic_friction=0.1),
                               )
        
        self._sim_config.apply_articulation_settings("target",
                                                     get_prim_at_path(target.prim_path),
                                                     self._sim_config.parse_actor_config("target"))

    def get_goal(self):
        goal = DynamicSphere(prim_path=self.default_zero_env_path + "/goal",
                             name="goal",
                             radius=0.025,
                             color=torch.tensor([0, 255, 0]))
        # goal.disable_rigid_body_physics()
        self._sim_config.apply_articulation_settings("goal",
                                                     get_prim_at_path(goal.prim_path),
                                                     self._sim_config.parse_actor_config("goal"))
        

    def init_data(self) -> None:
        self.robot_default_dof_pos = torch.tensor(np.radians([-40, -45, 60, -100, -90, 90.0,
                                                              0.0, 0.0, 0.0, 0.0]), device=self._device, dtype=torch.float32)

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        if self._control_space == "cartesian":
            # self.jacobians = torch.zeros((self._num_envs, 11, 6, 6), device=self._device)
            if self._robots.body_names == None:
                # self.jacobians = torch.zeros((self._num_envs, 11, 6, 6), device=self._device)
                self.jacobians = torch.zeros((self._num_envs, 9, 6, 6), device=self._device)
                # end-effector link index is 9 which is the flange
            else:
                self.jacobians = torch.zeros((self._num_envs,
                                              self._robots.body_names.index(self._flange_link),
                                              6,
                                              6), device=self._device)
                '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
                num_of_bodies - 1 due to start from 0 index'''
            self.flange_pos, self.flange_rot = torch.zeros((self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)


    def visualize_point_cloud(self, view_idx, lidar_position):
        '''
        args:
            view_idx: index of the cloner
            lidar_position: position of the lidar
        '''
        flange_pos, flange_rot = self._flanges.get_local_poses()

        lidar_prim_path = self._point_cloud[view_idx].prim_path
        point_cloud = self._point_cloud[view_idx]._lidar_sensor_interface.get_point_cloud_data(lidar_prim_path)
        semantic = self._point_cloud[view_idx]._lidar_sensor_interface.get_semantic_data(lidar_prim_path)

        pcl_reshape = np.reshape(point_cloud, (point_cloud.shape[0]*point_cloud.shape[1], 3))
        flange_pos_np = flange_pos[view_idx].cpu().numpy()
        flange_ori_np = flange_rot[view_idx].cpu().numpy()

        pcl_semantic = np.reshape(semantic, -1)        


        v3d = o3d.utility.Vector3dVector

        # get point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = v3d(pcl_reshape)

        # get sampled point cloud
        idx = pcu.downsample_point_cloud_poisson_disk(pcl_reshape, num_samples=int(0.2*pcl_reshape.shape[0]))
        pcl_reshape_sampled = pcl_reshape[idx]
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = v3d(pcl_reshape_sampled)

        # get lidar frame. lidar frame is a world [0, 0, 0] frame
        lidar_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0.0, 0.0, 0.0]))

        # get base pose    
        T_b = np.eye(4)
        # TODO: get rotation matrix from USD
        # R_b = lidar_coord.get_rotation_matrix_from_xyz((np.pi/6, 0, -np.pi/2))
        R_b = lidar_coord.get_rotation_matrix_from_xyz((np.pi/9, 0, -np.pi/2))  # rotation relationship between lidar and base
        T_b[:3, :3] = R_b
        T_b[:3, 3] = lidar_position # acquired position from prim_path is [0,0,0]
        T_b_inv = np.linalg.inv(T_b)
        base_coord = copy.deepcopy(lidar_coord).transform(T_b_inv)
        # o3d.visualization.draw_geometries([pcd, lidar_coord, base_coord])

        # get ee pose
        T_ee = np.eye(4)
        R_ee = base_coord.get_rotation_matrix_from_quaternion(flange_ori_np)
        T_ee[:3, :3] = R_ee
        T_ee[:3, 3] = flange_pos_np
        T_l_e = np.matmul(T_b_inv, T_ee)
        flange_coord = copy.deepcopy(lidar_coord).transform(T_l_e)
        # o3d.visualization.draw_geometries([pcd, lidar_coord, base_coord, flange_coord],
        #                                   window_name=f'scene of env_{view_idx}')


        # print(f'env index: {view_idx}', np.unique(pcl_semantic))

        # 아래는 마지막 index만 가시화 (도구 가시화를 의도함)
        # index = np.unique(pcl_semantic)[-1]
        # print(f'show index: {index}\n')
        # semantic = np.where(pcl_semantic==index)[0]
        # pcd_semantic = o3d.geometry.PointCloud()
        # pcd_semantic.points = o3d.utility.Vector3dVector(pcl_reshape[semantic])
        # o3d.visualization.draw_geometries([pcd_semantic, lidar_coord, base_coord, flange_coord],
        #                                     window_name=f'semantic_{index} of env_{view_idx}')
        

        for i in np.unique(pcl_semantic):
            semantic = np.where(pcl_semantic==i)[0]
            pcd_semantic = o3d.geometry.PointCloud()
            pcd_semantic.points = o3d.utility.Vector3dVector(pcl_reshape[semantic])
            o3d.visualization.draw_geometries([pcd_semantic, lidar_coord, base_coord, flange_coord],
                                              window_name=f'semantic_{i} of env_{view_idx}')


    def get_observations(self) -> dict:
        ''' retrieve point cloud data from all render products '''
        # tasks/utils/pcd_writer.py 에서 pcd sample하고 tensor로 변환해서 가져옴
        pointcloud = self.pointcloud_listener.get_pointcloud_data()
        # TODO: pointcloud로부터 각 환경의 cube 위치를 가져와야 한다.

        '''
        아래 순서로 최종 obs_buf에 concat. 첫 차원은 환경 갯수
        1. robot dof position
        2. robot dof velocity
        3. flange position
        4. flange orientation
        5. target position => 이건 pointcloud_listener에서 받아와야 할듯? pcd 평균으로 하자
        6. goal position
        '''

        robot_dof_pos = self._robots.get_joint_positions(clone=False)[:, 0:6]   # get robot dof position from 1st to 6th joint
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)[:, 0:6]  # get robot dof velocity from 1st to 6th joint
        # rest of the joints are not used for control. They are fixed joints at each episode.

        flange_pos, flange_rot = self._flanges.get_local_poses()
        target_pos, target_rot = self._targets.get_local_poses()    # TODO: 물체의 pose는 명시적인 것이 아닌, pcd를 이용하도록 하자.
        goal_pos, goal_rot = self._goals.get_local_poses()


        if self.previous_target_position == None:
            self.target_moving_distance = torch.zeros((self._num_envs), device=self._device)
            self.current_target_position = target_pos
        else:
            self.previous_target_position = self.current_target_position
            self.current_target_position = target_pos
            self.target_moving_distance = LA.norm(self.previous_target_position - self.current_target_position,
                                                  ord=2, dim=1)

        # compute distance for calculate_metrics() and is_done()
        # TOOD: 기존에는 물체와 tool 사이의 거리를 계산했으나, point cloud에서는 좀 다른 방식을 생각해봐야 할 것 같다.
        # TODO: 아래에는 명시적인 것이 아닌 pcd의 평균을 이용하여 계산토록 하자.
        if self.previous_tool_target_distance == None:
            self.current_tool_target_distance = LA.norm(flange_pos - target_pos, ord=2, dim=1)
            self.previous_tool_target_distance = self.current_tool_target_distance
            self.current_target_goal_distance = LA.norm(target_pos - goal_pos, ord=2, dim=1)
            self.previous_target_goal_distance = self.current_target_goal_distance
        
        else:
            self.previous_tool_target_distance = self.current_tool_target_distance
            self.previous_target_goal_distance = self.current_target_goal_distance
            self.current_tool_target_distance = LA.norm(flange_pos - target_pos, ord=2, dim=1)
            self.current_target_goal_distance = LA.norm(target_pos - goal_pos, ord=2, dim=1)

        robot_body_dof_lower_limits = self.robot_dof_lower_limits[:6]
        robot_body_dof_upper_limits = self.robot_dof_upper_limits[:6]

        # the scaler below comes from the CabinetTask
        dof_pos_scaled = 2.0 * (robot_dof_pos - robot_body_dof_lower_limits) \
            / (robot_body_dof_upper_limits - robot_body_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        generalization_noise = torch.rand((dof_vel_scaled.shape[0], 6), device=self._device) + 0.5

        '''
        NE: number of environmet, N: number of points, F: feature dimension, 3: x, y, z
        pointcloud: [NE, N, 3]
        '''
        self.obs_buf = torch.cat((pointcloud.view([pointcloud.shape[0], -1]),   # [NE, N*3], point cloud
                                dof_pos_scaled,                               # [NE, 6]
                                dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
                                flange_pos,                                   # [NE, 3]
                                flange_rot,                                   # [NE, 4]
                                target_pos,                                   # [NE, 3]
                                goal_pos,                                     # [NE, 3]
                                ), dim=1)
        # self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        # # 위에 있는게 꼭 들어가야 할까??? 없어도 될 것 같은데....
        
        # self._env_pos is the position of the each environment. It comse from RLTask.

        if self._control_space == "cartesian":
            self.jacobians = self._robots.get_jacobians(clone=False)

        return {self._robots.name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        self._env.render()  # add for get point cloud on headless mode
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        if self._control_space == "joint":
            targets = self.robot_dof_targets[:, :6] + self.robot_dof_speed_scales[:6] * self.dt * self.actions * self._action_scale

        elif self._control_space == "cartesian":
            goal_position = self.flange_pos + actions / 100.0
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, self._robots.body_names.index(self._flange_link)-1, :, :],
                                                        current_position=self.flange_pos,
                                                        current_orientation=self.flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=None
                                                        )
            targets = self.robot_dof_targets[:, :6] + delta_dof_pos[:, :6]

        self.robot_dof_targets[:, :6] = torch.clamp(targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])
        self.robot_dof_targets[:, 6:] = torch.tensor(0, device=self._device, dtype=torch.float16)
        self.robot_dof_targets[:, 6] = torch.tensor(0.09, device=self._device)
        self.robot_dof_targets[:, 7] = torch.tensor(1.2, device=self._device)
        ### 나중에는 윗 줄을 통해 tool position이 random position으로 고정되도록 변수화. reset_idx도 확인할 것
        # self._robots.get_joint_positions()
        # self._robots.set_joint_positions(self.robot_dof_targets, indices=env_ids_int32)
        self._robots.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)
        # self._targets.enable_rigid_body_physics()
        # self._targets.enable_rigid_body_physics(indices=env_ids_int32)
        # self._targets.enable_gravities(indices=env_ids_int32)

    def reset_idx(self, env_ids) -> None:
        # episode 끝나고 env ids를 받아서 reset
        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        # pos = torch.clamp(self.robot_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self._device) - 0.5),
        #                   self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        added_pos = 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs-4), device=self._device) - 0.5)
        tool_pos = torch.zeros((len(env_ids), 4), device=self._device)
        ### 나중에는 윗 줄을 통해 tool position이 random position으로 고정되도록 변수화. pre_physics_step도 확인할 것
        pos = torch.clamp(self.robot_default_dof_pos.unsqueeze(0) + torch.column_stack((added_pos, tool_pos)),
                          self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        #########################
        pos[:, 6] = torch.tensor(0.09, device=self._device)
        pos[:, 7] = torch.tensor(1.2, device=self._device)
        #########################

        
        
        dof_pos = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        dof_pos[:, :] = pos
        dof_vel = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)

        # ##############################################################################################################
        # self.robot_dof_targets[:, 6:] = torch.tensor(0, device=self._device, dtype=torch.float16)
        # # self._robots.get_joint_positions()
        # ##############################################################################################################

        # reset lidar #####################
        # self._lidar.add_depth_data_to_frame()
        # self._lidar.add_point_cloud_data_to_frame()
        # self._lidar.enable_visualization(high_lod=True,
        #                                   draw_points=True,
        #                                   draw_lines=False)
        # self._lidar.initialize()

        # reset target
        position = torch.tensor([0.50, -0.2, 0.15], device=self._device)
        target_pos = position.repeat(len(env_ids),1)
        # target_pos = (torch.rand((len(env_ids), 3), device=self._device) - 0.5) * 2 \
        #             * torch.tensor([0.25, 0.25, 0.10], device=self._device) \
        #             + torch.tensor([0.50, 0.00, 0.20], device=self._device)

        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        target_ori = orientation.repeat(len(env_ids),1)
        self._targets.set_world_poses(target_pos + self._env_pos[env_ids],
                                      target_ori,
                                      indices=indices)
        # self._targets.enable_rigid_body_physics()

        # reset goal
        # goal_pos_xy_variation = torch.rand((len(env_ids), 2), device=self._device) * 0.1
        # goal_pos_z_variation = torch.mul(torch.ones((len(env_ids), 1), device=self._device), 0.025)
        # goal_pos_variation = torch.cat((goal_pos_xy_variation, goal_pos_z_variation), dim=1)
        # goal_pos = torch.tensor(self._goal_position, device=self._device) + goal_pos_variation
        goal_pos = torch.tensor(self._goal_position, device=self._device)
        self._goals.set_world_poses(goal_pos + self._env_pos[env_ids], indices=indices)

        # TODO: 여기에 물체를 집는 task를 추가???

        # reset dummy
        # self._dummy.set_world_poses()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_robot_dofs = self._robots.num_dof
        self.robot_dof_pos = torch.zeros((self.num_envs, self.num_robot_dofs), device=self._device)
        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)
        # self._targets.enable_rigid_body_physics()
        # self._targets.enable_gravities()



        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # self.rew_buf[:] = -self._computed_tool_target_distance - self._computed_target_goal_distance
        tool_target_distance_reward = 1 / (self.current_tool_target_distance + 1e-5)
        target_goal_distance_reward = 1 / (self.current_target_goal_distance + 1e-5)

        delta_tool_target = self.previous_tool_target_distance - self.current_tool_target_distance
        delta_target_goal = self.previous_target_goal_distance - self.current_target_goal_distance

        avoiding_non_touch = torch.abs(1 / (self.target_moving_distance + 0.5))

        self.completion_reward = torch.zeros(self._num_envs).to(self._device)
        self.completion_reward[self.current_target_goal_distance<0.035] = 1
        self.rew_buf[:] = self.alpha * tool_target_distance_reward + \
                          self.beta * target_goal_distance_reward + \
                          self.gamma * self.completion_reward + \
                          self.zeta * delta_tool_target + \
                          self.eta * delta_target_goal + \
                          self.punishment * avoiding_non_touch        
        
        # 시간이 지나도 target이 움직이지 않으면 minus reward를 주어야 할 듯

        '''
        reward = alpha * (previous_distance_to_goal - current_distance_to_goal) +
                 beta * (previous_distance_to_object - current_distance_to_object) +
                 gamma * completion_reward
        '''

    def is_done(self) -> None:
        self.reset_buf.fill_(0)
        # target reached
        # self.reset_buf = torch.where(self._computed_target_goal_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.current_target_goal_distance <= 0.025, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

