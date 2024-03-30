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
from omni.isaac.core.objects import DynamicCylinder, DynamicCone, DynamicSphere, DynamicCuboid
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.materials.physics_material import PhysicsMaterial

from skrl.utils import omniverse_isaacgym_utils
from pxr import Usd, UsdGeom, Gf, UsdPhysics, Semantics # pxr usd imports used to create cube

from typing import Optional, Tuple

import open3d as o3d
import open3d.core as o3c
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

import copy

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()    

class PCDMovingTargetTask(RLTask):
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
        
        self.initial_target_goal_distance = torch.empty(self._num_envs).to(self.cfg["rl_device"])
        
        self.alpha = 0.15
        self.beta = 0.5
        self.gamma = 0
        self.zeta = 0.1
        self.eta = 0.25
        self.relu = torch.nn.ReLU()

        self.stage = omni.usd.get_context().get_stage()

        # tool orientation
        self.tool_rot_x = 1.221 # 70 degree
        self.tool_rot_z = 0     # 0 degree
        self.tool_rot_y = -1.5707 # -90 degree

        self.target_height = 0.05

        # workspace 2D boundary
        # self.x_min = 0.45
        # self.x_max = 1.2
        # self.y_min = -0.8
        # self.y_max = 0.4
        self.x_min = 0.3
        self.x_max = 0.9
        self.y_min = -0.6
        self.y_max = 0.6
        self.z_min = 0.1
        self.z_max = 0.7

        self._pcd_sampling_num = self._task_cfg["sim"]["point_cloud_samples"]
        self._num_pcd_masks = 2 # tool and target
        # TODO: point cloud의 sub mask 개수를 state 또는 config에서 받아올 것.

        # get tool point cloud from ply and convert it to torch tensor
        device = torch.device(self.cfg["rl_device"])
        tool_o3d_pcd = o3d.io.read_point_cloud("/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/tool.ply")
        o3d_downsampled_pcd = tool_o3d_pcd.farthest_point_down_sample(self._pcd_sampling_num)
        downsampled_points = np.array(o3d_downsampled_pcd.points)
        tool_pcd = torch.from_numpy(downsampled_points).to(device)
        self.tool_pcd = tool_pcd.unsqueeze(0).repeat(self._num_envs, 1, 1)
        self.tool_pcd = self.tool_pcd.float()

        target_o3d_pcd = o3d.io.read_point_cloud("/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/cylinder/cylinder.ply")
        # rotate with x-axis 90 degree
        R = target_o3d_pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        target_o3d_pcd.rotate(R, center=(0, 0, 0))
        o3d_downsampled_pcd = target_o3d_pcd.farthest_point_down_sample(self._pcd_sampling_num)
        downsampled_points = np.array(o3d_downsampled_pcd.points)
        target_pcd = torch.from_numpy(downsampled_points).to(device)
        self.target_pcd = target_pcd.unsqueeze(0).repeat(self._num_envs, 1, 1)
        self.target_pcd = self.target_pcd.float()

        # observation and action space
        pcd_observations = self._pcd_sampling_num * self._num_pcd_masks * 3     # 2 is a number of point cloud masks and 3 is a cartesian coordinate
        self._num_observations = pcd_observations + 6 + 6 + 3 + 4 + 2
        '''
        refer to observations in get_observations()
        pcd_observations                              # [NE, 3*2*pcd_sampling_num]
        dof_pos_scaled,                               # [NE, 6]
        dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
        flange_pos,                                   # [NE, 3]
        flange_rot,                                   # [NE, 4]
        goal_pos_xy,                                  # [NE, 2]
        
        '''

        if self._control_space == "joint":
            self._num_actions = 6
        elif self._control_space == "cartesian":
            self._num_actions = 7   # 3 for position, 4 for rotation(quaternion)
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
        self._goal_mark = self._task_cfg["env"]["goal"]
        self._target_position = self._task_cfg["env"]["target"]
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
        # tool view
        self._tools = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/tool", name="tool_view")        
        # target view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)        
        # goal view
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)
        self._goals._non_root_link = True   # do not set states for kinematics

        scene.add(self._robots)
        scene.add(self._flanges)
        scene.add(self._tools)
        scene.add(self._targets)
        scene.add(self._goals)
        
        
        # # point cloud view
        # self.render_products = []
        # camera_positions = {0: [1.5, 1.5, 0.5],
        #                     1: [2, -1.3, 0.5],
        #                     2: [-0.5, -1.2, 0.5]}
        #                     # 2: [-1.5, -2.2, 0.5]}
        #                     # 2: [-4.8, -6.1, 0.5]}
        # camera_rotations = {0: [0, -10, 50],
        #                     1: [0, -10, -45],
        #                     2: [0, -10, -130]}
        # env_pos = self._env_pos.cpu()

        # # Used to get depth data from the camera
        # for i in range(self._num_envs):
        #     self.rep.get.camera()
        #     # Get the parameters of the camera via links below
        #     # https://learn.microsoft.com/en-us/answers/questions/201906/pixel-size-of-rgb-and-tof-camera#:~:text=Pixel%20Size%20for,is%20~2.3%20mm
        #     # https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html#calibrated-camera-sensors
        #     '''
        #     Azure Kinect DK
        #     [Depth Camera]
        #     focal_length = 0.18 (cm)
        #     focus_distance = 3.86 (m) 그런데 clipping range가 있는데 이게 어떻게 작용할지 불확실함
        #     horizontal_aperture = 0.224 (cm)
        #     vertical_aperture = 0.2016 (cm)
        #     clipping_range = 0.5 ~ 3.86 (m)
        #     위 값을 RGB로 가시화 하면 이상하긴 한데, depth camera 니까 우선 해보자.
            
        #     여기 보면 또 mm임..
        #     https://docs.omniverse.nvidia.com/isaacsim/latest/manual_replicator_composer_parameter_list.html#camera-lens
            

        #     [RGB Camera]
        #     focal_length = 0.23 (cm)
        #     focus_distance = 5 (m) 그런데 clipping range가 있는데 이게 어떻게 작용할지 불확실함
        #     horizontal_aperture = 0.24 (cm)
        #     vertical_aperture = 0.135 (cm)
        #     clipping_range = 0.01 ~ 10 (m)
        #     '''
        #     # pos: (2 -1.3 0.5), ori: (0, -10, -45)
        #     # pos: (-1.5 -2.2 0.5), ori: (0, -10, -130)
        #     # 

        #     for j in range(3):
        #         locals()[f"camera_{j}"] = self.rep.create.camera(
        #                                                          position = (env_pos[i][0] + camera_positions[j][0],
        #                                                                      env_pos[i][1] + camera_positions[j][1],
        #                                                                      env_pos[i][2] + camera_positions[j][2]),
        #                                                          rotation=(camera_rotations[j][0],
        #                                                                    camera_rotations[j][1],
        #                                                                    camera_rotations[j][2]), 
        #                                                          focal_length=18.0,
        #                                                          focus_distance=400,
        #                                                          horizontal_aperture=20.955,
        #                                                          # vertical_aperture=0.2016,
        #                                                          # clipping_range=(0.5, 3.86),
        #                                                          clipping_range=(0.01, 3),
        #                                                          # TODO: clipping range 조절해서 환경이 서로 안 겹치게 하자.
        #                                                          )
                
        #         render_product = self.rep.create.render_product(locals()[f"camera_{j}"], resolution=(self.camera_width, self.camera_height))
        #         self.render_products.append(render_product)

        # # start replicator to capture image data
        # self.rep.orchestrator._orchestrator._is_started = True

        # # initialize pytorch writer for vectorized collection
        # self.pointcloud_listener = self.PointcloudListener()
        # self.pointcloud_writer = self.rep.WriterRegistry.get("PointcloudWriter")
        # self.pointcloud_writer.initialize(listener=self.pointcloud_listener,
        #                                   pcd_sampling_num=self._pcd_sampling_num,
        #                                   pcd_normalize = self._pcd_normalization,
        #                                   env_pos = self._env_pos.cpu(),
        #                                   camera_positions=camera_positions,
        #                                   camera_orientations=camera_rotations,
        #                                   device=self.device,
        #                                   )
        # self.pointcloud_writer.attach(self.render_products)
            
        # # get robot semantic data
        # # 그런데 어짜피 로봇 point cloud는 필요 없기 때문에 안 받아도 될듯
        # self._robot_semantics = {}
        # for i in range(self._num_envs):
        #     robot_prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/robot")
        #     self._robot_semantics[i] = Semantics.SemanticsAPI.Apply(robot_prim, "Semantics")
        #     self._robot_semantics[i].CreateSemanticTypeAttr()
        #     self._robot_semantics[i].CreateSemanticDataAttr()
        #     self._robot_semantics[i].GetSemanticTypeAttr().Set("class")
        #     self._robot_semantics[i].GetSemanticDataAttr().Set(f"robot_{i}")
        #     add_update_semantics(robot_prim, '0')

        # # get tool semantic data
        # self._tool_semantics = {}
        # for i in range(self._num_envs):
        #     tool_prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/robot/tool")
        #     self._tool_semantics[i] = Semantics.SemanticsAPI.Apply(tool_prim, "Semantics")
        #     self._tool_semantics[i].CreateSemanticTypeAttr()
        #     self._tool_semantics[i].CreateSemanticDataAttr()
        #     self._tool_semantics[i].GetSemanticTypeAttr().Set("class")
        #     self._tool_semantics[i].GetSemanticDataAttr().Set(f"tool_{i}")
        #     add_update_semantics(tool_prim, '1')    # added for fixing the order of semantic index

        # # get target object semantic data
        # self._target_semantics = {}
        # for i in range(self._num_envs):
        #     target_prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/target")
        #     self._target_semantics[i] = Semantics.SemanticsAPI.Apply(target_prim, "Semantics")
        #     self._target_semantics[i].CreateSemanticTypeAttr()
        #     self._target_semantics[i].CreateSemanticDataAttr()
        #     self._target_semantics[i].GetSemanticTypeAttr().Set("class")
        #     self._target_semantics[i].GetSemanticDataAttr().Set(f"target_{i}")
        #     add_update_semantics(target_prim, '2')  # added for fixing the order of semantic index

        self.init_data()


    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exist("robot_view"):
            scene.remove("robot_view", registry_only=True)
        if scene.object_exist("end_effector_view"):
            scene.remove("end_effector_view", registry_only=True)
        if scene.object_exist("tool_view"):
            scene.remove("tool_view", registry_only=True)
        if scene.object_exist("target_view"):
            scene.remove("target_view", registry_only=True)
        if scene.object_exist("goal_view"):
            scene.remove("goal_view", registry_only=True)

        self._robots = UR5eView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        self._flanges = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/{self._flange_link}", name="end_effector_view")
        self._tools = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/tool", name="tool_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)

        scene.add(self._robots)
        scene.add(self._flanges)
        scene.add(self._tools)
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
        target = DynamicCylinder(prim_path=self.default_zero_env_path + "/target",
                                 name="target",
                                 radius=0.04,
                                 height=self.target_height,
                                 density=1,
                                 color=torch.tensor([255, 0, 0]),
                                 mass=1,
                                #  physics_material=PhysicsMaterial(
                                #                                   prim_path="/World/physics_materials/target_material",
                                #                                   static_friction=0.05, dynamic_friction=0.6)
                                 physics_material=PhysicsMaterial(
                                                                  prim_path="/World/physics_materials/target_material",
                                                                  static_friction=0.02, dynamic_friction=0.3)
                                 )
        
        self._sim_config.apply_articulation_settings("target",
                                                     get_prim_at_path(target.prim_path),
                                                     self._sim_config.parse_actor_config("target"))

    def get_goal(self):
        goal = DynamicCone(prim_path=self.default_zero_env_path + "/goal",
                          name="goal",
                          radius=0.015,
                          height=0.03,
                          color=torch.tensor([0, 255, 0]))
        self._sim_config.apply_articulation_settings("goal",
                                                     get_prim_at_path(goal.prim_path),
                                                     self._sim_config.parse_actor_config("goal"))
        goal.set_collision_enabled(False)
        

    def init_data(self) -> None:
        # self.robot_default_dof_pos = torch.tensor(np.radians([-40, -45, 60, -100, -90, 90.0,
        #                                                       0.0, 0.0, 0.0, 0.0]), device=self._device, dtype=torch.float32)
        self.robot_default_dof_pos = torch.tensor(np.radians([-50, -40, 50, -100, -90, 130,
                                                              5, 70, 0.0, -90]), device=self._device, dtype=torch.float32)

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


    # def visualize_point_cloud(self, view_idx, lidar_position):
    #     '''
    #     args:
    #         view_idx: index of the cloner
    #         lidar_position: position of the lidar
    #     '''
    #     flange_pos, flange_rot = self._flanges.get_local_poses()

    #     lidar_prim_path = self._point_cloud[view_idx].prim_path
    #     point_cloud = self._point_cloud[view_idx]._lidar_sensor_interface.get_point_cloud_data(lidar_prim_path)
    #     semantic = self._point_cloud[view_idx]._lidar_sensor_interface.get_semantic_data(lidar_prim_path)

    #     pcl_reshape = np.reshape(point_cloud, (point_cloud.shape[0]*point_cloud.shape[1], 3))
    #     flange_pos_np = flange_pos[view_idx].cpu().numpy()
    #     flange_ori_np = flange_rot[view_idx].cpu().numpy()

    #     pcl_semantic = np.reshape(semantic, -1)        


    #     v3d = o3d.utility.Vector3dVector

    #     # get point cloud
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = v3d(pcl_reshape)

    #     # get sampled point cloud
    #     idx = pcu.downsample_point_cloud_poisson_disk(pcl_reshape, num_samples=int(0.2*pcl_reshape.shape[0]))
    #     pcl_reshape_sampled = pcl_reshape[idx]
    #     sampled_pcd = o3d.geometry.PointCloud()
    #     sampled_pcd.points = v3d(pcl_reshape_sampled)

    #     # get lidar frame. lidar frame is a world [0, 0, 0] frame
    #     lidar_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0.0, 0.0, 0.0]))

    #     # get base pose    
    #     T_b = np.eye(4)
    #     # TODO: get rotation matrix from USD
    #     # R_b = lidar_coord.get_rotation_matrix_from_xyz((np.pi/6, 0, -np.pi/2))
    #     R_b = lidar_coord.get_rotation_matrix_from_xyz((np.pi/9, 0, -np.pi/2))  # rotation relationship between lidar and base
    #     T_b[:3, :3] = R_b
    #     T_b[:3, 3] = lidar_position # acquired position from prim_path is [0,0,0]
    #     T_b_inv = np.linalg.inv(T_b)
    #     base_coord = copy.deepcopy(lidar_coord).transform(T_b_inv)
    #     # o3d.visualization.draw_geometries([pcd, lidar_coord, base_coord])

    #     # get ee pose
    #     T_ee = np.eye(4)
    #     R_ee = base_coord.get_rotation_matrix_from_quaternion(flange_ori_np)
    #     T_ee[:3, :3] = R_ee
    #     T_ee[:3, 3] = flange_pos_np
    #     T_l_e = np.matmul(T_b_inv, T_ee)
    #     flange_coord = copy.deepcopy(lidar_coord).transform(T_l_e)
    #     # o3d.visualization.draw_geometries([pcd, lidar_coord, base_coord, flange_coord],
    #     #                                   window_name=f'scene of env_{view_idx}')


    #     # print(f'env index: {view_idx}', np.unique(pcl_semantic))

    #     # 아래는 마지막 index만 가시화 (도구 가시화를 의도함)
    #     # index = np.unique(pcl_semantic)[-1]
    #     # print(f'show index: {index}\n')
    #     # semantic = np.where(pcl_semantic==index)[0]
    #     # pcd_semantic = o3d.geometry.PointCloud()
    #     # pcd_semantic.points = o3d.utility.Vector3dVector(pcl_reshape[semantic])
    #     # o3d.visualization.draw_geometries([pcd_semantic, lidar_coord, base_coord, flange_coord],
    #     #                                     window_name=f'semantic_{index} of env_{view_idx}')
        

    #     for i in np.unique(pcl_semantic):
    #         semantic = np.where(pcl_semantic==i)[0]
    #         pcd_semantic = o3d.geometry.PointCloud()
    #         pcd_semantic.points = o3d.utility.Vector3dVector(pcl_reshape[semantic])
    #         o3d.visualization.draw_geometries([pcd_semantic, lidar_coord, base_coord, flange_coord],
    #                                           window_name=f'semantic_{i} of env_{view_idx}')


    def get_observations(self) -> dict:
        ''' retrieve point cloud data from all render products '''
        # tasks/utils/pcd_writer.py 에서 pcd sample하고 tensor로 변환해서 가져옴
        # pointcloud = self.pointcloud_listener.get_pointcloud_data()

        self.flange_pos, self.flange_rot = self._flanges.get_local_poses()
        # self.goal_pos, _ = self._goals.get_local_poses()
        # self.goal_pos_xy = self.goal_pos[:, [0, 1]]
        target_pos, target_rot_quaternion = self._targets.get_local_poses()
        target_rot = quaternion_to_matrix(target_rot_quaternion)
        tool_pos, tool_rot_quaternion = self._tools.get_local_poses()
        tool_rot = quaternion_to_matrix(tool_rot_quaternion)      

        # get transformation matrix from base to tool
        T_base_to_tool = torch.eye(4, device=self._device).unsqueeze(0).repeat(self._num_envs, 1, 1)
        T_base_to_tool[:, :3, :3] = tool_rot.clone().detach()
        T_base_to_tool[:, :3, 3] = tool_pos.clone().detach()

        ##### point cloud registration for tool #####
        B, N, _ = self.tool_pcd.shape
        # Convert points to homogeneous coordinates by adding a dimension with ones
        homogeneous_points = torch.cat([self.tool_pcd, torch.ones(B, N, 1, device=self.tool_pcd.device)], dim=-1)
        # Perform batch matrix multiplication
        transformed_points_homogeneous = torch.bmm(homogeneous_points, T_base_to_tool.transpose(1, 2))
        # Convert back from homogeneous coordinates by removing the last dimension
        tool_pcd_transformed = transformed_points_homogeneous[..., :3]
        ##### point cloud registration for tool #####


        # get transformation matrix from base to target object
        T_base_to_tgt = torch.eye(4, device=self._device).unsqueeze(0).repeat(self._num_envs, 1, 1)
        T_base_to_tgt[:, :3, :3] = target_rot.clone().detach()
        T_base_to_tgt[:, :3, 3] = target_pos.clone().detach()

        ##### point cloud registration for tool #####
        B, N, _ = self.target_pcd.shape
        # Convert points to homogeneous coordinates by adding a dimension with ones
        homogeneous_points = torch.cat([self.target_pcd, torch.ones(B, N, 1, device=self.target_pcd.device)], dim=-1)
        # Perform batch matrix multiplication
        transformed_points_homogeneous = torch.bmm(homogeneous_points, T_base_to_tgt.transpose(1, 2))
        # Convert back from homogeneous coordinates by removing the last dimension
        target_pcd_transformed = transformed_points_homogeneous[..., :3]
        ##### point cloud registration for tool #####


        ##################### ####################################
        ################# visualize point cloud #################
        # view_idx = 0

        # base_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))
        # tool_pos_np = tool_pos[view_idx].cpu().numpy()
        # tool_rot_np = tool_rot[view_idx].cpu().numpy()
        # tgt_pos_np = target_pos[view_idx].cpu().numpy()
        # tgt_rot_np = target_rot[view_idx].cpu().numpy()
        
        # tool_transformed_pcd_np = tool_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
        # tool_transformed_point_cloud = o3d.geometry.PointCloud()
        # tool_transformed_point_cloud.points = o3d.utility.Vector3dVector(tool_transformed_pcd_np)
        # T_t = np.eye(4)
        # T_t[:3, :3] = tool_rot_np
        # T_t[:3, 3] = tool_pos_np
        # tool_coord = copy.deepcopy(base_coord).transform(T_t)

        # tgt_transformed_pcd_np = target_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
        # tgt_transformed_point_cloud = o3d.geometry.PointCloud()
        # tgt_transformed_point_cloud.points = o3d.utility.Vector3dVector(tgt_transformed_pcd_np)
        # T_o = np.eye(4)

        # # R_b = tgt_rot_np.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        # T_o[:3, :3] = tgt_rot_np
        # # T_o[:3, :3] = R_b
        # T_o[:3, 3] = tgt_pos_np
        # tgt_coord = copy.deepcopy(base_coord).transform(T_o)

        # goal_pos_np = self.goal_pos[view_idx].cpu().numpy()
        # goal_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.01, height=0.03)
        # goal_cone.paint_uniform_color([0, 1, 0])
        # T_g_p = np.eye(4)
        # T_g_p[:3, 3] = goal_pos_np
        # goal_position = copy.deepcopy(goal_cone).transform(T_g_p)

        # o3d.visualization.draw_geometries([base_coord,
        #                                    tool_transformed_point_cloud,
        #                                    tgt_transformed_point_cloud,
        #                                    tool_coord,
        #                                    tgt_coord,
        #                                    goal_position,],
        #                                     window_name=f'point cloud')
        ################# visualize point cloud #################
        #########################################################



        target_pos_mean = torch.mean(target_pcd_transformed, dim=1)
        self.target_pos_xy = target_pos_mean[:, [0, 1]]


        # ##### farthest point from the tool to the goal #####
        # # calculate farthest distance and idx from the tool to the goal
        # diff = points_transformed - self.flange_pos[:, None, :]
        # distance = diff.norm(dim=2)  # [B, N]

        # # Find the index and value of the farthest point from the base coordinate
        # farthest_idx = distance.argmax(dim=1)  # [B]
        # # farthest_val = distance.gather(1, farthest_idx.unsqueeze(1)).squeeze(1)  # [B]
        # self.tool_end_point = points_transformed.gather(1, farthest_idx.view(B, 1, 1).expand(B, 1, 3)).squeeze(1).squeeze(1)  # [B, 3]
        # ##### farthest point from the tool to the goal #####

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


        # if self.previous_target_position == None:
        #     self.target_moving_distance = torch.zeros((self._num_envs), device=self._device)
        #     self.current_target_position = target_pos
        # else:
        #     self.previous_target_position = self.current_target_position
        #     self.current_target_position = target_pos
        #     self.target_moving_distance = LA.norm(self.previous_target_position - self.current_target_position,
        #                                           ord=2, dim=1)

        # # compute distance for calculate_metrics() and is_done()
        # if self.previous_tool_target_distance == None:
        #     self.current_tool_target_distance = LA.norm(flange_pos - target_pos, ord=2, dim=1)
        #     self.previous_tool_target_distance = self.current_tool_target_distance
        #     self.current_target_goal_distance = LA.norm(target_pos - goal_pos, ord=2, dim=1)
        #     self.previous_target_goal_distance = self.current_target_goal_distance
        
        # else:
        #     self.previous_tool_target_distance = self.current_tool_target_distance
        #     self.previous_target_goal_distance = self.current_target_goal_distance
        #     self.current_tool_target_distance = LA.norm(flange_pos - target_pos, ord=2, dim=1)
        #     self.current_target_goal_distance = LA.norm(target_pos - goal_pos, ord=2, dim=1)


        robot_body_dof_lower_limits = self.robot_dof_lower_limits[:6]
        robot_body_dof_upper_limits = self.robot_dof_upper_limits[:6]

        # # normalize robot_dof_pos
        # dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0   # normalized by [-1, 1]
        # dof_pos_scaled = (robot_dof_pos - self.robot_dof_lower_limits) \
        #                 /(self.robot_dof_upper_limits - self.robot_dof_lower_limits)    # normalized by [0, 1]
        dof_pos_scaled = robot_dof_pos    # non-normalized

        
        # # normalize robot_dof_vel
        # dof_vel_scaled = robot_dof_vel * self._dof_vel_scale
        # generalization_noise = torch.rand((dof_vel_scaled.shape[0], 6), device=self._device) + 0.5
        dof_vel_scaled = robot_dof_vel    # non-normalized

        self.obs_buf = torch.cat((
                                #   tool_pcd_transformed.reshape([tool_pcd_transformed.shape[0], -1]),     # [NE, N*3], point cloud
                                #   target_pcd_transformed.reshape([target_pcd_transformed.shape[0], -1]), # [NE, N*3], point cloud
                                  tool_pcd_transformed.contiguous().view(self._num_envs, -1),
                                  target_pcd_transformed.contiguous().view(self._num_envs, -1),
                                  dof_pos_scaled,                               # [NE, 6]
                                #   dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
                                  dof_vel_scaled[:, :6],                        # [NE, 6]
                                  self.flange_pos,                                   # [NE, 3]
                                  self.flange_rot,                                   # [NE, 4]
                                  self.goal_pos_xy,                                  # [NE, 2]
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
            goal_position = self.flange_pos + self.actions[:, :3] / 50.0
            goal_orientation = self.flange_rot + self.actions[:, 3:] / 70.0
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, self._robots.body_names.index(self._flange_link)-1, :, :],
                                                        current_position=self.flange_pos,
                                                        current_orientation=self.flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=None
                                                        )
            targets = self.robot_dof_targets[:, :6] + delta_dof_pos[:, :6]

        self.robot_dof_targets[:, :6] = torch.clamp(targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])
        self.robot_dof_targets[:, 7] = torch.tensor(self.tool_rot_x, device=self._device)
        self.robot_dof_targets[:, 8] = torch.tensor(self.tool_rot_z, device=self._device)
        self.robot_dof_targets[:, 9] = torch.tensor(self.tool_rot_y, device=self._device)
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
        pos = self.robot_default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)   # non-randomized
        # ##### randomize robot pose #####
        # randomize_manipulator_pos = 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs-4), device=self._device) - 0.5)
        # # tool_pos = torch.zeros((len(env_ids), 4), device=self._device)  # 여기에 rand를 추가
        # ### TODO: 나중에는 윗 줄을 통해 tool position이 random position으로 고정되도록 변수화. pre_physics_step도 확인할 것
        # pos[:, 0:6] = pos[:, 0:6] + randomize_manipulator_pos
        # ##### randomize robot pose #####        
        
        dof_pos = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        dof_pos[:, :] = pos
        dof_vel = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)

        # reset target
        position = torch.tensor(self._target_position, device=self._device)
        target_pos = position.repeat(len(env_ids),1)
        # ### randomize target position ###
        # # x, y randomize 는 ±0.1로 uniform random
        # # z_ref = torch.abs(target_pos[2])
        # z_ref = torch.unsqueeze(torch.abs(target_pos[:, 2]),1)
        # # generate uniform random values for randomizing target position
        # # reference: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        # x_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # y_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # # z_rand = z_ref.repeat(len(env_ids),1)   ### Do not randomize z position
        # rand = torch.cat((x_rand, y_rand, z_ref), dim=1)    ### Do not randomize z position
        # target_pos = target_pos + rand
        # ### randomize target position ###

        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        target_ori = orientation.repeat(len(env_ids),1)
        self._targets.set_world_poses(target_pos + self._env_pos[env_ids],
                                      target_ori,
                                      indices=indices)
        # self._targets.enable_rigid_body_physics()

        # reset goal
        goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)
        goal_mark_pos = goal_mark_pos.repeat(len(env_ids),1)
        # ### randomize goal position ###
        # # x, y randomize 는 ±0.1로 uniform random
        # # z_ref = torch.abs(goal_mark_pos[2])
        # z_ref = torch.unsqueeze(torch.abs(goal_mark_pos[:, 2]),1)
        # # # generate uniform random values for randomizing goal position
        # x_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # y_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # # z_rand = z_ref.repeat(len(env_ids),1)   ### Do not randomize z position
        # rand = torch.cat((x_rand, y_rand, z_ref), dim=1)    ### Do not randomize z position
        # goal_mark_pos = goal_mark_pos + rand
        # ### randomize goal position ###
        self._goals.set_world_poses(goal_mark_pos + self._env_pos[env_ids], indices=indices)
        goal_pos = self._goals.get_local_poses()
        self.goal_pos_xy = goal_pos[0][:, [0, 1]]


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
        initialized_idx = self.progress_buf == 1
        self.current_target_goal_distance = LA.norm(self.goal_pos_xy - self.target_pos_xy, ord=2, dim=1)
        self.initial_target_goal_distance[initialized_idx] = self.current_target_goal_distance[initialized_idx]

        init_t_g_d = self.initial_target_goal_distance
        cur_t_g_d = self.current_target_goal_distance
        target_goal_distance_reward = self.relu(-(cur_t_g_d - init_t_g_d)/init_t_g_d)

        # self.completion_reward = torch.zeros(self._num_envs).to(self._device)
        # self.completion_reward[self.current_tool_goal_distance <= 0.05] = 10.0
        # self.rew_buf[:] = tool_goal_distance_reward + self.completion_reward
        self.rew_buf[:] = target_goal_distance_reward    
        
        # 시간이 지나도 target이 움직이지 않으면 minus reward를 주어야 할 듯

        '''
        reward = alpha * (previous_distance_to_goal - current_distance_to_goal) +
                 beta * (previous_distance_to_object - current_distance_to_object) +
                 gamma * completion_reward
        '''

    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        reset = torch.zeros_like(self.reset_buf)

        # # workspace regularization
        reset = torch.where(self.flange_pos[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.flange_pos[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.flange_pos[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.flange_pos[:, 1] > self.y_max, ones, reset)
        reset = torch.where(self.flange_pos[:, 2] > 0.5, ones, reset)
        reset = torch.where(self.target_pos_xy[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.target_pos_xy[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.target_pos_xy[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.target_pos_xy[:, 1] > self.y_max, ones, reset)
        # reset = torch.where(self.target_pos_xy[:, 2] > 0.5, ones, reset)

        # target reached
        reset = torch.where(self.current_target_goal_distance < 0.05, ones, reset)

        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, reset)

