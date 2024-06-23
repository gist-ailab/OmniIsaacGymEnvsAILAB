import torch
from torch import linalg as LA
import numpy as np
import cv2
from gym import spaces
from copy import deepcopy
 
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")   # required for PytorchListener

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface
import omni.replicator.core as rep
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ur5e_tool import UR5eTool
from omniisaacgymenvs.robots.articulations.views.ur5e_view import UR5eView
from omniisaacgymenvs.tasks.utils.pcd_writer import PointcloudWriter
from omniisaacgymenvs.tasks.utils.pcd_listener import PointcloudListener
# from omniisaacgymenvs.ur_ikfast.ur_ikfast import ur_kinematics
from omni.isaac.core.utils.semantics import add_update_semantics

import omni
from omni.isaac.core.prims import RigidPrimView, RigidContactView
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
        
        self.initial_tool_approach_distance = torch.empty(self._num_envs).to(self.cfg["rl_device"])
        self.initial_tool_target_distance = torch.empty(self._num_envs).to(self.cfg["rl_device"])
        self.initial_target_goal_distance = torch.empty(self._num_envs).to(self.cfg["rl_device"])
        self.completion_reward = torch.zeros(self._num_envs).to(self.cfg["rl_device"])

        self.alpha = 0.1
        self.beta = 0.3
        self.gamma = 1
        
        self.relu = torch.nn.ReLU()

        # self.stage1_satisfied = torch.ones(self._num_envs).to(self.cfg["rl_device"])
        self.stage1_satisfied = torch.zeros(self._num_envs, dtype=torch.bool).to(self.cfg["rl_device"])
        self.stage2_satisfied = torch.zeros(self._num_envs, dtype=torch.bool).to(self.cfg["rl_device"])
        self.stage_buf = torch.zeros(self._num_envs, dtype=int).to(self.cfg["rl_device"])
        self.previous_tool_approach_distance = None
        

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
        self.x_min = 0.2
        self.x_max = 1.2
        self.y_min = -0.7
        self.y_max = 0.7
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

        self._flange_link = "flange"

        # ############################################################ BSH
        # # use multi-dimensional observation for camera RGB => 이거 쓰면 오류남
        # self.observation_space = spaces.Box(
        #     np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
        #     np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)
        # ############################################################ BSH
        
        self.PointcloudWriter = PointcloudWriter
        self.PointcloudListener = PointcloudListener

        # Solving I.K. with cuRobo
        self.init_cuRobo()
        

        RLTask.__init__(self, name, env)

    
    def init_cuRobo(self):
        # Solving I.K. with cuRobo
        tensor_args = TensorDeviceType()
        robot_config_file = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
        robot_config = robot_config_file["robot_cfg"]
        # urdf_file = config_file["robot_cfg"]["kinematics"][
        #     "urdf_path"
        # ]  # Send global path starting with "/"
        collision_file = "/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/collision_bar.yml"
        
        # base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        # ee_link = 'flange'
        # ee_link = 'tool0'   # 기본 제공 파일엔 끝 link가 tool0으로 되어 있음
        # robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
        world_cfg = WorldConfig.from_dict(load_yaml(collision_file))

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            world_cfg,
            # None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)


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
        self.get_cube()
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
        # self._cubes = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=True)
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
        # self._cubes = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=True)
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)

        scene.add(self._robots)
        scene.add(self._flanges)
        scene.add(self._tools)
        scene.add(self._targets)
        scene.add(self._goals)
        
        self.init_data()

    def get_robot(self):
        self.robot = UR5eTool(prim_path=self.default_zero_env_path + "/robot",
                         translation=torch.tensor([0.0, 0.0, 0.0]),
                         orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                         name="robot")
        self._sim_config.apply_articulation_settings("robot",
                                                     get_prim_at_path(self.robot.prim_path),
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


    def get_cube(self):
        cube = DynamicCuboid(prim_path=self.default_zero_env_path + "/cube",
                                 name="cube",
                                 size=0.04,
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
        self._sim_config.apply_articulation_settings("cube",
                                                     get_prim_at_path(cube.prim_path),
                                                     self._sim_config.parse_actor_config("cube"))

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
        # self.robot_default_dof_pos = torch.tensor(np.radians([-50, -40, 50, -100, -90, 130,
        #                                                       5, 70, 0.0, -90]), device=self._device, dtype=torch.float32)
        self.robot_default_dof_pos = torch.tensor(np.radians([-60, -70, 90, -110, -90, 130,
                                                              5, 70, 0.0, -90, 0]), device=self._device, dtype=torch.float32)

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        if self._control_space == "cartesian":
            # self.jacobians = torch.zeros((self._num_envs, 11, 6, 6), device=self._device)
            if self._robots.body_names == None:
                # self.jacobians = torch.zeros((self._num_envs, 11, 6, 6), device=self._device)
                self.jacobians = torch.zeros((self._num_envs, 15, 6, 11), device=self._device)
                # ['world', 'base_link', 'base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'flange', 'flange_tool_rot_x', 'flange_tool_rot_z', 'flange_tool_rot_y', 'flange_tool_tran_y', 'tool']
                # end-effector link index is 9 which is the flange
            else:
                self.jacobians = torch.zeros((self._num_envs,
                                            #   self._robots.body_names.index(self._flange_link),
                                              len(self._robots.body_names),
                                              6,
                                              11), device=self._device)
                '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
                num_of_bodies - 1 due to start from 0 index'''
            self.flange_pos, self.flange_rot = torch.zeros((self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)


    def get_observations(self) -> dict:
        self.step_num += 1
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

        # calculate farthest distance and idx from the tool to the goal
        diff = tool_pcd_transformed - self.flange_pos[:, None, :]
        distance = diff.norm(dim=2)  # [B, N]

        # Find the index and value of the farthest point from the base coordinate
        farthest_idx = distance.argmax(dim=1)  # [B]
        # farthest_val = distance.gather(1, farthest_idx.unsqueeze(1)).squeeze(1)  # [B]
        self.tool_end_point = tool_pcd_transformed.gather(1, farthest_idx.view(B, 1, 1).expand(B, 1, 3)).squeeze(1).squeeze(1)  # [B, 3]


        # get transformation matrix from base to target object
        T_base_to_tgt = torch.eye(4, device=self._device).unsqueeze(0).repeat(self._num_envs, 1, 1)
        T_base_to_tgt[:, :3, :3] = target_rot.clone().detach()
        T_base_to_tgt[:, :3, 3] = target_pos.clone().detach()

        ##### point cloud registration for target object #####
        B, N, _ = self.target_pcd.shape
        # Convert points to homogeneous coordinates by adding a dimension with ones
        homogeneous_points = torch.cat([self.target_pcd, torch.ones(B, N, 1, device=self.target_pcd.device)], dim=-1)
        # Perform batch matrix multiplication
        transformed_points_homogeneous = torch.bmm(homogeneous_points, T_base_to_tgt.transpose(1, 2))
        # Convert back from homogeneous coordinates by removing the last dimension
        target_pcd_transformed = transformed_points_homogeneous[..., :3]
        ##### point cloud registration for target object #####

        self.target_pos_xyz = torch.mean(target_pcd_transformed, dim=1)
        self.target_pos_xy = self.target_pos_xyz[:, [0, 1]]

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
            goal_position = self.flange_pos + self.actions[:, :3] / 70.0
            goal_orientation = self.flange_rot + self.actions[:, 3:] / 70.0
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, self._robots.body_names.index(self._flange_link), :, :6],
                                                        current_position=self.flange_pos,
                                                        current_orientation=self.flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=goal_orientation
                                                        )
            '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
            num_of_bodies - 1 due to start from 0 index
            s'''
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
        env_ids = env_ids.to(dtype=torch.int32)

        # reset target
        position = torch.tensor(self._target_position, device=self._device)
        target_pos = position.repeat(len(env_ids),1)
        ### randomize target position ###
        # # x, y randomize 는 ±0.1로 uniform random
        # # z_ref = torch.abs(target_pos[2])
        # z_ref = torch.unsqueeze(torch.abs(target_pos[:, 2]),1)
        # # generate uniform random values for randomizing target position
        # # reference: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        # x_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.6, 1.0).to(device=self._device)    # 0.8 ± 0.2
        # y_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.3, -0.1).to(device=self._device)  # -0.2 ± 0.1
        # target_pos = torch.cat((x_rand, y_rand, z_ref), dim=1) ### Do not randomize z position
        ### randomize target position ###

        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        target_ori = orientation.repeat(len(env_ids),1)
        self._targets.set_world_poses(target_pos + self._env_pos[env_ids],
                                      target_ori,
                                      indices=env_ids)
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
        self._goals.set_world_poses(goal_mark_pos + self._env_pos[env_ids], indices=env_ids)
        goal_pos = self._goals.get_local_poses()
        
        self.goal_pos_xy = goal_pos[0][:, [0, 1]]


        # reset robot
        pos = self.robot_default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)   # non-randomized
        # pos = torch.tensor([-0.4363, -1.2217, 1.4835, -1.7453, -1.5708, 1.5708, 0.0873, 1.2217, 0.0000, -1.5708, 0.0000], device=self._device).repeat(len(env_ids), 1)
        # ##### randomize robot pose #####
        # randomize_manipulator_pos = 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs-4), device=self._device) - 0.5)
        # # tool_pos = torch.zeros((len(env_ids), 4), device=self._device)  # 여기에 rand를 추가
        # ### TODO: 나중에는 윗 줄을 통해 tool position이 random position으로 고정되도록 변수화. pre_physics_step도 확인할 것
        # pos[:, 0:6] = pos[:, 0:6] + randomize_manipulator_pos
        # ##### randomize robot pose #####

        dof_pos = torch.zeros((len(env_ids), self._robots.num_dof), device=self._device)
        dof_pos[:, :] = pos
        dof_vel = torch.zeros((len(env_ids), self._robots.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self._robots.set_joint_positions(self.robot_dof_targets[env_ids], indices=env_ids)

        # target_pos 에서 y축으로 0.2만큼 뒤로 이동한 위치로 flange를 이동시키기
        ee_pos = deepcopy(target_pos)
        ee_pos[:, 0] -= 0.15    # x
        ee_pos[:, 1] -= 0.1     # y
        ee_pos[:, 2] = 0.35     # z
        # ee_pos = torch.tensor([ 0.6140, -0.1384,  0.4], device=self._device)
        # ee_pos = ee_pos.repeat(len(env_ids),1)

        # ee_ori = torch.tensor([0.707, 0.0, 0.707, 0.0], device=self._device)
        # ee_ori = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self._device)
        # ee_ori = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device)
        ee_ori = torch.tensor([0.0, -0.707, -0.707, 0.0], device=self._device)
        ee_ori = ee_ori.repeat(len(env_ids),1)

        initialized_pos = Pose(ee_pos, ee_ori, name="tool0")

        target_dof_pos = torch.empty(0).to(device=self._device)
        for i in range(initialized_pos.batch):
            result = self.ik_solver.solve_single(initialized_pos[i])
            target_dof_pos = torch.cat((target_dof_pos, result.solution[result.success]), dim=0)

        self.robot_dof_targets[env_ids, :6] = torch.clamp(target_dof_pos,
                                                          self.robot_dof_lower_limits[:6].repeat(len(env_ids),1),
                                                          self.robot_dof_upper_limits[:6].repeat(len(env_ids),1))
        self._robots.set_joint_positions(self.robot_dof_targets[env_ids], indices=env_ids)
        # self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

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
        # self.stage_buf[self.progress_buf ==1] = 0   # also initializing stage_buf to 0 when progress_buf is 1.
        initialized_idx = self.progress_buf == 1    # initialized index를 통해 progress_buf가 1인 경우에만 initial distance 계산
        self.completion_reward[:] = 0.0 # reset completion reward

        # # Stage 0: make the tool go to behind the target
        # behind_target = copy.deepcopy(self.target_pos_xyz)
        # behind_target[:, 1] -= 0.1
        # # TODO: 추후에는 y값에서 일관되게 0.1을 빼주는 것이 아니라 goal position을 고려하여 수식적으로 계산되도록 해주자.
        # current_tool_approach_distance = LA.norm(self.tool_end_point - behind_target, ord=2, dim=1)
        current_tool_target_distance = LA.norm(self.tool_end_point - self.target_pos_xyz, ord=2, dim=1)
        # self.initial_tool_approach_distance[initialized_idx] = current_tool_approach_distance[initialized_idx]
        # self.initial_tool_target_distance[initialized_idx] = current_tool_target_distance[initialized_idx]  # 초기화 시점의 tool-target distance

        # init_t_a_d = self.initial_tool_approach_distance
        # cur_t_a_d = current_tool_approach_distance
        # approach_threshold = 0.05

        # tool_approach_distance_reward = self.relu(-(cur_t_a_d - init_t_a_d)/init_t_a_d)
        
        # # Update stage_buf and store target position for Stage 1
        # stage1_satisfied = (self.stage_buf == 0) & (cur_t_a_d <= approach_threshold)
        # self.stage_buf[stage1_satisfied] = 1
        # # self.stage1_target_pos[stage1_satisfied] = self.target_pos_xyz[stage1_satisfied]


        # # Stage 1: tool touching the target
        # touching_threshold = 0.005 # Define a threshold for the tool touching the target
        current_target_goal_distance = LA.norm(self.goal_pos_xy - self.target_pos_xy, ord=2, dim=1)
        # self.initial_target_goal_distance[stage1_satisfied] = current_target_goal_distance[stage1_satisfied]
        # self.initial_tool_target_distance[stage1_satisfied] = current_tool_target_distance[stage1_satisfied]    # reward function에서 사용하기 위해 다시 초기화하여 저장

        cur_t_t_d = current_tool_target_distance
        cur_t_g_d = current_target_goal_distance
        init_t_t_d = self.initial_tool_target_distance
        init_t_g_d = self.initial_target_goal_distance
        
        # tool_target_distance_reward = self.relu(-(cur_t_t_d - init_t_t_d)/init_t_t_d)
        # stage2_satisfied = (self.stage_buf == 1) & ((cur_t_g_d-init_t_g_d) > touching_threshold)
        # self.stage_buf[stage2_satisfied] = 2

        # Stage 2: target reaching the goal
        target_goal_distance_reward = self.relu(-(cur_t_g_d - init_t_g_d)/init_t_g_d)

        # completion reward
        self.done_envs = cur_t_g_d <= 0.1
        # completion_reward = torch.where(self.done_envs, torch.full_like(cur_t_g_d, 100.0)[self.done_envs], torch.zeros_like(cur_t_g_d))
        self.completion_reward[self.done_envs] = torch.full_like(cur_t_g_d, 300.0)[self.done_envs]

        # stage3_satisfied = (self.stage_buf == 2) & self.done_envs

        # Combine rewards based on the stages
        # total_reward = torch.where(self.stage_buf==0,
        #                            self.alpha * tool_approach_distance_reward,
        #                            torch.where(self.stage_buf==1,
        #                                        self.beta * tool_target_distance_reward + self.alpha,
        #                                        self.gamma * target_goal_distance_reward + self.alpha + self.beta + self.completion_reward))            
        total_reward = target_goal_distance_reward + self.completion_reward


        # self.stage_buf[stage3_satisfied] = 0

        self.rew_buf[:] = total_reward
    

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
        reset = torch.where(self.done_envs, ones, reset)

        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, reset)




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

    # tool_end_point = o3d.geometry.TriangleMesh().create_sphere(radius=0.01)
    # tool_end_point.paint_uniform_color([0, 0, 1])
    # farthest_pt = tool_transformed_pcd_np[farthest_idx.detach().cpu().numpy()][view_idx]
    # T_t_p = np.eye(4)
    # T_t_p[:3, 3] = farthest_pt
    # tool_tip_position = copy.deepcopy(tool_end_point).transform(T_t_p)

    # tgt_transformed_pcd_np = target_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
    # tgt_transformed_point_cloud = o3d.geometry.PointCloud()
    # tgt_transformed_point_cloud.points = o3d.utility.Vector3dVector(tgt_transformed_pcd_np)
    # T_o = np.eye(4)

    # # R_b = tgt_rot_np.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    # T_o[:3, :3] = tgt_rot_np
    # # T_o[:3, :3] = R_b
    # T_o[:3, 3] = tgt_pos_np
    # tgt_coord = copy.deepcopy(base_coord).transform(T_o)

    # self.goal_pos, _ = self._goals.get_local_poses()
    # goal_pos_np = self.goal_pos[view_idx].cpu().numpy()
    # goal_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.01, height=0.03)
    # goal_cone.paint_uniform_color([0, 1, 0])
    # T_g_p = np.eye(4)
    # T_g_p[:3, 3] = goal_pos_np
    # goal_position = copy.deepcopy(goal_cone).transform(T_g_p)

    # goal_pos_xy_np = copy.deepcopy(goal_pos_np)
    # goal_pos_xy_np[2] = self.target_height
    # goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # goal_sphere.paint_uniform_color([1, 0, 0])
    # T_g = np.eye(4)
    # T_g[:3, 3] = goal_pos_xy_np
    # goal_position_xy = copy.deepcopy(goal_sphere).transform(T_g)

    # o3d.visualization.draw_geometries([base_coord,
    #                                    tool_transformed_point_cloud,
    #                                    tgt_transformed_point_cloud,
    #                                    tool_tip_position,
    #                                    tool_coord,
    #                                    tgt_coord,
    #                                    goal_position,
    #                                    goal_position_xy],
    #                                     window_name=f'point cloud')
    ################# visualize point cloud #################
    #########################################################