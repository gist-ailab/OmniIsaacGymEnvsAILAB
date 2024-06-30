import torch
from torch import linalg as LA
import numpy as np
import cv2
from gym import spaces
from copy import deepcopy
from collections import OrderedDict
 
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
from omniisaacgymenvs.robots.articulations.ur5e_tool.ur5e_tool import UR5eTool
from omniisaacgymenvs.robots.articulations.views.ur5e_view import UR5eView
from omniisaacgymenvs.tasks.utils.get_toolmani_assets import get_robot, get_object, get_goal
from omniisaacgymenvs.tasks.utils.pcd_processing import get_pcd, pcd_registration
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
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

import copy

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()    


class PCDMovingObjectTask(RLTask):
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

        self.robot_list = ['ur5e_tool', 'ur5e_fork', 'ur5e_knife', 'ur5e_ladle', 'ur5e_spatular', 'ur5e_spoon']
        self.robot_num = len(self.robot_list)        
        self.initial_object_goal_distance = torch.empty(self._num_envs).to(self.cfg["rl_device"])
        self.completion_reward = torch.zeros(self._num_envs).to(self.cfg["rl_device"])
        
        self.relu = torch.nn.ReLU()

        # tool orientation
        self.tool_rot_x = 70 # 70 degree
        self.tool_rot_z = 0     # 0 degree
        self.tool_rot_y = -90 # -90 degree

        # workspace 2D boundary
        self.x_min = 0.3
        self.x_max = 0.9
        self.y_min = -0.7
        self.y_max = 0.7
        self.z_min = 0.1
        self.z_max = 0.7
        
        self._pcd_sampling_num = self._task_cfg["sim"]["point_cloud_samples"]
        # observation and action space
        pcd_observations = self._pcd_sampling_num * 2 * 3
        # 2 is a number of point cloud masks(tool and object) and 3 is a cartesian coordinate
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

        # get tool point cloud
        for name in self.robot_list:
            # get tool pcd
            tool_name = name.split('_')[1]
            tool_ply_path = f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/{tool_name}/{tool_name}.ply"
            tool_pcd = get_pcd(tool_ply_path, self._num_envs, self._pcd_sampling_num, device=self.cfg["rl_device"], tools=True)
            globals()[f"{tool_name}_pcd"] = tool_pcd    # save tool pcd as global variable such as spatular_pcd, spoon_pcd etc.

        # TODO: if the environment has multiple objects, the object pcd should be loaded in the loop
        # get object pcd
        object_ply_path = f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/cylinder/cylinder.ply"
        self.object_pcd = get_pcd(object_ply_path, self._num_envs, self._pcd_sampling_num, device=self.cfg["rl_device"], tools=False)

        if self._control_space == "joint":
            self._num_actions = 6
        elif self._control_space == "cartesian":
            self._num_actions = 7   # 3 for position, 4 for rotation(quaternion)
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))

        self._flange_link = "tool0"
        
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
        collision_file = "/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/collision_bar.yml"
        
        world_cfg = WorldConfig.from_dict(load_yaml(collision_file))

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
            ee_link_name="flange",
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
        self._object_position = self._task_cfg["env"]["object"]
        self._pcd_normalization = self._task_cfg["sim"]["point_cloud_normalization"]


    def set_up_scene(self, scene) -> None:
        for name in self.robot_list:
            get_robot(name, self._sim_config, self.default_zero_env_path)
        
        # TODO: if the environment has multiple objects, the object should be loaded in the loop
        get_object('object', self._sim_config, self.default_zero_env_path)
        get_goal('goal',self._sim_config, self.default_zero_env_path)
        
        # RLTask.set_up_scene(self, scene)
        super().set_up_scene(scene)
        # self._rl_task_setup_scene(scene)

        # self.robots_dict = {}
        for idx, name in enumerate(self.robot_list):
            # Create an instance variable for each name in robot_list
            setattr(self, f"_{name}s", UR5eView(prim_paths_expr=f"/World/envs/.*/{name}", name=f"{name}_view"))
            scene.add(getattr(self, f"_{name}s"))
            scene.add(getattr(self, f"_{name}s")._flanges)
            scene.add(getattr(self, f"_{name}s")._tools)

        self._object = RigidPrimView(prim_paths_expr="/World/envs/.*/object", name="object_view", reset_xform_properties=False)
        scene.add(self._object)
        self._goal = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)
        self._goal._non_root_link = True    # do not set states for kinematics
        scene.add(self._goal)

        self.init_data()

    # region initialize_views 사용 안 하는 거 같은데 다른 예제들에 있음... 일단 주석 처리
    # def initialize_views(self, scene):
    #     super().initialize_views(scene)
    #     if scene.object_exist("robot_view"):
    #         scene.remove("robot_view", registry_only=True)
    #     if scene.object_exist("end_effector_view"):
    #         scene.remove("end_effector_view", registry_only=True)
    #     if scene.object_exist("tool_view"):
    #         scene.remove("tool_view", registry_only=True)
    #     if scene.object_exist("target_view"):
    #         scene.remove("target_view", registry_only=True)
    #     if scene.object_exist("goal_view"):
    #         scene.remove("goal_view", registry_only=True)
        
    #     # robots view
    #     self._ur5e_tools = UR5eView(prim_paths_expr="/World/envs/.*/_ur5e_tools", name="_ur5e_tools_view")
    #     # self._flanges = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/{self._flange_link}", name="end_effector_view")
    #     # self._tools = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/tool", name="tool_view")
    #     self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
    #     # self._cubes = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=True)
    #     self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)

    #     scene.add(self._ur5e_tools)
    #     # scene.add(self._flanges)
    #     # scene.add(self._tools)
    #     scene.add(self._ur5e_tools._flanges)
    #     scene.add(self._ur5e_tools._tools)
    #     scene.add(self._targets)
    #     scene.add(self._goals)
        
    #     self.init_data()
    # endregion
        

    def init_data(self) -> None:
        self.robot_default_dof_pos = torch.tensor(np.radians([-60, -80, 80, -90, -90, -40,
                                                              0, 30, 0.0, 0, -0.03]), device=self._device, dtype=torch.float32)
        ''' ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
             'grasped_position', 'flange_revolute_yaw', 'flange_revolute_pitch', 'flange_revolute_roll', 'tool_prismatic'] '''
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        self.jacobians = torch.zeros((self._num_envs, 15, 6, 11), device=self._device)
        ''' ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
             'grasped_position', 'flange_revolute_yaw', 'flange_revolute_pitch', 'flange_revolute_roll', 'tool_prismatic'] '''
        self.flange_pos = torch.zeros((self._num_envs, 3), device=self._device)
        self.flange_rot = torch.zeros((self._num_envs, 4), device=self._device)

    def post_reset(self):
        self.num_robot_dofs = self._ur5e_forks.num_dof  # all robots have the same dof
        
        dof_limits = self._ur5e_forks.get_dof_limits()  # every robot has the same dof limits
        # dof_limits = dof_limits.repeat(self.robot_num, 1, 1)

        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)
        
        self._objects.enable_rigid_body_physics()
        self._objects.enable_gravities()

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)


    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        if self.step_num == 0:
            pass
        else:
            self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(self._ur5e_forks.count,
                                     dtype=torch.int32,
                                     device=self._device)

        if self._control_space == "joint":
            dof_targets = self.robot_dof_targets[:, :6] + self.robot_dof_speed_scales[:6] * self.dt * self.actions * self._action_scale

        elif self._control_space == "cartesian":
            goal_position = self.flange_pos + self.actions[:, :3] / 70.0
            goal_orientation = self.flange_rot + self.actions[:, 3:] / 70.0
            flange_link_idx = self._ur5e_forks.body_names.index(self._flange_link)
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, flange_link_idx, :, :6],
                                                        current_position=self.flange_pos,
                                                        current_orientation=self.flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=goal_orientation
                                                        )
            '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
            num_of_bodies - 1 due to start from 0 index
            '''
            dof_targets = self.robot_dof_targets[:, :6] + delta_dof_pos[:, :6]

        self.robot_dof_targets[:, :6] = torch.clamp(dof_targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])

        self.robot_dof_targets[:, 7] = torch.deg2rad(torch.tensor(self.tool_rot_x, device=self._device))
        self.robot_dof_targets[:, 8] = torch.deg2rad(torch.tensor(self.tool_rot_z, device=self._device))
        self.robot_dof_targets[:, 9] = torch.deg2rad(torch.tensor(self.tool_rot_y, device=self._device))
        ### TODO: 나중에는 self.reset_idx에 tool pose가 episode마다 random position으로 바뀌도록 할 것. 환경의 index를 이용해야 할 것 같다.

        for name in self.robot_list:
            getattr(self, f"_{name}s").set_joint_positions(self.robot_dof_targets, indices=env_ids_int32)

        # self._targets.enable_rigid_body_physics()
        # self._targets.enable_rigid_body_physics(indices=env_ids_int32)
        # self._targets.enable_gravities(indices=env_ids_int32)

    def reset_idx(self, env_ids) -> None:
        env_ids = env_ids.to(dtype=torch.int32)

        # reset robot
        default_pos = self.robot_default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)   # non-randomized


        # TODO: 환경 index에 따라 정해진 로봇이 reset되도록 수정
        dof_pos = torch.zeros((len(env_ids), self._robots.num_dof), device=self._device)
        dof_pos[:, :] = default_pos
        dof_vel = torch.zeros((len(env_ids), self._robots.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self._robots.set_joint_positions(self.robot_dof_targets[env_ids], indices=env_ids)

        # target_pos 에서 y축으로 0.2만큼 뒤로 이동한 위치로 flange를 이동시키기
        ee_pos = deepcopy(target_pos)
        ee_pos[:, 0] -= 0.15    # x
        ee_pos[:, 1] -= 0.1     # y
        ee_pos[:, 2] = 0.35     # z
        '''github에서 코드 확인해가며 수정할 것'''

        ee_ori = torch.tensor([0.579, -0.579, -0.406, 0.406], device=self._device)
        ee_ori = ee_ori.repeat(len(env_ids),1)

        initialized_pos = Pose(ee_pos, ee_ori, name="tool0")
        target_dof_pos = torch.empty(0).to(device=self._device)
        for i in range(initialized_pos.batch):
            result = self.ik_solver.solve_single(initialized_pos[i])
            if result.success:
                target_dof_pos = torch.cat((target_dof_pos, result.solution[result.success]), dim=0)
            else:
                print(f"IK solver failed. Initialize a robot in env {env_ids[i]} with default pose.")
                target_dof_pos = torch.cat((target_dof_pos, self.robot_default_dof_pos[:6].unsqueeze(0)), dim=0)
        
        self.robot_dof_targets[env_ids, :6] = torch.clamp(target_dof_pos,
                                                          self.robot_dof_lower_limits[:6].repeat(len(env_ids),1),
                                                          self.robot_dof_upper_limits[:6].repeat(len(env_ids),1))







        # reset object
        position = torch.tensor(self._object_position, device=self._device)
        object_pos = position.repeat(len(env_ids),1)

        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        object_ori = orientation.repeat(len(env_ids),1)
        self._objects.set_world_poses(object_pos + self._env_pos[env_ids],
                                      object_ori,
                                      indices=env_ids)
        # self._objects.enable_rigid_body_physics()

        # reset goal
        goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)
        goal_mark_pos = goal_mark_pos.repeat(len(env_ids),1)
        self._goals.set_world_poses(goal_mark_pos + self._env_pos[env_ids], indices=env_ids)
        goal_pos = self._goals.get_local_poses()
        self.goal_pos_xy = goal_pos[0][:, [0, 1]]




        for idx, name in enumerate(self.robot_list):

            world_env_ids = env_ids[idx*self.num_envs:(idx+1)*self.num_envs]
            robot_env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self._device)

            robot_dof_targets = self.robot_dof_targets[(idx)*self.num_envs:(idx+1)*self.num_envs, :]

            # reset target
            position = torch.tensor(self._object_position, device=self._device)
            target_pos = position.repeat(len(robot_env_ids), 1)
            orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
            target_ori = orientation.repeat(len(robot_env_ids),1)
            self.robots_dict[name]['target_view'].set_world_poses(target_pos + self.robots_dict[name]['offset'] + self._env_pos[robot_env_ids],
                                                               target_ori,
                                                               indices=robot_env_ids)
            
            goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)
            goal_mark_pos = goal_mark_pos.repeat(len(robot_env_ids),1)
            self.robots_dict[name]['goal_view'].set_world_poses(goal_mark_pos + self.robots_dict[name]['offset'] + self._env_pos[robot_env_ids],
                                                             indices=robot_env_ids)
            
            # reset robot
            # reset to default pose            
            # self.robots_dict[name]['robot_view'].set_joint_positions(self.robot_default_dof_pos.unsqueeze(0).repeat(len(env_ids)), indices=env_ids)
            
            # move to start position near the target
            # TODO: start position을 random position으로 고정되도록 변수화. orientation이 물체를 바라보도록 설정
            ee_pos = deepcopy(target_pos)
            ee_pos[:, 0] -= 0.2    # x
            ee_pos[:, 1] -= 0.2     # y
            ee_pos[:, 2] = 0.35     # z

            ee_ori = torch.tensor([0.579, -0.579, -0.406, 0.406], device=self._device)
            ee_ori = ee_ori.repeat(len(robot_env_ids),1)

            initialized_pos = Pose(ee_pos, ee_ori, name="tool0")
            
            target_dof_pos = torch.empty(0).to(device=self._device)
            for i in range(initialized_pos.batch):
                result = self.ik_solver.solve_single(initialized_pos[i])
                #IK를 못 풀었을 때를 대비하여 result.success로 체크하고 false인 경우에는 기본 값으로 설정
                if result.success:
                    target_dof_pos = torch.cat((target_dof_pos, result.solution[result.success]), dim=0)
                else:
                    print(f"IK solver failed. Initialize a robot in {name} env {robot_env_ids[i]} with default pose.")
                    target_dof_pos = torch.cat((target_dof_pos, self.robot_default_dof_pos[:6].unsqueeze(0)), dim=0)
            robot_dof_targets[:, :6] = torch.clamp(target_dof_pos,
                                                    self.robot_dof_lower_limits[:6].repeat(len(robot_env_ids),1),
                                                    self.robot_dof_upper_limits[:6].repeat(len(robot_env_ids),1))
            self.robots_dict[name]['robot_view'].set_joint_positions(robot_dof_targets, indices=robot_env_ids)
            self.robots_dict[name]['robot_view'].set_joint_position_targets(robot_dof_targets, indices=robot_env_ids)
            self.robots_dict[name]['robot_view'].set_joint_velocities(torch.zeros((len(robot_env_ids), self.num_robot_dofs), device=self._device), indices=robot_env_ids)

            # bookkeeping
            self.reset_buf[sub_env_ids[idx]] = 0
            self.progress_buf[sub_env_ids[idx]] = 0

    def get_observations(self) -> dict:
        self._env.render()  # add for get point cloud on headless mode
        self.step_num += 1
        ''' retrieve point cloud data from all render products '''
        # tasks/utils/pcd_writer.py 에서 pcd sample하고 tensor로 변환해서 가져옴
        # pointcloud = self.pointcloud_listener.get_pointcloud_data()

        tools_pcd_flattened = torch.empty(0).to(device=self._device)
        targets_pcd_flattened = torch.empty(0).to(device=self._device)
        target_pcd_concat = torch.empty(0).to(device=self._device)  # concatenate target point cloud for getting xyz position
        self.goal_pos = torch.empty(0).to(device=self._device)  # concatenate goal position for getting xy position
        robots_dof_pos = torch.empty(0).to(device=self._device)
        robots_dof_vel = torch.empty(0).to(device=self._device)

        for idx, name in enumerate(self.robot_list):
            self.flange_pos[idx*self.num_envs:(idx+1)*self.num_envs], self.flange_rot[idx*self.num_envs:(idx+1)*self.num_envs] = self.robots_dict[name]['robot_view'].get_local_poses()
            target_pos, target_rot_quaternion = self.robots_dict[name]['target_view'].get_local_poses()
            target_rot = quaternion_to_matrix(target_rot_quaternion)
            tool_pos, tool_rot_quaternion = self.robots_dict[name]['robot_view']._tools.get_local_poses()
            tool_rot = quaternion_to_matrix(tool_rot_quaternion)

            # concat tool point cloud
            tool_pcd_transformed = pcd_registration(self.robots_dict[name]['tool_pcd'],
                                                    tool_pos,
                                                    tool_rot,
                                                    self.num_envs,
                                                    device=self._device)
            tool_pcd_flattend = tool_pcd_transformed.contiguous().view(self._num_envs, -1)
            tools_pcd_flattened = torch.cat((tools_pcd_flattened, tool_pcd_flattend), dim=0)

            # concat target point cloud
            target_pcd_transformed = pcd_registration(self.robots_dict[name]['target_pcd'],
                                                      target_pos,
                                                      target_rot,
                                                      self.num_envs,
                                                      device=self._device)
            target_pcd_concat = torch.cat((target_pcd_concat, target_pcd_transformed), dim=0)   # concat for calculating xyz position
            target_pcd_flattend = target_pcd_transformed.contiguous().view(self._num_envs, -1)
            targets_pcd_flattened = torch.cat((targets_pcd_flattened, target_pcd_flattend), dim=0)

            self.goal_pos = torch.cat((self.goal_pos, self.robots_dict[name]['goal_view'].get_local_poses()[0]), dim=0)

            # TODO: get farthest point from the tool to the goal

            # get robot dof position and velocity from 1st to 6th joint
            robots_dof_pos = torch.cat((robots_dof_pos, self.robots_dict[name]['robot_view'].get_joint_positions(clone=False)[:, 0:6]), dim=0)
            robots_dof_vel = torch.cat((robots_dof_vel, self.robots_dict[name]['robot_view'].get_joint_velocities(clone=False)[:, 0:6]), dim=0)
            # rest of the joints are not used for control. They are fixed joints at each episode.
        
        self.target_pos_xyz = torch.mean(target_pcd_concat, dim=1)
        self.target_pos_xy = self.target_pos_xyz[:, [0, 1]]

        self.goal_pos_xy = self.goal_pos[:, [0, 1]]

        '''self.target_pos_xy는 [num_envs * robot_num, 2]의 형태로 되어야 함'''

        # # normalize robot_dof_pos
        # dof_pos_scaled = 2.0 * (robots_dof_pos - self.robot_dof_lower_limits) \
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0   # normalized by [-1, 1]
        # dof_pos_scaled = (robots_dof_pos - self.robot_dof_lower_limits) \
        #                 /(self.robot_dof_upper_limits - self.robot_dof_lower_limits)    # normalized by [0, 1]
        dof_pos_scaled = robots_dof_pos    # non-normalized

        # # normalize robot_dof_vel
        # dof_vel_scaled = robots_dof_vel * self._dof_vel_scale
        # generalization_noise = torch.rand((dof_vel_scaled.shape[0], 6), device=self._device) + 0.5
        dof_vel_scaled = robots_dof_vel    # non-normalized

        '''
        아래 순서로 최종 obs_buf에 concat. 첫 차원은 환경 갯수
        1. tool point cloud (flattened)
        2. target object point cloud (flattened)
        3. robot dof position
        4. robot dof velocity
        5. flange position
        6. flange orientation
        7. goal position
        '''

        '''NE = self._num_envs * self.robot_num'''
        self.obs_buf = torch.cat((
                                #   tool_pcd_transformed.reshape([tool_pcd_transformed.shape[0], -1]),     # [NE, N*3], point cloud
                                #   target_pcd_transformed.reshape([target_pcd_transformed.shape[0], -1]), # [NE, N*3], point cloud
                                #   tool_pcd_transformed.contiguous().view(self._num_envs, -1),   # [NE, N*3], point cloud
                                #   target_pcd_transformed.contiguous().view(self._num_envs, -1), # [NE, N*3], point cloud
                                  tools_pcd_flattened,
                                  targets_pcd_flattened,
                                  dof_pos_scaled,                                               # [NE, 6]
                                #   dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
                                  dof_vel_scaled[:, :6],                                        # [NE, 6]
                                  self.flange_pos,                                              # [NE, 3]
                                  self.flange_rot,                                              # [NE, 4]
                                  self.goal_pos_xy,                                             # [NE, 2]
                                 ), dim=1)

        if self._control_space == "cartesian":
            ''' 위에있는 jacobian 차원을 참고해서 값을 넣어주어야 함. 로봇 종류에 맞추어 넣어주어야 할 것 같다.
             '''
            for idx, name in enumerate(self.robot_list):
                self.jacobians[idx*self.num_envs:(idx+1)*self.num_envs] = self.robots_dict[name]['robot_view'].get_jacobians(clone=False)

        # TODO: name???? 
        return {self.robots_dict['ur5e_fork']['robot_view'].name: {"obs_buf": self.obs_buf}}


    def calculate_metrics(self) -> None:
        initialized_idx = self.progress_buf == 1    # initialized index를 통해 progress_buf가 1인 경우에만 initial distance 계산
        self.completion_reward[:] = 0.0 # reset completion reward
        current_target_goal_distance = LA.norm(self.goal_pos_xy - self.target_pos_xy, ord=2, dim=1)
        self.initial_target_goal_distance[initialized_idx] = current_target_goal_distance[initialized_idx]

        init_t_g_d = self.initial_target_goal_distance
        cur_t_g_d = current_target_goal_distance
        target_goal_distance_reward = self.relu(-(cur_t_g_d - init_t_g_d)/init_t_g_d)

        # completion reward
        self.done_envs = cur_t_g_d <= 0.1
        # completion_reward = torch.where(self.done_envs, torch.full_like(cur_t_g_d, 100.0)[self.done_envs], torch.zeros_like(cur_t_g_d))
        self.completion_reward[self.done_envs] = torch.full_like(cur_t_g_d, 300.0)[self.done_envs]

        total_reward = target_goal_distance_reward + self.completion_reward

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