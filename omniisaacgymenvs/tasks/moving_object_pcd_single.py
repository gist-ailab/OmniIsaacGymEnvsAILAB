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

import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()    

class PCDMovingObjectSingleTask(RLTask):
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

        # self.robot_list = ['ur5e_tool', 'ur5e_fork', 'ur5e_knife', 'ur5e_ladle', 'ur5e_spatular', 'ur5e_spoon']
        self.robot_list = ['ur5e_tool']
        self.robot_num = len(self.robot_list)
        self.initial_object_goal_distance = torch.empty(self._num_envs).to(self.cfg["rl_device"])
        self.completion_reward = torch.zeros(self._num_envs).to(self.cfg["rl_device"])
        
        self.relu = torch.nn.ReLU()

        # tool orientation
        self.tool_rot_x = 1.221 # 70 degree
        self.tool_rot_z = 0     # 0 degree
        self.tool_rot_y = -1.5707 # -90 degree

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
            setattr(self, f"{tool_name}_pcd", tool_pcd) # save tool pcd as global variable such as spatular_pcd, spoon_pcd etc.

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
            ee_link_name="tool0",
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
        self._target_position = self._task_cfg["env"]["object"]
        self._pcd_normalization = self._task_cfg["sim"]["point_cloud_normalization"]

        
    def set_up_scene(self, scene) -> None:
        for name in self.robot_list:
            get_robot(name, self._sim_config, self.default_zero_env_path)

        # TODO: if the environment has multiple objects, the object should be loaded in the loop
        get_object('object', self._sim_config, self.default_zero_env_path)
        get_goal('goal',self._sim_config, self.default_zero_env_path)

        # RLTask.set_up_scene(self, scene)
        super().set_up_scene(scene)

        default_translations=torch.tensor([0.0, 0.0, 0.0]).repeat(self._num_envs,1)
        visibilities = torch.tensor([True]).repeat(self._num_envs)

        for name in self.robot_list:
            # Create an instance variable for each name in robot_list
            setattr(self, f"_{name}", UR5eView(
                                               prim_paths_expr=f"/World/envs/.*/{name}",
                                               name=f"{name}_view",
                                            #    translations=default_translations,
                                            #    visibilities=visibilities
                                               ))
            self._flanges = getattr(self, f"_{name}")._flanges
            self._tools = getattr(self, f"_{name}")._tools
            scene.add(getattr(self, f"_{name}"))
            scene.add(self._flanges)
            scene.add(self._tools)
        self.ref_robot = getattr(self, f"_{self.robot_list[0]}")

        self._object = RigidPrimView(prim_paths_expr="/World/envs/.*/object", name="object_view", reset_xform_properties=False)
        scene.add(self._object)
        self._goal = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)
        self._goal._non_root_link = True    # do not set states for kinematics
        scene.add(self._goal)

        self.init_data()

    def init_data(self) -> None:
        self.robot_default_dof_pos = torch.tensor(np.radians([-60, -80, 80, -90, -90, -40,
                                                              0, 30, 0.0, 0, -0.03]), device=self._device, dtype=torch.float32)
        ''' ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
             'grasped_position', 'flange_revolute_yaw', 'flange_revolute_pitch', 'flange_revolute_roll', 'tool_prismatic'] '''
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        self.jacobians = torch.zeros((self._num_envs, 15, 6, 11), device=self._device)
        ''' ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
             'grasped_position', 'flange_revolute_yaw', 'flange_revolute_pitch', 'flange_revolute_roll', 'tool_prismatic'] '''
        '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
        num_of_bodies - 1 due to start from 0 index
        '''
        self.flange_pos = torch.zeros((self._num_envs, 3), device=self._device)
        self.flange_rot = torch.zeros((self._num_envs, 4), device=self._device)
        self.object_pos = torch.zeros((self._num_envs, 3), device=self._device)
        self.object_rot = torch.zeros((self._num_envs, 4), device=self._device)

        self.empty_separated_envs = [torch.empty(0, dtype=torch.int32, device=self._device) for _ in self.robot_list]
        self.total_env_ids = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)


    def get_observations(self) -> dict:
        self._env.render()  # add for get point cloud on headless mode
        ''' retrieve point cloud data from all render products '''
        # tasks/utils/pcd_writer.py 에서 pcd sample하고 tensor로 변환해서 가져옴
        # pointcloud = self.pointcloud_listener.get_pointcloud_data()

        self.flange_pos, self.flange_rot = self._flanges.get_local_poses()
        # self.goal_pos, _ = self._goal.get_local_poses()
        # self.goal_pos_xy = self.goal_pos[:, [0, 1]]
        target_pos, target_rot_quaternion = self._object.get_local_poses()
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
        B, N, _ = self.object_pcd.shape
        # Convert points to homogeneous coordinates by adding a dimension with ones
        homogeneous_points = torch.cat([self.object_pcd, torch.ones(B, N, 1, device=self.object_pcd.device)], dim=-1)
        # Perform batch matrix multiplication
        transformed_points_homogeneous = torch.bmm(homogeneous_points, T_base_to_tgt.transpose(1, 2))
        # Convert back from homogeneous coordinates by removing the last dimension
        target_pcd_transformed = transformed_points_homogeneous[..., :3]
        ##### point cloud registration for tool ####

        target_pos_mean = torch.mean(target_pcd_transformed, dim=1)
        self.target_pos_xy = target_pos_mean[:, [0, 1]]

        '''
        아래 순서로 최종 obs_buf에 concat. 첫 차원은 환경 갯수
        1. robot dof position
        2. robot dof velocity
        3. flange position
        4. flange orientation
        5. target position => 이건 pointcloud_listener에서 받아와야 할듯? pcd 평균으로 하자
        6. goal position
        '''

        robot_dof_pos = getattr(self, f"_{self.robot_list[0]}").get_joint_positions(clone=False)[:, 0:6]   # get robot dof position from 1st to 6th joint
        robot_dof_vel = getattr(self, f"_{self.robot_list[0]}").get_joint_velocities(clone=False)[:, 0:6]  # get robot dof velocity from 1st to 6th joint
        # rest of the joints are not used for control. They are fixed joints at each episode.

        dof_pos_scaled = robot_dof_pos    # non-normalized


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

        if self._control_space == "cartesian":
            self.jacobians = getattr(self, f"_{self.robot_list[0]}").get_jacobians(clone=False)

        return {getattr(self, f"_{self.robot_list[0]}").name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(self.ref_robot.count, dtype=torch.int32, device=self._device)

        if self._control_space == "joint":
            targets = self.robot_dof_targets[:, :6] + self.robot_dof_speed_scales[:6] * self.dt * self.actions * self._action_scale

        elif self._control_space == "cartesian":
            goal_position = self.flange_pos + self.actions[:, :3] / 70.0
            goal_orientation = self.flange_rot + self.actions[:, 3:] / 70.0
            flange_link_idx = self.ref_robot.body_names.index(self._flange_link)
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, flange_link_idx-1, :, :6],
                                                        current_position=self.flange_pos,
                                                        current_orientation=self.flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=goal_orientation
                                                        )
            targets = self.robot_dof_targets[:, :6] + delta_dof_pos[:, :6]

        self.robot_dof_targets[:, :6] = torch.clamp(targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])
        self.robot_dof_targets[:, 7] = torch.deg2rad(torch.tensor(self.tool_rot_x, device=self._device))
        self.robot_dof_targets[:, 8] = torch.deg2rad(torch.tensor(self.tool_rot_z, device=self._device))
        self.robot_dof_targets[:, 9] = torch.deg2rad(torch.tensor(self.tool_rot_y, device=self._device))

        # # Diagnostic print statements to check targets and velocities
        # print(f"Actions: {self.actions}")
        # print(f"Targets: {targets}")
        # print(f"Clamped Targets: {self.robot_dof_targets[:, :6]}")
        
        # self._ur5e_tool.set_joint_position_targets(self.robot_dof_targets[env_ids_int32], indices=env_ids_int32)
        getattr(self, f"_{self.robot_list[0]}").set_joint_positions(self.robot_dof_targets[env_ids_int32], indices=env_ids_int32)
        # 절대 `set_joint_positions` 쓰면 안됨

    def reset_idx(self, env_ids) -> None:
        # episode 끝나고 env ids를 받아서 reset
        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        pos = self.robot_default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)   # non-randomized    
        
        dof_pos = torch.zeros((len(indices), self.ref_robot.num_dof), device=self._device)
        dof_pos[:, :] = pos
        dof_vel = torch.zeros((len(indices), self.ref_robot.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        getattr(self, f"_{self.robot_list[0]}").set_joint_positions(dof_pos, indices=indices)
        getattr(self, f"_{self.robot_list[0]}").set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)
        getattr(self, f"_{self.robot_list[0]}").set_joint_velocities(dof_vel, indices=indices)

        # reset target
        position = torch.tensor(self._target_position, device=self._device)
        target_pos = position.repeat(len(env_ids),1)
        
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        target_ori = orientation.repeat(len(env_ids),1)
        self._object.set_world_poses(target_pos + self._env_pos[env_ids],
                                      target_ori,
                                      indices=indices)
        # self._object.enable_rigid_body_physics()

        # reset goal
        goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)
        goal_mark_pos = goal_mark_pos.repeat(len(env_ids),1)
        self._goal.set_world_poses(goal_mark_pos + self._env_pos[env_ids], indices=indices)
        goal_pos = self._goal.get_local_poses()
        
        self.goal_pos_xy = goal_pos[0][:, [0, 1]]


        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_robot_dofs = self.ref_robot.num_dof
        self.robot_dof_pos = torch.zeros((self.num_envs, self.num_robot_dofs), device=self._device)
        dof_limits = self.ref_robot.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        initialized_idx = self.progress_buf == 1
        self.current_target_goal_distance = LA.norm(self.goal_pos_xy - self.target_pos_xy, ord=2, dim=1)
        self.initial_object_goal_distance[initialized_idx] = self.current_target_goal_distance[initialized_idx]

        init_t_g_d = self.initial_object_goal_distance
        cur_t_g_d = self.current_target_goal_distance
        target_goal_distance_reward = self.relu(-(cur_t_g_d - init_t_g_d)/init_t_g_d)

        self.completion_reward = torch.zeros(self._num_envs).to(self._device)
        self.completion_reward[cur_t_g_d <= 0.05] = 100.0
        self.rew_buf[:] = target_goal_distance_reward + self.completion_reward

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