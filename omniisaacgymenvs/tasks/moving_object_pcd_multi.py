import torch
import torch.nn.functional as F
from torch import linalg as LA
import os
import numpy as np
import math
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
from omniisaacgymenvs.robots.articulations.views.ur5e_view import UR5eView
from omniisaacgymenvs.tasks.utils.get_toolmani_assets import get_robot, get_object, get_goal
from omniisaacgymenvs.tasks.utils.transform_pcd_coord import *
from omniisaacgymenvs.tasks.utils.pcd_processing import get_pcd, pcd_registration
from omniisaacgymenvs.tasks.utils.pcd_writer import PointcloudWriter
from omniisaacgymenvs.tasks.utils.pcd_listener import PointcloudListener

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.types import ArticulationActions

from skrl.utils import omniverse_isaacgym_utils

import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix,axis_angle_to_quaternion, quaternion_multiply, matrix_to_quaternion

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()    


class PCDMovingObjectTaskMulti(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        #################### BSH
        self.rep = rep
        self.camera_width = 640
        self.camera_height = 640
        #################### BSH

        self.isaac_root_path = os.path.join(os.path.expanduser('~'), ".local/share/ov/pkg/isaac-sim-2023.1.1")
        self.update_config(sim_config)

        self.step_num = 0
        self.dt = self._sim_cfg['dt']
        self._env = env

        # self.robot_list = ['ur5e_fork', 'ur5e_hammer', 'ur5e_ladle', 'ur5e_roller',
        #                    'ur5e_spanner', 'ur5e_spatular', 'ur5e_spoon']
        self.robot_list = ['ur5e_fork', 'ur5e_hammer', 'ur5e_ladle',
                           'ur5e_spatular', 'ur5e_spoon']
        # self.robot_list = ['ur5e_spatular', 'ur5e_spoon']
        self.robot_num = len(self.robot_list)
        self.total_env_num = self._num_envs * self.robot_num
        
        self.relu = torch.nn.ReLU()

        self.grasped_position = torch.tensor(0.0, device=self.cfg["rl_device"])         # prismatic
        self.rev_yaw = torch.deg2rad(torch.tensor(30, device=self.cfg["rl_device"]))     # revolute
        self.rev_pitch = torch.deg2rad(torch.tensor(0.0, device=self.cfg["rl_device"]))  # revolute
        self.rev_roll = torch.deg2rad(torch.tensor(0.0, device=self.cfg["rl_device"]))   # revolute
        self.tool_pris = torch.tensor(-0.03, device=self.cfg["rl_device"])               # prismatic

        self.tool_6d_pos = torch.cat([
            self.grasped_position.unsqueeze(0),
            self.rev_yaw.unsqueeze(0),
            self.rev_pitch.unsqueeze(0),
            self.rev_roll.unsqueeze(0),
            self.tool_pris.unsqueeze(0)
        ])

        # workspace 2D boundary
        self.x_min, self.x_max = (0.2, 0.8)
        self.y_min, self.y_max = (-0.8, 0.8)
        self.z_min, self.z_max = (0.2, 0.7)
        
        # object min-max range
        self.obj_x_min, self.obj_x_max = (0.25, 0.7)    
        self.obj_y_min, self.obj_y_max = (-0.15, 0.4)
        self.obj_z = 0.03

        # goal min-max range
        self.goal_x_min, self.goal_x_max = (0.25, 0.7)
        self.goal_y_min, self.goal_y_max = (0.51, 0.71)
        self.goal_z = 0.1
        
        self._pcd_sampling_num = self._task_cfg["sim"]["point_cloud_samples"]
        # observation and action space
        pcd_observations = self._pcd_sampling_num * 2 * 3   # TODO: 환경 개수 * 로봇 대수 인데 이게 맞는지 확인 필요
        # 2 is a number of point cloud masks(tool and object) and 3 is a cartesian coordinate
        self._num_observations = pcd_observations + 6 + 6 + 3 + 4 + 2 + 2 + 3 + 4 + 1
        '''
        refer to observations in get_observations()
        tools_pcd_flattened                           # [NE, 3*pcd_sampling_num]
        objects_pcd_flattened                         # [NE, 3*pcd_sampling_num]
        dof_pos_scaled,                               # [NE, 6]
        dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
        flange_pos,                                   # [NE, 3]
        flange_rot,                                   # [NE, 4]
        goal_pos_xy,                                  # [NE, 2]
        object_to_goal_vector_norm                    # [NE, 2]
        grasping_points                               # [NE, 3]
        approx_tool_rots                              # [NE, 4]
        tool-object distance                          # [NE, 1]
        
        '''

        self.exp_dict = {}
        # get tool and object point cloud
        for name in self.robot_list:
            # get tool pcd
            tool_name = name.split('_')[1]
            tool_ply_path = os.path.join(self.isaac_root_path, f"OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/{tool_name}/{tool_name}.ply")
            tool_pcd = get_pcd(tool_ply_path, self._num_envs, self._pcd_sampling_num, device=self.cfg["rl_device"], tools=True)

            # get object pcd
            object_ply_path = os.path.join(self.isaac_root_path, f"OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/cylinder/cylinder.ply")
            object_pcd = get_pcd(object_ply_path, self._num_envs, self._pcd_sampling_num, device=self.cfg["rl_device"], tools=False)

            self.exp_dict[name] = {
                'tool_pcd' : tool_pcd,
                'object_pcd' : object_pcd,
            }

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
        collision_file = os.path.join(self.isaac_root_path, "OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/collision_bar.yml")
        
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
        self._sim_cfg = sim_config.sim_params

        self._base_coord = self._task_cfg["env"]["baseFrame"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._sub_spacing = self._task_cfg["env"]["subSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self._action_scale = self._task_cfg["env"]["actionScale"]
        # self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        # self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]

        self._dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]

        self._control_space = self._task_cfg["env"]["controlSpace"]
        self._pcd_normalization = self._task_cfg["sim"]["point_cloud_normalization"]

        # fixed object and goal position
        self._goal_mark = self._task_cfg["env"]["goal"]
        self._object_position = self._task_cfg["env"]["object"]


    def set_up_scene(self, scene) -> None:
        self.num_cols = math.ceil(self.robot_num ** 0.5)    # Calculate the side length of the square

        for idx, name in enumerate(self.robot_list):
            # Make the suv-environments into a grid
            x = idx // self.num_cols
            y = idx % self.num_cols
            get_robot(name, self._sim_config, self.default_zero_env_path,
                      translation=torch.tensor([x * self._sub_spacing, y * self._sub_spacing, 0.0]))
            get_object(name+'_object', self._sim_config, self.default_zero_env_path)
            get_goal(name+'_goal',self._sim_config, self.default_zero_env_path)
        self.robot_num = len(self.robot_list)

        super().set_up_scene(scene)

        for idx, name in enumerate(self.robot_list):
            self.exp_dict[name]['robot_view'] = UR5eView(prim_paths_expr=f"/World/envs/.*/{name}", name=f"{name}_view")
            self.exp_dict[name]['object_view'] = RigidPrimView(prim_paths_expr=f"/World/envs/.*/{name}_object", name=f"{name}_object_view", reset_xform_properties=False)
            self.exp_dict[name]['goal_view'] = RigidPrimView(prim_paths_expr=f"/World/envs/.*/{name}_goal", name=f"{name}_goal_view", reset_xform_properties=False)
            
            # offset is only need for the object and goal
            x = idx // self.num_cols
            y = idx % self.num_cols
            self.exp_dict[name]['offset'] = torch.tensor([x * self._sub_spacing,
                                                          y* self._sub_spacing,
                                                          0.0],
                                                          device=self._device).repeat(self.num_envs, 1)

            scene.add(self.exp_dict[name]['robot_view'])
            scene.add(self.exp_dict[name]['robot_view']._flanges)
            scene.add(self.exp_dict[name]['robot_view']._tools)

            scene.add(self.exp_dict[name]['object_view'])
            scene.add(self.exp_dict[name]['goal_view'])
        self.ref_robot = self.exp_dict[name]['robot_view']

        self.init_data()
        

    def init_data(self) -> None:
        self.robot_default_dof_pos = torch.tensor(np.radians([-60, -80, 80, -90, -90, -40,
                                                              0, 30, 0.0, 0, -0.03]), device=self._device, dtype=torch.float32)
        ''' ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
             'grasped_position', 'flange_revolute_yaw', 'flange_revolute_pitch', 'flange_revolute_roll', 'tool_prismatic'] '''
        self.actions = torch.zeros((self._num_envs*self.robot_num, self.num_actions), device=self._device)

        self.jacobians = torch.zeros((self._num_envs*self.robot_num, 15, 6, 11), device=self._device)
        ''' ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
             'grasped_position', 'flange_revolute_yaw', 'flange_revolute_pitch', 'flange_revolute_roll', 'tool_prismatic'] '''
        
        self.completion_reward = torch.zeros(self._num_envs*self.robot_num).to(self._device)

        self.abs_flange_pos = torch.zeros((self._num_envs*self.robot_num, 3), device=self._device)
        self.abs_flange_rot = torch.zeros((self._num_envs*self.robot_num, 4), device=self._device)
        # self.abs_object_pos = torch.zeros((self._num_envs*self.robot_num, 3), device=self._device)

        self.robot_part_pos = torch.zeros((self._num_envs*self.robot_num, 3), device=self._device)
        self.robot_part_rot = torch.zeros((self._num_envs*self.robot_num, 4), device=self._device)

        # self.tool_6d_pos = torch.zeros((self._num_envs*self.robot_num, 5), device=self._device)

        self.empty_separated_envs = [torch.empty(0, dtype=torch.int32, device=self._device) for _ in self.robot_list]
        self.total_env_ids = torch.arange(self._num_envs*self.robot_num, dtype=torch.int32, device=self._device)
        self.local_env_ids = torch.arange(self.ref_robot.count, dtype=torch.int32, device=self._device)

        self.initial_object_goal_distance = torch.empty(self._num_envs*self.robot_num).to(self._device)
        self.initial_tool_object_distance = torch.empty(self._num_envs*self.robot_num).to(self._device)
        self.prev_object_pos_xy = None
        # self.prev_action = None   # TODO: 코드 실행해보고 없어도 되면 이 줄 삭제


    # change from RLTask.cleanup()
    def cleanup(self) -> None:
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self._num_envs*self.robot_num, self.num_observations), device=self._device, dtype=torch.float)
        self.states_buf = torch.zeros((self._num_envs*self.robot_num, self.num_states), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs*self.robot_num, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs*self.robot_num, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs*self.robot_num, device=self._device, dtype=torch.long)
        self.extras = {}


    def post_reset(self):
        self.num_robot_dofs = self.ref_robot.num_dof
        
        dof_limits = self.ref_robot.get_dof_limits()  # every robot has the same dof limits
        # dof_limits = dof_limits.repeat(self.robot_num, 1, 1)

        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # self.robot_dof_targets = torch.zeros((self._num_envs*self.robot_num, self.num_robot_dofs), dtype=torch.float, device=self._device)
        self.robot_dof_targets = self.robot_default_dof_pos.unsqueeze(0).repeat(self.num_envs*self.robot_num, 1)
        self.zero_joint_velocities = torch.zeros((self._num_envs*self.robot_num, self.num_robot_dofs), dtype=torch.float, device=self._device)
        
        for name in self.robot_list:
            self.exp_dict[name]['object_view'].enable_rigid_body_physics()
            self.exp_dict[name]['object_view'].enable_gravities()

        indices = torch.arange(self._num_envs*self.robot_num, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        for name in self.robot_list:
            self.exp_dict[name]['object_view'].enable_rigid_body_physics()
            self.exp_dict[name]['object_view'].enable_gravities()
            self.exp_dict[name]['goal_view'].disable_rigid_body_physics()


    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        if self.step_num == 0:
            pass    # At the very first step, action dimension is [num_envs, 7]. but it should be [num_envs*robot_num, 7]
        else:
            self.actions = actions.clone().to(self._device)

        if self._control_space == "joint":
            dof_targets = self.robot_dof_targets[:, :6] + self.robot_dof_speed_scales[:6] * self.dt * self.actions * self._action_scale
        elif self._control_space == "cartesian":
            goal_position = self.abs_flange_pos + self.actions[:, :3] / 70.0
            goal_orientation = quaternion_multiply(self.actions[:, 3:] / 70.0, self.abs_flange_rot)
            flange_link_idx = self.ref_robot.body_names.index(self._flange_link)
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, flange_link_idx-1, :, :6],
                                                        current_position=self.abs_flange_pos,
                                                        current_orientation=self.abs_flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=goal_orientation
                                                        )
            '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
            num_of_bodies - 1 due to the body start from 'world'
            '''
            dof_targets = self.robot_dof_targets[:, :6] + delta_dof_pos

        self.robot_dof_targets[:, :6] = torch.clamp(dof_targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])       
        self.robot_dof_targets[:, :6] = torch.clamp(dof_targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])

        for name in self.robot_list:
            '''
            Caution:
             DO NOT USE set_joint_positions at pre_physics_step !!!!!!!!!!!!!!
             set_joint_positions: This method will immediately set (teleport) the affected joints to the indicated value.
                                  It make the robot unstable.
             set_joint_position_targets: Set the joint position targets for the implicit Proportional-Derivative (PD) controllers
             apply_action: apply multiple targets (position, velocity, and/or effort) in the same call
             (effort control means force/torque control)
            '''
            robot_env_ids = self.local_env_ids * self.robot_num + self.robot_list.index(name)
            robot_dof_targets = self.robot_dof_targets[robot_env_ids]
            articulation_actions = ArticulationActions(joint_positions=robot_dof_targets)
            self.exp_dict[name]['robot_view'].apply_action(articulation_actions, indices=self.local_env_ids)
            # apply_action의 환경 indices에 env_ids외의 index를 넣으면 GPU오류가 발생한다. 그래서 env_ids를 넣어야 한다.


    def separate_env_ids(self, env_ids):
        # Calculate the local index for each env_id
        local_indices = env_ids // self.robot_num
        # Calculate indices for each env_id based on its original value
        robot_indices = env_ids % self.robot_num
        # Create a mask for each robot
        masks = torch.stack([robot_indices == i for i in range(self.robot_num)], dim=1)
        # Apply the masks to separate the env_ids
        separated_envs = [local_indices[mask] for mask in masks.unbind(dim=1)]
        
        return separated_envs
    

    def reset_idx(self, env_ids) -> None:
        """
        Parameters  
        ----------
        exp_done_info : dict
        {
            "ur5e_spoon": torch.tensor([0 , 2, 3], device='cuda:0', dtype=torch.int32),
            "ur5e_spatular": torch.tensor([1, 2, 3], device='cuda:0', dtype=torch.int32),
            "ur5e_ladle": torch.tensor([3], device='cuda:0', dtype=torch.int32),
            "ur5e_fork": torch.tensor([2, 3], device='cuda:0', dtype=torch.int32),
            ...
        }
        """
        env_ids = env_ids.to(dtype=torch.int32)

        # Split env_ids using the split_env_ids function
        separated_envs = self.separate_env_ids(env_ids)

        for idx, sub_envs in enumerate(separated_envs):
            sub_env_size = sub_envs.size(0)
            if sub_env_size == 0:
                continue

            robot_name = self.robot_list[idx]
            robot_dof_targets = self.robot_dof_targets[sub_envs, :]

            # reset object
            ## fixed_values
            object_position = torch.tensor(self._object_position, device=self._device)  # objects' local pos
            object_pos = object_position.repeat(len(sub_envs), 1)

            # ## random_values
            # object_pos = torch.rand(sub_env_size, 2).to(device=self._device)
            # object_pos[:, 0] = self.obj_x_min + object_pos[:, 0] * (self.obj_x_max - self.obj_x_min)
            # object_pos[:, 1] = self.obj_y_min + object_pos[:, 1] * (self.obj_y_max - self.obj_y_min)
            # obj_z_coord = torch.full((sub_env_size, 1), self.obj_z, device=self._device)
            # object_pos = torch.cat([object_pos, obj_z_coord], dim=1)

            orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)   # objects' local orientation
            object_ori = orientation.repeat(len(sub_envs),1)
            obj_world_pos = object_pos + self.exp_dict[robot_name]['offset'][sub_envs, :] + self._env_pos[sub_envs, :]  # objects' world pos
            zero_vel = torch.zeros((len(sub_envs), 6), device=self._device)
            self.exp_dict[robot_name]['object_view'].set_world_poses(obj_world_pos,
                                                                     object_ori,
                                                                     indices=sub_envs)
            self.exp_dict[robot_name]['object_view'].set_velocities(zero_vel,
                                                                    indices=sub_envs)

            # reset goal
            ## fixed_values
            goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)  # goals' local pos
            goal_mark_pos = goal_mark_pos.repeat(len(sub_envs),1)

            # ## random_values
            # goal_mark_pos = torch.rand(sub_env_size, 2).to(device=self._device)
            # goal_mark_pos[:, 0] = self.goal_x_min + goal_mark_pos[:, 0] * (self.goal_x_max - self.goal_x_min)
            # goal_mark_pos[:, 1] = self.goal_y_min + goal_mark_pos[:, 1] * (self.goal_y_max - self.goal_y_min)
            # goal_z_coord = torch.full((sub_env_size, 1), self.goal_z, device=self._device)
            # goal_mark_pos = torch.cat([goal_mark_pos, goal_z_coord], dim=1)

            goal_mark_ori = orientation.repeat(len(sub_envs),1)
            goals_world_pos = goal_mark_pos + self.exp_dict[robot_name]['offset'][sub_envs, :] + self._env_pos[sub_envs, :]
            self.exp_dict[robot_name]['goal_view'].set_world_poses(goals_world_pos,
                                                                   goal_mark_ori,
                                                                   indices=sub_envs)
            # self.exp_dict[robot_name]['goal_view'].disable_rigid_body_physics()

            flange_pos = deepcopy(object_pos)
            # flange_pos[:, 0] -= 0.2     # x
            flange_pos[:, 0] -= 0.0     # x
            flange_pos[:, 1] -= 0.3    # y
            flange_pos[:, 2] = 0.45      # z            # Extract the x and y coordinates
            flange_xy = flange_pos[:, :2]
            object_xy = object_pos[:, :2]

            direction = object_xy - flange_xy                       # Calculate the vector from flange to object
            angle = torch.atan2(direction[:, 1], direction[:, 0])   # Calculate the angle of the vector relative to the x-axis
            axis_angle_z = torch.zeros_like(flange_pos)             # Create axis-angle representation for rotation about z-axis
            axis_angle_z[:, 2] = -angle                             # Negative angle to point towards the object
            quat_z = axis_angle_to_quaternion(axis_angle_z)         # Convert to quaternion

            # Create axis-angle representation for 180-degree rotation about y-axis
            axis_angle_y = torch.tensor([0., torch.pi, 0.], device=flange_pos.device).expand_as(flange_pos)
            quat_y = axis_angle_to_quaternion(axis_angle_y)     # Convert to quaternion
            flange_ori = quaternion_multiply(quat_y, quat_z)    # Combine the rotations (first rotate around z, then around y)

            # # reset tool pose with randomization
            # random_values = torch.rand(sub_envs.size()[0], 5).to(device=self._device)
            # tool_pos = self.robot_dof_lower_limits[6:] + random_values * (self.robot_dof_upper_limits[6:] - self.robot_dof_lower_limits[6:])

            # reset tool pose with fixed value
            tool_pos = self.tool_6d_pos.repeat(len(sub_envs), 1)

            initialized_pos = Pose(flange_pos, flange_ori, name="tool0")
            target_dof_pos = torch.empty(0).to(device=self._device)

            for i in range(initialized_pos.batch):  # solve IK with cuRobo
                solving_ik = False
                while not solving_ik:
                    # Though the all initialized poses are valid, there is a possibility that the IK solver fails.
                    result = self.ik_solver.solve_single(initialized_pos[i])
                    solving_ik = result.success
                    if not result.success:
                        # print(f"IK solver failed. Initialize a robot in {robot_name} env {sub_envs[i]} with default pose.")
                        # print(f"Failed pose: {initialized_pos[i]}")
                        continue
                    target_dof_pos = torch.cat((target_dof_pos, result.solution[result.success]), dim=0)

            robot_dof_targets[:, :6] = torch.clamp(target_dof_pos,
                                                   self.robot_dof_lower_limits[:6].repeat(len(sub_envs),1),
                                                   self.robot_dof_upper_limits[:6].repeat(len(sub_envs),1))
            robot_dof_targets[:, 6:] = tool_pos

            self.exp_dict[robot_name]['robot_view'].set_joint_positions(robot_dof_targets, indices=sub_envs)
            # self.exp_dict[robot_name]['robot_view'].set_joint_position_targets(robot_dof_targets, indices=sub_envs)
            self.exp_dict[robot_name]['robot_view'].set_joint_velocities(torch.zeros((len(sub_envs), self.num_robot_dofs), device=self._device),
                                                                   indices=sub_envs)
            
            # bookkeeping
            separated_abs_env = separated_envs[idx]*self.robot_num + idx
            self.reset_buf[separated_abs_env] = 0
            self.progress_buf[separated_abs_env] = 0


    def get_observations(self) -> dict:
        self._env.render()  # add for get point cloud on headless mode
        self.step_num += 1
        ''' retrieve point cloud data from all render products '''
        # tasks/utils/pcd_writer.py 에서 pcd sample하고 tensor로 변환해서 가져옴
        # pointcloud = self.pointcloud_listener.get_pointcloud_data()

        tools_pcd_flattened = torch.empty(self._num_envs*self.robot_num, self._pcd_sampling_num*3).to(device=self._device)
        objects_pcd_flattened = torch.empty(self._num_envs*self.robot_num, self._pcd_sampling_num*3).to(device=self._device)
        abs_object_pcd_set = torch.zeros((self._num_envs*self.robot_num, self._pcd_sampling_num, 3), device=self._device)
        object_pcd_set = torch.empty(self._num_envs*self.robot_num, self._pcd_sampling_num, 3).to(device=self._device) # object point cloud set for getting xyz position
        # multiply by 3 because the point cloud has 3 channels (x, y, z)

        flange_pos = torch.empty(self._num_envs*self.robot_num, 3).to(device=self._device)
        flange_rot = torch.empty(self._num_envs*self.robot_num, 4).to(device=self._device)
        goal_pos = torch.empty(self._num_envs*self.robot_num, 3).to(device=self._device)  # save goal position for getting xy position

        robots_dof_pos = torch.empty(self._num_envs*self.robot_num, 6).to(device=self._device)
        robots_dof_vel = torch.empty(self._num_envs*self.robot_num, 6).to(device=self._device)
        approx_tool_rots = torch.empty(self._num_envs*self.robot_num, 4).to(device=self._device)
        grasping_points = torch.empty(self._num_envs*self.robot_num, 3).to(device=self._device)

        self.tool_obj_distance = torch.empty(self._num_envs*self.robot_num, 1).to(self._device)

        for idx, robot_name in enumerate(self.robot_list):
            local_abs_env_ids = self.local_env_ids*self.robot_num + idx
            robot_flanges = self.exp_dict[robot_name]['robot_view']._flanges
            
            abs_flange_pos, abs_flange_rot = robot_flanges.get_local_poses()
            self.abs_flange_pos[local_abs_env_ids], self.abs_flange_rot[local_abs_env_ids] = abs_flange_pos, abs_flange_rot

            object_pos, object_rot_quaternion = self.exp_dict[robot_name]['object_view'].get_local_poses()
            
            # local object pose values are indicate with the environment ids with regard to the robot set
            x = (idx // self.num_cols) * self._sub_spacing
            y = (idx % self.num_cols) * self._sub_spacing
            object_pos[:, :2] -= torch.tensor([x, y], device=self._device)
            object_rot = quaternion_to_matrix(object_rot_quaternion)
            tool_pos, tool_rot_quaternion = self.exp_dict[robot_name]['robot_view']._tools.get_local_poses()
            tool_rot = quaternion_to_matrix(tool_rot_quaternion)

            if self._base_coord == 'flange':
                ee_transform = create_ee_transform(abs_flange_pos, abs_flange_rot)

            # concat tool point cloud
            tool_pcd_transformed = pcd_registration(self.exp_dict[robot_name]['tool_pcd'],
                                                    tool_pos,
                                                    tool_rot,
                                                    self.num_envs,
                                                    device=self._device)

            # concat object point cloud
            object_pcd_transformed = pcd_registration(self.exp_dict[robot_name]['object_pcd'],
                                                      object_pos,
                                                      object_rot,
                                                      self.num_envs,
                                                      device=self._device)

            local_goal_pos = self.exp_dict[robot_name]['goal_view'].get_local_poses()[0]
            local_goal_pos[:, :2] -= torch.tensor([x, y], device=self._device)  # revise the goal pos

            # get robot dof position and velocity from 1st to 6th joint
            robots_dof_pos[local_abs_env_ids] = self.exp_dict[robot_name]['robot_view'].get_joint_positions(clone=False)[:, 0:6]
            robots_dof_vel[local_abs_env_ids] = self.exp_dict[robot_name]['robot_view'].get_joint_velocities(clone=False)[:, 0:6]
            
            # Calculate tool's grasping point and orientation from transformed PCD using PCA and predefined positions/rotations
            imaginary_grasping_point = get_imaginary_grasping_point(abs_flange_pos, abs_flange_rot)
            cropped_tool_pcd = crop_tool_pcd(tool_pcd_transformed, imaginary_grasping_point)
            tool_tip_point = get_tool_tip_position(imaginary_grasping_point, tool_pcd_transformed)
            first_principal_axes, cropped_pcd_mean = apply_pca(cropped_tool_pcd)
            #####
            real_grasping_point = get_real_grasping_point(abs_flange_pos, abs_flange_rot, first_principal_axes, cropped_pcd_mean)
            approx_tool_rot = calculate_tool_orientation(first_principal_axes, tool_tip_point, imaginary_grasping_point)
            #####

            # transform pcd and coordinates
            if self._base_coord == 'flange':
                abs_object_pcd_set[local_abs_env_ids] = object_pcd_transformed
                flange_pos[local_abs_env_ids] = torch.zeros_like(robot_flanges.get_local_poses()[0])
                flange_rot[local_abs_env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).expand(self.num_envs, 4)
                tool_pcd_transformed = transform_points(tool_pcd_transformed, ee_transform)
                object_pcd_transformed = transform_points(object_pcd_transformed, ee_transform)
                goal_pos[local_abs_env_ids] = transform_points(local_goal_pos.unsqueeze(1), ee_transform).squeeze(1)
                real_grasping_point, approx_tool_rot = transform_pose(real_grasping_point, approx_tool_rot, ee_transform)
            else:
                flange_pos[local_abs_env_ids] = abs_flange_pos
                flange_rot[local_abs_env_ids] = abs_flange_rot
                goal_pos[local_abs_env_ids] = local_goal_pos
            
            # transform pcd and coordinates
            tool_pcd_flattend = tool_pcd_transformed.contiguous().view(self._num_envs, -1)
            tools_pcd_flattened[local_abs_env_ids] = tool_pcd_flattend

            object_pcd_set[local_abs_env_ids] = object_pcd_transformed  # for calculating xyz position
            object_pcd_flattend = object_pcd_transformed.contiguous().view(self._num_envs, -1)
            objects_pcd_flattened[local_abs_env_ids] = object_pcd_flattend

            grasping_points[local_abs_env_ids] = real_grasping_point
            approx_tool_rots[local_abs_env_ids] = approx_tool_rot

            self.tool_obj_distance[local_abs_env_ids], min_indices = closest_distance_between_sets(tool_pcd_transformed, object_pcd_transformed)


            '''
            # visualize the point cloud
            if self._base_coord == 'flange':
                cropped_tool_pcd = transform_points(cropped_tool_pcd, ee_transform)
                tool_pos, tool_rot_quaternion = transform_pose(tool_pos, tool_rot_quaternion, ee_transform)
                tool_rot = quaternion_to_matrix(tool_rot_quaternion)

                imaginary_grasping_point = transform_points(imaginary_grasping_point.unsqueeze(1), ee_transform).squeeze(1)
                tool_tip_point = transform_points(tool_tip_point.unsqueeze(1), ee_transform).squeeze(1)
                object_pos, object_rot_quaternion = transform_pose(object_pos, object_rot_quaternion, ee_transform)
                object_rot = quaternion_to_matrix(object_rot_quaternion)
                first_principal_axes = transform_principal_axis(ee_transform, first_principal_axes)
                base_pos, base_rot = get_base_in_flange_frame(self.abs_flange_pos[local_abs_env_ids], self.abs_flange_rot[local_abs_env_ids])
            else:
                base_pos = torch.tensor([0.0, 0.0, 0.0], device=self._device).expand(self.num_envs, 3)
                base_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).expand(self.num_envs, 4)

            visualize_pcd(tool_pcd_transformed, cropped_tool_pcd,
                               tool_pos, tool_rot,
                               imaginary_grasping_point,
                               real_grasping_point,
                               tool_tip_point,
                               first_principal_axes,
                               approx_tool_rot,
                               object_pcd_transformed,
                               base_pos, base_rot,
                               object_pos, object_rot,
                               flange_pos[local_abs_env_ids], flange_rot[local_abs_env_ids],
                               goal_pos[local_abs_env_ids],
                               min_indices,
                               self._base_coord,
                               view_idx=0)
            # visualize the point cloud
            '''


        if self._base_coord == 'flange':
            robot_part_position, robot_part_orientation = get_base_in_flange_frame(self.abs_flange_pos, self.abs_flange_rot)
        else:
            robot_part_position = flange_pos
            robot_part_orientation = flange_rot

        abs_obj_pos_xyz = torch.mean(abs_object_pcd_set, dim=1)
        self.abs_obj_pos_xy = abs_obj_pos_xyz[:, [0, 1]]

        object_pos_xyz = torch.mean(object_pcd_set, dim=1)
        self.object_pos_xy = object_pos_xyz[:, [0, 1]]

        self.goal_pos_xy = goal_pos[:, [0, 1]]

        # the unit vector from the object to the goal
        object_to_goal_vec = self.goal_pos_xy - self.object_pos_xy
        object_to_goal_unit_vec = object_to_goal_vec / (LA.vector_norm(object_to_goal_vec, dim=1, keepdim=True) + 1e-8)

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
        2. object object point cloud (flattened)
        3. robot dof position
        4. robot dof velocity
        5. robot part position
        6. robot part orientation
        7. goal position
        8. grasping_point
        9. tool_orientation
        '''

        '''the number of evironments; NE = self._num_envs * self.robot_num'''
        self.obs_buf = torch.cat((
                                  tools_pcd_flattened,                                          # [NE, N*3], point cloud
                                  objects_pcd_flattened,                                        # [NE, N*3], point cloud
                                  dof_pos_scaled,                                               # [NE, 6]
                                #   dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
                                  dof_vel_scaled,                                               # [NE, 6]
                                  robot_part_position,                                          # [NE, 3]
                                  robot_part_orientation,                                       # [NE, 4]
                                  self.goal_pos_xy,                                             # [NE, 2]
                                  object_to_goal_unit_vec,                                      # [NE, 2]
                                  grasping_points,                                              # [NE, 3]
                                  approx_tool_rots,                                             # [NE, 4]
                                  self.tool_obj_distance                                        # [NE, 1]
                                 ), dim=1)

        if self._control_space == "cartesian":
            ''' 위에있는 jacobian 차원을 참고해서 값을 넣어주어야 함. 로봇 종류에 맞추어 넣어주어야 할 것 같다.
             '''
            for idx, name in enumerate(self.robot_list):
                self.jacobians[idx*self.num_envs:(idx+1)*self.num_envs] = self.exp_dict[name]['robot_view'].get_jacobians(clone=False)

        # TODO: name???? 
        # return {self.exp_dict['ur5e_fork']['robot_view'].name: {"obs_buf": self.obs_buf}}
        return self.obs_buf


    def calculate_metrics(self) -> None:
        initialized_idx = self.progress_buf == 1    # initialized index를 통해 progress_buf가 1인 경우에만 initial distance 계산
        self.completion_reward[:] = 0.0 # reset completion reward

        if not hasattr(self, 'prev_object_pos_xy') or not hasattr(self, 'prev_flange_pos'):
            self.prev_object_pos_xy = self.object_pos_xy.clone()
            self.prev_flange_pos = self.abs_flange_pos.clone()
            self.prev_tool_obj_distance = self.tool_obj_distance.view(-1).clone()

        current_object_goal_distance = LA.vector_norm(self.goal_pos_xy - self.object_pos_xy, ord=2, dim=1)
        current_tool_obj_distance = self.tool_obj_distance.view(-1)
        self.initial_object_goal_distance[initialized_idx] = current_object_goal_distance[initialized_idx]
        self.initial_tool_object_distance[initialized_idx] = current_tool_obj_distance[initialized_idx]

        '''When delta object-goal distance is negative, distance reward = 0'''
        # Delta Object-Goal Distance Reward
        prev_obj_goal_distance = LA.vector_norm(self.goal_pos_xy - self.prev_object_pos_xy, ord=2, dim=1)
        curr_obj_goal_distance = LA.vector_norm(self.goal_pos_xy - self.object_pos_xy, ord=2, dim=1)
        # delta_object_goal_distance = curr_obj_goal_distance - prev_obj_goal_distance
        delta_object_goal_distance = prev_obj_goal_distance - curr_obj_goal_distance
        delta_object_goal_distance = torch.where(delta_object_goal_distance > -0.001,
                                                 torch.abs(delta_object_goal_distance),
                                                 delta_object_goal_distance)
        
        # Object-Goal Distance Reward
        object_goal_distance_reward = torch.where(delta_object_goal_distance >= 0,
                                                  self.relu(-(current_object_goal_distance - self.initial_object_goal_distance) / (self.initial_object_goal_distance)),
                                                  torch.zeros_like(delta_object_goal_distance))
        '''obj-goal 사이 거리가 가까워 지거나(prev - curr > 0) 멀어지지 않으면(prev - curr ≒ 0) reward를 준다.'''

        # Delta Tool-Object Distance Reward
        # delta_tool_obj_distance = current_tool_obj_distance - self.prev_tool_obj_distance
        delta_tool_obj_distance = self.prev_tool_obj_distance - current_tool_obj_distance
        delta_tool_obj_distance = torch.where(delta_tool_obj_distance > -0.001,
                                              torch.abs(delta_tool_obj_distance),
                                              delta_tool_obj_distance)
        
        # Tool-Object Distance Reward
        tool_object_distance_reward = torch.where(delta_tool_obj_distance >= 0,
                                                  self.relu(-(current_tool_obj_distance - self.initial_tool_object_distance) / (self.initial_tool_object_distance)),
                                                  torch.zeros_like(delta_tool_obj_distance))
        '''tool-obj 사이 거리가 가까워 지거나(prev - curr > 0) 멀어지지 않으면(prev - curr ≒ 0) reward를 준다.'''


        # Combine rewards
        total_reward = (
            object_goal_distance_reward +
            tool_object_distance_reward * 0.2       
        )

        # completion reward
        self.done_envs = current_object_goal_distance <= 0.05
        # completion_reward = torch.where(self.done_envs, torch.full_like(cur_t_g_d, 100.0)[self.done_envs], torch.zeros_like(cur_t_g_d))
        self.completion_reward[self.done_envs] = 300.0

        total_reward += self.completion_reward

        self.rew_buf[:] = total_reward

        # Store current state for next iteration
        self.prev_object_pos_xy = self.object_pos_xy.clone()
        self.prev_flange_pos = self.abs_flange_pos.clone()
        self.prev_tool_obj_distance = self.tool_obj_distance.view(-1).clone()
    

    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        reset = torch.zeros_like(self.reset_buf)

        # # workspace regularization
        reset = torch.where(self.abs_flange_pos[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 2] < self.z_min, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 1] > self.y_max, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 2] > self.z_max, ones, reset)
        
        reset = torch.where(self.abs_obj_pos_xy[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.abs_obj_pos_xy[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.abs_obj_pos_xy[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.abs_obj_pos_xy[:, 1] > self.y_max, ones, reset)
        # reset = torch.where(self.object_pos_xy[:, 2] > 0.5, ones, reset)    # prevent unexpected object bouncing

        # object reached
        reset = torch.where(self.done_envs, ones, reset)

        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, reset)