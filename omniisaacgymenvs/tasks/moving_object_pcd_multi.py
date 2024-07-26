import torch
import torch.nn.functional as F
from torch import linalg as LA
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

        self.update_config(sim_config)

        self.step_num = 0
        self.dt = self._sim_cfg['dt']
        self._env = env

        self.robot_list = ['ur5e_fork', 'ur5e_hammer', 'ur5e_ladle', 'ur5e_roller',
                           'ur5e_spanner', 'ur5e_spatular', 'ur5e_spoon']
        # self.robot_list = ['ur5e_spatular']
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
        self.z_min, self.z_max = (0.5, 0.7)
        
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
        self._num_observations = pcd_observations + 6 + 6 + 3 + 4 + 2 + 2 + 3 + 4
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
        
        '''

        self.exp_dict = {}
        # get tool and object point cloud
        for name in self.robot_list:
            # get tool pcd
            tool_name = name.split('_')[1]
            tool_ply_path = f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/{tool_name}/{tool_name}.ply"
            tool_pcd = get_pcd(tool_ply_path, self._num_envs, self._pcd_sampling_num, device=self.cfg["rl_device"], tools=True)

            # get object pcd
            object_ply_path = f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/cylinder/cylinder.ply"
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

        self.robot_part_pos = torch.zeros((self._num_envs*self.robot_num, 3), device=self._device)
        self.robot_part_rot = torch.zeros((self._num_envs*self.robot_num, 4), device=self._device)

        # self.tool_6d_pos = torch.zeros((self._num_envs*self.robot_num, 5), device=self._device)

        self.empty_separated_envs = [torch.empty(0, dtype=torch.int32, device=self._device) for _ in self.robot_list]
        self.total_env_ids = torch.arange(self._num_envs*self.robot_num, dtype=torch.int32, device=self._device)
        self.local_env_ids = torch.arange(self.ref_robot.count, dtype=torch.int32, device=self._device)

        self.initial_object_goal_distance = torch.empty(self._num_envs*self.robot_num).to(self._device)
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
            flange_pos[:, 2] = 0.4      # z            # Extract the x and y coordinates
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
        object_pcd_set = torch.empty(self._num_envs*self.robot_num, self._pcd_sampling_num, 3).to(device=self._device) # object point cloud set for getting xyz position
        # multiply by 3 because the point cloud has 3 channels (x, y, z)

        flange_pos = torch.empty(self._num_envs*self.robot_num, 3).to(device=self._device)
        flange_rot = torch.empty(self._num_envs*self.robot_num, 4).to(device=self._device)
        goal_pos = torch.empty(self._num_envs*self.robot_num, 3).to(device=self._device)  # save goal position for getting xy position

        robots_dof_pos = torch.empty(self._num_envs*self.robot_num, 6).to(device=self._device)
        robots_dof_vel = torch.empty(self._num_envs*self.robot_num, 6).to(device=self._device)
        approx_tool_rots = torch.empty(self._num_envs*self.robot_num, 4).to(device=self._device)
        grasping_points = torch.empty(self._num_envs*self.robot_num, 3).to(device=self._device)

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
                ee_transform = self.create_ee_transform(abs_flange_pos, abs_flange_rot)

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
            imaginary_grasping_point = self.get_imaginary_grasping_point(abs_flange_pos, abs_flange_rot)
            cropped_tool_pcd = self.crop_tool_pcd(tool_pcd_transformed, imaginary_grasping_point)
            tool_tip_point = self.get_tool_tip_position(imaginary_grasping_point, tool_pcd_transformed)
            first_principal_axes, cropped_pcd_mean = self.apply_pca(cropped_tool_pcd)
            #####
            real_grasping_point = self.get_real_grasping_point(abs_flange_pos, abs_flange_rot, first_principal_axes, cropped_pcd_mean)
            approx_tool_rot = self.calculate_tool_orientation(first_principal_axes, tool_tip_point, imaginary_grasping_point)
            #####

            # transform pcd and coordinates
            if self._base_coord == 'flange':
                flange_pos[local_abs_env_ids] = torch.zeros_like(robot_flanges.get_local_poses()[0])
                flange_rot[local_abs_env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).expand(self.num_envs, 4)
                tool_pcd_transformed = self.transform_points(tool_pcd_transformed, ee_transform)
                object_pcd_transformed = self.transform_points(object_pcd_transformed, ee_transform)
                goal_pos[local_abs_env_ids] = self.transform_points(local_goal_pos.unsqueeze(1), ee_transform).squeeze(1)
                real_grasping_point, approx_tool_rot = self.transform_pose(real_grasping_point, approx_tool_rot, ee_transform)
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


            # # visualize the point cloud
            # if self._base_coord == 'flange':
            #     cropped_tool_pcd = self.transform_points(cropped_tool_pcd, ee_transform)
            #     tool_pos, tool_rot_quaternion = self.transform_pose(tool_pos, tool_rot_quaternion, ee_transform)
            #     tool_rot = quaternion_to_matrix(tool_rot_quaternion)

            #     imaginary_grasping_point = self.transform_points(imaginary_grasping_point.unsqueeze(1), ee_transform).squeeze(1)
            #     tool_tip_point = self.transform_points(tool_tip_point.unsqueeze(1), ee_transform).squeeze(1)
            #     object_pos, object_rot_quaternion = self.transform_pose(object_pos, object_rot_quaternion, ee_transform)
            #     object_rot = quaternion_to_matrix(object_rot_quaternion)
            #     first_principal_axes = self.transform_principal_axis(ee_transform, first_principal_axes)
            #     base_pos, base_rot = self.get_base_in_flange_frame(self.abs_flange_pos[local_abs_env_ids], self.abs_flange_rot[local_abs_env_ids])
            # else:
            #     base_pos = torch.tensor([0.0, 0.0, 0.0], device=self._device).expand(self.num_envs, 3)
            #     base_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).expand(self.num_envs, 4)

            
            # '''
            # view_idx = 0
            # base_coord = 'flange'
            # flange_pos_np = flange_pos[local_abs_env_ids][0].detach().cpu().numpy()
            # flange_rot_np = flange_rot[local_abs_env_ids][0].detach().cpu().numpy()
            # tool_pos_np = tool_pos[view_idx].cpu().numpy()
            # tool_rot_np = tool_rot[view_idx].cpu().numpy()
            # base_pos_np = base_pos[view_idx].cpu().numpy()
            # base_rot_np = base_rot[view_idx].cpu().numpy()
            # obj_pos_np = object_pos[view_idx].cpu().numpy()
            # obj_rot_np = object_rot[view_idx].cpu().numpy()
            # # 지금 변환된 pcd랑 좌표계랑 안 맞음
            
            # '''

            # self.visualize_pcd(tool_pcd_transformed, cropped_tool_pcd,
            #                    tool_pos, tool_rot,
            #                    imaginary_grasping_point,
            #                    real_grasping_point,
            #                    tool_tip_point,
            #                    first_principal_axes,
            #                    approx_tool_rot,
            #                    object_pcd_transformed,
            #                    base_pos, base_rot,
            #                    object_pos, object_rot,
            #                    flange_pos[local_abs_env_ids], flange_rot[local_abs_env_ids],
            #                    goal_pos[local_abs_env_ids],
            #                    self._base_coord,
            #                    view_idx=0)
            # # visualize the point cloud


        if self._base_coord == 'flange':
            robot_part_position, robot_part_orientation = self.get_base_in_flange_frame(self.abs_flange_pos, self.abs_flange_rot)
        else:
            robot_part_position = flange_pos
            robot_part_orientation = flange_rot

        self.object_pos_xyz = torch.mean(object_pcd_set, dim=1)
        self.object_pos_xy = self.object_pos_xyz[:, [0, 1]]

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
                                  approx_tool_rots                                              # [NE, 4]
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

        current_object_goal_distance = LA.vector_norm(self.goal_pos_xy - self.object_pos_xy, ord=2, dim=1)
        self.initial_object_goal_distance[initialized_idx] = current_object_goal_distance[initialized_idx]

        # Object-Goal Distance Reward
        object_goal_distance_reward = torch.where(
            current_object_goal_distance < self.initial_object_goal_distance,
            self.relu(-(current_object_goal_distance - self.initial_object_goal_distance) / (self.initial_object_goal_distance + 1e-8)),
            torch.zeros_like(current_object_goal_distance)  # No reward if distance increased
        )

        # Object movement
        object_movement = self.object_pos_xy - self.prev_object_pos_xy
        object_moved_norm = LA.vector_norm(object_movement, ord=2, dim=1)

        # End-effector movement and velocity
        ee_movement = self.abs_flange_pos[:, :2] - self.prev_flange_pos[:, :2]
        ee_moved_norm = LA.vector_norm(ee_movement, ord=2, dim=1)

        # Movement towards goal
        # How aligned the object’s movement is with the direction towards the goal. It's essentially calculating the dot product of the movement vector and the direction to the goal.
        # The denominator calculates the magnitude of the vector pointing from the previous object position to the goal.
        movement_towards_goal = torch.sum(object_movement * (self.goal_pos_xy - self.prev_object_pos_xy), dim=1) / (LA.norm(self.goal_pos_xy - self.prev_object_pos_xy, ord=2, dim=1) + 1e-8)
        # Reward for movement towards goal, penalize movement away from goal
        movement_reward = torch.where(movement_towards_goal > 0,
                                      movement_towards_goal * 2.0,  # Positive reward for moving towards goal
                                      movement_towards_goal * 4.0)  # Stronger negative reward for moving away from goal

        # Tool-Object Interaction Reward
        object_moved = torch.linalg.vector_norm(object_movement, ord=2, dim=1) > 1e-4  # Threshold to detect movement
        interaction_reward = object_moved.float() * 0.1  # Small reward for moving the object

        # End-effector Movement Efficiency Reward
        efficiency_reward = torch.where(ee_moved_norm > 0,
                                        object_moved_norm / (ee_moved_norm + 1e-8),
                                        torch.zeros_like(ee_moved_norm))
        efficiency_reward = torch.clamp(efficiency_reward, 0, 1) * 0.5

        # Relative velocity between end-effector and object
        object_velocity = object_movement / self.dt
        ee_velocity = ee_movement / self.dt
        relative_velocity = torch.linalg.vector_norm(ee_velocity - object_velocity, ord=2, dim=1)
        relative_velocity_panaly = -relative_velocity * 0.1  # Penalize high relative velocity


        # Combine rewards
        total_reward = (
            object_goal_distance_reward * 2.0 +
            movement_reward +
            interaction_reward +
            efficiency_reward +
            relative_velocity_panaly            
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
    

    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        reset = torch.zeros_like(self.reset_buf)

        # # workspace regularization
        reset = torch.where(self.abs_flange_pos[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 1] > self.y_max, ones, reset)
        reset = torch.where(self.abs_flange_pos[:, 2] > 0.5, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 1] > self.y_max, ones, reset)
        # reset = torch.where(self.object_pos_xy[:, 2] > 0.5, ones, reset)    # prevent unexpected object bouncing

        # object reached
        reset = torch.where(self.done_envs, ones, reset)

        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, reset)


    def visualize_pcd(self,
                      tool_pcd_transformed, cropped_tool_pcd,
                      tool_pos, tool_rot,
                      imaginary_grasping_point,
                      real_grasping_point,
                      tool_tip_point,
                      first_principal_axes,
                      approx_tool_rot,
                      object_pcd_transformed,
                      base_pos, base_rot,
                      object_pos, object_rot,
                      flange_pos, flange_rot,
                      goal_pos,
                      base_coord='flange',  # robot_base or flange
                      view_idx=0):
        tool_pos_np = tool_pos[view_idx].cpu().numpy()
        tool_rot_np = tool_rot[view_idx].cpu().numpy()
        base_pos_np = base_pos[view_idx].cpu().numpy()
        base_rot_np = base_rot[view_idx].cpu().numpy()
        obj_pos_np = object_pos[view_idx].cpu().numpy()
        obj_rot_np = object_rot[view_idx].cpu().numpy()
        flange_pos_np = flange_pos[view_idx].detach().cpu().numpy()
        flange_rot_np = flange_rot[view_idx].detach().cpu().numpy()

        world_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))

        if base_coord == 'flange':
            flange_coord = world_coord
            base_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))
            # Visualize base coordination
            base_rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(base_rot_np)
            T_base = np.eye(4)
            T_base[:3, :3] = base_rot_matrix
            T_base[:3, 3] = base_pos_np
            base_coord = base_coord.transform(T_base)

        else:
            base_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))
            # Visualize flange coordination
            flange_rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(flange_rot_np)
            T_flange = np.eye(4)
            T_flange[:3, :3] = flange_rot_matrix
            T_flange[:3, 3] = flange_pos_np
            flange_coord = deepcopy(base_coord).transform(T_flange)



        # Create a square plane
        flange_rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(flange_rot_np)
        plane_normal = flange_rot_matrix[:, 0]  # x-axis of flange coordinate system
        plane_center = flange_pos_np
        # Create a square plane
        plane_size = 0.2
        plane_points = [
            plane_center + plane_size * (flange_rot_matrix[:, 1] + flange_rot_matrix[:, 2]),
            plane_center + plane_size * (flange_rot_matrix[:, 1] - flange_rot_matrix[:, 2]),
            plane_center + plane_size * (-flange_rot_matrix[:, 1] - flange_rot_matrix[:, 2]),
            plane_center + plane_size * (-flange_rot_matrix[:, 1] + flange_rot_matrix[:, 2])
        ]
        plane_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

        yz_plane = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(plane_points),
            lines=o3d.utility.Vector2iVector(plane_lines)
        )
        yz_plane.paint_uniform_color([0.5, 0.5, 0])  # Yellow for yz-plane
        # Visualize full tool point cloud
        tool_transformed_pcd_np = tool_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
        tool_transformed_point_cloud = o3d.geometry.PointCloud()
        tool_transformed_point_cloud.points = o3d.utility.Vector3dVector(tool_transformed_pcd_np)
        T_t = np.eye(4)
        T_t[:3, :3] = tool_rot_np
        T_t[:3, 3] = tool_pos_np
        tool_coord = deepcopy(world_coord).transform(T_t)

        # Visualize grasping point
        imaginary_grasping_point_np = imaginary_grasping_point[view_idx].detach().cpu().numpy()
        imaginary_grasping_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        imaginary_grasping_sphere.paint_uniform_color([0, 1, 1])  # Cyan color for grasping point
        imaginary_grasping_sphere.translate(imaginary_grasping_point_np)

        # Visualize tool tip point
        tool_tip_point_np = tool_tip_point[view_idx].detach().cpu().numpy()
        tool_tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        tool_tip_sphere.paint_uniform_color([0, 0, 1])  # Blue color for tool tip
        tool_tip_sphere.translate(tool_tip_point_np)

        # Visualize cropped point cloud
        cropped_pcd_np = cropped_tool_pcd[view_idx].squeeze(0).detach().cpu().numpy()
        cropped_point_cloud = o3d.geometry.PointCloud()
        cropped_point_cloud.points = o3d.utility.Vector3dVector(cropped_pcd_np)

        # Visualize real grasping point
        real_grasping_point_np = real_grasping_point[view_idx].detach().cpu().numpy()
        real_grasping_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        real_grasping_sphere.paint_uniform_color([1, 0, 1])  # Magenta color for real grasping point
        real_grasping_sphere.translate(real_grasping_point_np)

        # Principal axis
        principal_axis = first_principal_axes[view_idx, :].cpu().numpy()
        mean_point = cropped_pcd_np.mean(axis=0)
        start_point = mean_point - principal_axis * 0.5
        end_point = mean_point + principal_axis * 0.5
        # Create line set for the principal axis
        pca_line = o3d.geometry.LineSet()
        pca_line.points = o3d.utility.Vector3dVector([start_point, end_point])
        pca_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        pca_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Yellow color for principal axis

        # Quaternion to rotation matrix
        approx_tool_rot_np = approx_tool_rot[view_idx].cpu().numpy()
        rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(approx_tool_rot_np)
        
        # Create arrow for the principal axis
        ori_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.1, cone_height=0.02)
        ori_arrow.paint_uniform_color([1, 0, 1])  # Magenta color for orientation arrow
        T_arrow = np.eye(4)
        T_arrow[:3, :3] = rot_matrix
        ori_arrow.transform(T_arrow)
        ori_arrow.translate(real_grasping_point_np)  # Translate the arrow to the grasping point

        obj_transformed_pcd_np = object_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
        obj_transformed_point_cloud = o3d.geometry.PointCloud()
        obj_transformed_point_cloud.points = o3d.utility.Vector3dVector(obj_transformed_pcd_np)
        T_o = np.eye(4)

        # R_b = tgt_rot_np.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        T_o[:3, :3] = obj_rot_np
        # T_o[:3, :3] = R_b
        T_o[:3, 3] = obj_pos_np
        obj_coord = deepcopy(world_coord).transform(T_o)

        goal_pos_np = goal_pos[view_idx].cpu().numpy()
        goal_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.01, height=0.03)
        goal_cone.paint_uniform_color([0, 1, 0])
        T_g_p = np.eye(4)
        T_g_p[:3, 3] = goal_pos_np
        goal_position = deepcopy(goal_cone).transform(T_g_p)

        # goal_pos_xy_np = copy.deepcopy(goal_pos_np)
        # goal_pos_xy_np[2] = self.target_height
        # goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # goal_sphere.paint_uniform_color([1, 0, 0])
        # T_g = np.eye(4)
        # T_g[:3, 3] = goal_pos_xy_np
        # goal_position_xy = copy.deepcopy(goal_sphere).transform(T_g)

        o3d.visualization.draw_geometries([base_coord,
                                           tool_transformed_point_cloud,
                                           cropped_point_cloud,
                                           tool_coord,
                                           imaginary_grasping_sphere,
                                           real_grasping_sphere,
                                           tool_tip_sphere,
                                           pca_line,
                                           ori_arrow,
                                           obj_transformed_point_cloud,                                        
                                           obj_coord,
                                           flange_coord,
                                           goal_position,
                                           yz_plane,
                                        # goal_position_xy
                                        ],
                                            window_name=f'point cloud')
    


    def get_imaginary_grasping_point(self, flange_pos, flange_rot):
        # Convert quaternion to rotation matrix
        flange_rot_matrix = quaternion_to_matrix(flange_rot)

        # Y-direction in the flange's coordinate system
        y_direction = torch.tensor([0, 1, 0], device=flange_pos.device, dtype=flange_pos.dtype)
        y_direction_flange = torch.matmul(flange_rot_matrix, y_direction)

        # Imaginary grasping point, 0.16m away in the y-direction at Robotiq 2F-85
        imaginary_grasping_point = flange_pos + 0.16 * y_direction_flange

        return imaginary_grasping_point


    def crop_tool_pcd(self, tool_pcd, grasping_point, radius=0.05):
        """
        Crop the tool point cloud around the imaginary grasping point using a spherical region.

        Args:
        - tool_pcd: Tensor of shape [num_envs, num_points, 3]
        - grasping_point: Tensor of shape [num_envs, 3]
        - radius: Float, radius of the sphere to crop around the grasping point

        Returns:
        - cropped_pcd: Tensor of cropped points with shape [num_envs, num_cropped_points, 3]
        """
        # Calculate the distance from each point to the grasping point
        distances = torch.linalg.norm(tool_pcd - grasping_point.unsqueeze(1), dim=2)
        mask = distances <= radius  # Create a mask to select points within the sphere
        mask_int = mask.int()   # Convert mask to integer type for sorting

        # Find the maximum number of points that meet the condition for each environment
        max_cropped_points = mask_int.sum(dim=1).max().item()

        # Use the mask to select points and maintain the shape [num_envs, num_cropped_points, 3]
        sorted_idx = torch.argsort(mask_int, descending=True, dim=1)
        sorted_idx = sorted_idx[:, :max_cropped_points]
        cropped_pcd = torch.gather(tool_pcd, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))

        return cropped_pcd


    def apply_pca(self, cropped_pcd):
        """
        Apply PCA to the cropped point cloud to get the principal axis.

        Args:
        - cropped_pcd: Tensor of shape [num_envs, num_cropped_points, 3]

        Returns:
        - principal_axes: Tensor of shape [num_envs, 3]
        """
        num_envs, num_cropped_points, _ = cropped_pcd.shape

        # Center the data
        mean = torch.mean(cropped_pcd, dim=1, keepdim=True)
        centered_data = cropped_pcd - mean

        # # Reshape the data for batched SVD
        # centered_data_flat = centered_data.view(num_envs, num_cropped_points, 3)
        # U, S, V = torch.svd(centered_data_flat) # Perform batched SVD
        # principal_axes = V[:, :, 0] # Extract the first principal component
        # return principal_axes

        # Perform PCA
        _, _, V = torch.svd(centered_data)

        return V[:, :, 0], mean # Return the first principal axis and the mean of the cropped point cloud

    
    def get_tool_tip_position(self, imaginary_grasping_point, tool_pcd):
        B, N, _ = tool_pcd.shape
        # calculate farthest distance and idx from the tool to the goal
        diff = tool_pcd - imaginary_grasping_point[:, None, :]
        distance = diff.norm(dim=2)  # [B, N]

        # Find the index and value of the farthest point from the base coordinate
        farthest_idx = distance.argmax(dim=1)  # [B]
        # farthest_val = distance.gather(1, farthest_idx.unsqueeze(1)).squeeze(1)  # [B]
        tool_end_point = tool_pcd.gather(1, farthest_idx.view(B, 1, 1).expand(B, 1, 3)).squeeze(1).squeeze(1)  # [B, 3]
        
        return tool_end_point



    def find_intersection_with_yz_plane(self, flange_pos, flange_rot, principal_axis, grasping_point):
        x_direction = torch.tensor([1, 0, 0], device=flange_pos.device, dtype=flange_pos.dtype)
        x_direction_flange = torch.matmul(quaternion_to_matrix(flange_rot), x_direction)
        t = torch.dot((flange_pos - grasping_point), x_direction_flange) / torch.dot(principal_axis, x_direction_flange)
        intersection_point = grasping_point + t * principal_axis
        return intersection_point

    def calculate_tool_orientation(self, principal_axes, tool_tip_points, grasping_points):
        """
        Calculate the tool orientation based on the principal axes and the vector from grasping point to tool tip.

        Args:
        - principal_axes: Tensor of shape [num_envs, 3]
        - tool_tip_points: Tensor of shape [num_envs, 3]
        - grasping_points: Tensor of shape [num_envs, 3]

        Returns:
        - approx_tool_rot: Tensor of shape [num_envs, 4]
        """
        # Use the first principal axis as the primary axis
        primary_axes = principal_axes
        opposite_axes = -principal_axes

        # Vector from grasping point to tool tip
        vector_grasp_to_tip = tool_tip_points - grasping_points
        vector_grasp_to_tip = F.normalize(vector_grasp_to_tip, dim=1)

        # Calculate the dot product between the primary axis and the vector from grasping point to tool tip
        dot_product_primary = torch.sum(primary_axes * vector_grasp_to_tip, dim=1)
        dot_product_opposite = torch.sum(opposite_axes * vector_grasp_to_tip, dim=1)

        selected_axes = torch.where(dot_product_primary.unsqueeze(1) > 0, primary_axes, opposite_axes)

        # Calculate rotation from [0, 0, 1] to selected_axes
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=selected_axes.device).expand_as(selected_axes)
        
        # Compute the rotation axis
        rotation_axis = torch.cross(z_axis, selected_axes, dim=1)
        rotation_axis_norm = torch.norm(rotation_axis, dim=1, keepdim=True)
        
        # Handle the case where selected_axes is already [0, 0, 1] or [0, 0, -1]
        rotation_axis = torch.where(rotation_axis_norm > 1e-6, rotation_axis / rotation_axis_norm, torch.tensor([1.0, 0.0, 0.0], device=selected_axes.device).expand_as(selected_axes))
        
        # Compute the rotation angle
        cos_angle = torch.sum(z_axis * selected_axes, dim=1)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        
        # Construct the quaternion
        approx_tool_rot = torch.zeros((selected_axes.shape[0], 4), device=selected_axes.device)
        approx_tool_rot[:, 0] = torch.cos(angle / 2)
        approx_tool_rot[:, 1:] = rotation_axis * torch.sin(angle / 2).unsqueeze(1)
        return approx_tool_rot


    def get_real_grasping_point(self, flange_pos, flange_rot, principal_axes, cropped_pcd_mean):
        """
        Calculate the real grasping point as the intersection of the yz-plane 
        in the flange coordinate system and the line along the principal axis
        that passes through the mean of the cropped point cloud.

        Args:
        - flange_pos: Tensor of shape [num_envs, 3] representing flange positions
        - flange_rot: Tensor of shape [num_envs, 4] representing flange rotations as quaternions
        - principal_axes: Tensor of shape [num_envs, 3] representing the first principal axes
        - cropped_pcd_mean: Tensor of shape [num_envs, 1, 3] representing the mean of cropped point clouds

        Returns:
        - real_grasping_point: Tensor of shape [num_envs, 3] representing the real grasping points
        """
        # Convert flange rotation quaternion to rotation matrix
        flange_rot_matrix = quaternion_to_matrix(flange_rot)

        # Get the x-axis of the flange coordinate system
        flange_x_axis = flange_rot_matrix[:, :, 0]

        # Remove the extra dimension from cropped_pcd_mean
        cropped_pcd_mean = cropped_pcd_mean.squeeze(1)

        # Calculate the parameter t
        # The plane equation is: (p - flange_pos) · flange_x_axis = 0
        # The line equation is: p = cropped_pcd_mean + t * principal_axis
        # Substituting the line equation into the plane equation:
        # ((cropped_pcd_mean + t * principal_axis) - flange_pos) · flange_x_axis = 0
        # Solving for t: t = ((flange_pos - cropped_pcd_mean) · flange_x_axis) / (principal_axis · flange_x_axis)
        
        numerator = torch.sum((flange_pos - cropped_pcd_mean) * flange_x_axis, dim=1)
        denominator = torch.sum(principal_axes * flange_x_axis, dim=1)
        
        # Handle cases where the principal axis is parallel to the yz-plane
        t = torch.where(
            torch.abs(denominator) > 1e-6,
            numerator / denominator,
            torch.zeros_like(numerator)
        )

        # Calculate the intersection point
        real_grasping_point = cropped_pcd_mean + t.unsqueeze(1) * principal_axes

        return real_grasping_point


    def create_ee_transform(self, flange_pos, flange_rot):
        """
        Create a transformation matrix from the end-effector frame to the robot base frame.
        
        Args:
        - flange_pos: Tensor of shape [num_envs, 3] representing the end-effector positions
        - flange_rot: Tensor of shape [num_envs, 4] representing the end-effector orientations as quaternions
        
        Returns:
        - transform: Tensor of shape [num_envs, 4, 4] representing the transformation matrices
        """
        num_envs = flange_pos.shape[0]
        transform = torch.eye(4, device=flange_pos.device).unsqueeze(0).repeat(num_envs, 1, 1)

        # Set the rotation part of the transform (transpose of the original rotation)
        rotation_matrix = quaternion_to_matrix(flange_rot)
        transform[:, :3, :3] = rotation_matrix.transpose(1, 2)
        # Set the translation part of the transform
        transform[:, :3, 3] = -torch.bmm(rotation_matrix.transpose(1, 2), flange_pos.unsqueeze(2)).squeeze(2)

        # rotation_matrix = quaternion_to_matrix(flange_rot)
        # transform[:, :3, :3] = rotation_matrix
        # transform[:, :3, 3] = flange_pos

        return transform
    
    
    def transform_points(self, points, transform):
        """
        Apply transformation to points.
        
        Args:
        - points: Tensor of shape [num_envs, num_points, 3]
        - transform: Tensor of shape [num_envs, 4, 4]
        
        Returns:
        - transformed_points: Tensor of shape [num_envs, num_points, 3]
        """
        num_envs, num_points, _ = points.shape
        # Homogeneous coordinates
        points_homogeneous = torch.cat([points, torch.ones(num_envs, num_points, 1, device=points.device)], dim=-1)
        
        # Apply transformation
        transformed_points = torch.bmm(points_homogeneous, transform.transpose(1, 2))
        
        return transformed_points[:, :, :3]
    

    def transform_pose(self, position, orientation, transform):
        """
        Apply transformation to a pose (position and orientation).
        
        Args:
        - position: Tensor of shape [num_envs, 3] (x, y, z)
        - orientation: Tensor of shape [num_envs, 4] (qw, qx, qy, qz)
        - transform: Tensor of shape [num_envs, 4, 4]
        
        Returns:
        - transformed_pose: Tensor of shape [num_envs, 7]
        """
        # position = pose[:, :3]
        # orientation = pose[:, 3:]
        
        # Transform position
        transformed_position = self.transform_points(position.unsqueeze(1), transform).squeeze(1)
        # transformed_position = -torch.bmm(transform[:, :3, :3], position.unsqueeze(2)).squeeze(2)

        # Transform orientation
        original_rot = quaternion_to_matrix(orientation)
        transformed_rot = torch.bmm(transform[:, :3, :3], original_rot)
        transformed_orientation = matrix_to_quaternion(transformed_rot)

        return transformed_position, transformed_orientation
        
    def get_base_in_flange_frame(self, flange_pos, flange_rot):
        """
        Calculate the base coordinate in the flange frame.
        
        Args:
        - flange_pos: Tensor of shape [num_envs, 3] representing flange positions in base frame
        - flange_rot: Tensor of shape [num_envs, 4] representing flange orientations as quaternions in base frame
        
        Returns:
        - base_pose: Tensor of shape [num_envs, 7] representing base pose (position + orientation) in flange frame
        """
        # # Position of base in flange frame
        # base_pos_in_flange = -flange_pos

        transform = self.create_ee_transform(flange_pos, flange_rot)
        # The base position in the flange frame is just the negative of the flange position transformed
        base_pos_in_flange = -torch.bmm(transform[:, :3, :3], flange_pos.unsqueeze(2)).squeeze(2)
        # base_pos_in_flange = self.transform_points(flange_pos.unsqueeze(1), transform).squeeze(1)


        # Orientation of base in flange frame
        flange_rot_matrix = quaternion_to_matrix(flange_rot)
        base_rot_in_flange = flange_rot_matrix.transpose(1, 2)
        base_quat_in_flange = matrix_to_quaternion(base_rot_in_flange)

        return base_pos_in_flange, base_quat_in_flange
    

    def axis_to_quaternion(self, axis):
        """
        Convert an axis to a quaternion representation.
        
        Args:
        - axis: Tensor of shape [num_envs, 3] representing the principal axes
        
        Returns:
        - quaternions: Tensor of shape [num_envs, 4] representing the quaternions
        """
        # Ensure the axis is normalized
        axis = F.normalize(axis, dim=1)
        
        # Our reference vector (typically the z-axis)
        reference = torch.tensor([0.0, 0.0, 1.0], device=axis.device).expand_as(axis)
        
        # Compute the rotation axis
        rotation_axis = torch.cross(reference, axis, dim=1)
        rotation_axis_norm = torch.norm(rotation_axis, dim=1, keepdim=True)
        
        # Handle cases where the axis is parallel to the reference vector
        rotation_axis = torch.where(
            rotation_axis_norm > 1e-6,
            rotation_axis / rotation_axis_norm,
            torch.tensor([1.0, 0.0, 0.0], device=axis.device).expand_as(axis)
        )
        
        # Compute the rotation angle
        cos_angle = torch.sum(reference * axis, dim=1)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        
        # Construct the rotation matrix
        K = torch.zeros((axis.shape[0], 3, 3), device=axis.device)
        K[:, 0, 1] = -rotation_axis[:, 2]
        K[:, 0, 2] = rotation_axis[:, 1]
        K[:, 1, 0] = rotation_axis[:, 2]
        K[:, 1, 2] = -rotation_axis[:, 0]
        K[:, 2, 0] = -rotation_axis[:, 1]
        K[:, 2, 1] = rotation_axis[:, 0]
        
        rotation_matrix = torch.eye(3, device=axis.device).unsqueeze(0).expand(axis.shape[0], 3, 3) + \
                        torch.sin(angle).unsqueeze(-1).unsqueeze(-1) * K + \
                        (1 - torch.cos(angle).unsqueeze(-1).unsqueeze(-1)) * torch.bmm(K, K)
        
        # Convert rotation matrix to quaternion
        quaternions = matrix_to_quaternion(rotation_matrix)
        
        return quaternions
    

    def transform_principal_axis(self, transformation_matrix, principal_axis):
        """
        Transform a principal axis using a transformation matrix.

        Args:
        - transformation_matrix: Tensor of shape [num_envs, 4, 4] representing the transformation matrices
        - principal_axis: Tensor of shape [num_envs, 3] representing the principal axes

        Returns:
        - transformed_axis: Tensor of shape [num_envs, 3] representing the transformed principal axes
        """
        # Ensure the principal axis is a unit vector
        principal_axis = F.normalize(principal_axis, dim=1)

        # Add a homogeneous coordinate to the principal axis
        homogeneous_axis = torch.cat([principal_axis, torch.ones(principal_axis.shape[0], 1, device=principal_axis.device)], dim=1)

        # Apply the transformation
        transformed_homogeneous = torch.bmm(transformation_matrix, homogeneous_axis.unsqueeze(-1)).squeeze(-1)

        # Extract the transformed axis (first 3 components) and normalize
        transformed_axis = F.normalize(transformed_homogeneous[:, :3], dim=1)

        return transformed_axis


    

    # Helper function to calculate rotation matrix from two vectors
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))
        return rotation_matrix