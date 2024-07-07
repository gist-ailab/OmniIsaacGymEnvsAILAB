import torch
from torch import linalg as LA
import numpy as np
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

from omni.isaac.core.prims import RigidPrimView, RigidContactView
from omni.isaac.core.utils.types import ArticulationActions
from omni.isaac.core.materials.physics_material import PhysicsMaterial

from skrl.utils import omniverse_isaacgym_utils

import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_quaternion, quaternion_multiply

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
        # self.robot_list = ['ur5e_tool']
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
        
        self.PointcloudWriter = PointcloudWriter
        self.PointcloudListener = PointcloudListener

        # Solving I.K. with cuRobo
        self.init_cuRobo()
        

        RLTask.__init__(self, name, env)


    def init_cuRobo(self):
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

        default_translations=torch.tensor([0.0, 0.0, 10.0]).repeat(self._num_envs,1)
        visibilities = torch.tensor([False]).repeat(self._num_envs)
        
        for name in self.robot_list:
            # Create an instance variable for each name in robot_list
            setattr(self, f"_{name}", UR5eView(prim_paths_expr=f"/World/envs/.*/{name}",
                                                name=f"{name}_view",
                                                translations=default_translations,
                                                visibilities=visibilities))
            scene.add(getattr(self, f"_{name}"))
            scene.add(getattr(self, f"_{name}")._flanges)
            scene.add(getattr(self, f"_{name}")._tools)
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


    def post_reset(self):
        self.num_robot_dofs = self.ref_robot.num_dof  # all robots have the same dof
        
        dof_limits = self.ref_robot.get_dof_limits()  # every robot has the same dof limits

        # initialize robot's local poses to be above the ground and distinguish the visibility along with the env number
        entire_env_ids = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        sub_env_size = len(entire_env_ids) // len(self.robot_list)  # Calculate the size of each sub-env size
        remainder = len(entire_env_ids) % len(self.robot_list)  # Calculate the remainder
        self.separated_envs = []    # Initialize the result list

        start = 0
        for i, _ in enumerate(self.robot_list):
            end = start + sub_env_size + (1 if i < remainder else 0)
            self.separated_envs.append(entire_env_ids[start:end])
            start = end

        for idx, sub_envs in enumerate(self.separated_envs):
            robot_name = self.robot_list[idx]
            pos = torch.tensor([0.0, 0.0, 0.0], device=self._device).repeat(len(sub_envs),1)
            env_pos = self._env_pos[sub_envs]
            getattr(self, f"_{robot_name}").set_world_poses(pos+env_pos, indices=sub_envs)

            # Set visibilities for the current subset of environments
            visibilities = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            visibilities[sub_envs] = True
            getattr(self, f"_{robot_name}").set_visibilities(visibilities=visibilities)

        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)
        self.zero_joint_velocities = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)
        
        self._object.enable_rigid_body_physics()
        self._object.enable_gravities()

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)


    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)

        if self._control_space == "joint":
            dof_targets = self.robot_dof_targets[:, :6] + self.robot_dof_speed_scales[:6] * self.dt * self.actions * self._action_scale

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
            '''jacobian : (self._num_envs, num_of_bodies-1, wrench, num of joints)
            num_of_bodies - 1 due to start from 0 index
            '''
            dof_targets = self.robot_dof_targets[:, :6] + delta_dof_pos[:, :6]

        self.robot_dof_targets[:, :6] = torch.clamp(dof_targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])

        self.robot_dof_targets[:, 7] = torch.deg2rad(torch.tensor(self.tool_rot_x, device=self._device))
        self.robot_dof_targets[:, 8] = torch.deg2rad(torch.tensor(self.tool_rot_z, device=self._device))
        self.robot_dof_targets[:, 9] = torch.deg2rad(torch.tensor(self.tool_rot_y, device=self._device))
        ### TODO: 나중에는 self.reset_idx에 tool pose가 episode마다 random position으로 바뀌도록 할 것. 환경의 index를 이용해야 할 것 같다.
        
        for idx, sub_envs in enumerate(self.separated_envs):
            '''
            Caution:
             pre_physics_step에서 set_joint_positions는 절대 쓰면 안 된다.
             set_joint_positions는 controller 없이 joint position을 바로 설정하는 함수이다. 이걸 쓰면 로봇이 덜덜 떠는 방식으로 움직임.
             set_joint_position_targets: Set the joint position targets for the implicit Proportional-Derivative (PD) controllers
             apply_action: apply multiple targets (position, velocity, and/or effort) in the same call
             (effort control means force/torque control)
            '''
            action = ArticulationActions(joint_positions=self.robot_dof_targets[sub_envs])
            getattr(self, f"_{self.robot_list[idx]}").apply_action(action, indices=sub_envs)
        # self._targets.enable_rigid_body_physics()
        # self._targets.enable_rigid_body_physics(indices=env_ids_int32)
        # self._targets.enable_gravities(indices=env_ids_int32)

    def split_env_ids(self, env_ids):
        # 결과를 저장할 리스트 초기화
        separated_envs = copy.deepcopy(self.empty_separated_envs)
        
        # env_ids를 정렬
        sorted_env_ids = torch.sort(env_ids)[0]
        
        # env_ids를 순서대로 그룹에 할당
        group_size = (len(self.total_env_ids) + self.robot_num - 1) // self.robot_num
        for i, env_id in enumerate(sorted_env_ids):
            group_idx = ((env_id.item() % len(self.total_env_ids)) // group_size) % self.robot_num
            separated_envs[group_idx] = torch.cat([separated_envs[group_idx], env_id.unsqueeze(0)])
        
        return separated_envs


    def reset_idx(self, env_ids) -> None:
        env_ids = env_ids.to(dtype=torch.int32)

        # Split env_ids using the split_env_ids function
        separated_envs = self.split_env_ids(env_ids)

        # reset object
        position = torch.tensor(self._object_position, device=self._device)
        object_pos = position.repeat(len(env_ids),1)
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        object_ori = orientation.repeat(len(env_ids),1)
        self._object.set_world_poses(object_pos + self._env_pos[env_ids],
                                      object_ori,
                                      indices=env_ids)
        
        # reset goal
        goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)
        goal_mark_pos = goal_mark_pos.repeat(len(env_ids),1)
        self._goal.set_world_poses(goal_mark_pos + self._env_pos[env_ids], indices=env_ids)
        goal_pos = self._goal.get_local_poses()
        self.goal_pos_xy = goal_pos[0][:, [0, 1]]
        
        # object_pos 에서 y축으로 0.2만큼 뒤로 이동한 위치로 flange를 이동시키기
        flange_pos = deepcopy(object_pos)
        flange_pos[:, 0] -= 0.2    # x
        flange_pos[:, 1] -= 0.3    # y
        flange_pos[:, 2] = 0.4     # z

        # Extract the x and y coordinates
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

        initialized_pos = Pose(flange_pos, flange_ori, name="tool0")
        target_dof_pos = torch.empty(0).to(device=self._device)
        for i in range(initialized_pos.batch):
            result = self.ik_solver.solve_single(initialized_pos[i])    # Solving I.K. with cuRobo
            if result.success:
                target_dof_pos = torch.cat((target_dof_pos, result.solution[result.success]), dim=0)
            else:
                print(f"IK solver failed. Initialize a robot in env {env_ids[i]} with default pose.")
                target_dof_pos = torch.cat((target_dof_pos, self.robot_default_dof_pos[:6].unsqueeze(0)), dim=0)
        
        self.robot_dof_targets[env_ids, :6] = torch.clamp(target_dof_pos,
                                                          self.robot_dof_lower_limits[:6].repeat(len(env_ids),1),
                                                          self.robot_dof_upper_limits[:6].repeat(len(env_ids),1))
        
        for idx, sub_envs in enumerate(separated_envs):
            if sub_envs.numel() == 0:
                continue  # Skip if sub_envs is empty
            robot_name = self.robot_list[idx]
            assert sub_envs.max().item() < self._num_envs, "Sub-env index out of bounds"
            getattr(self, f"_{robot_name}").set_joint_positions(self.robot_dof_targets[sub_envs], indices=sub_envs)
            getattr(self, f"_{robot_name}").set_joint_velocities(self.zero_joint_velocities[sub_envs], indices=sub_envs)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def get_observations(self) -> dict:
        self._env.render()  # add for get point cloud on headless mode
        self.step_num += 1
        ''' retrieve point cloud data from all render products '''
        # tasks/utils/pcd_writer.py 에서 pcd sample하고 tensor로 변환해서 가져옴
        # pointcloud = self.pointcloud_listener.get_pointcloud_data()

        tools_pcd_flattened = torch.empty(0).to(device=self._device)
        objects_pcd_flattened = torch.empty(0).to(device=self._device)
        object_pcd_concat = torch.empty(0).to(device=self._device)  # concatenate target point cloud for getting xyz position
        self.goal_pos = torch.empty(0).to(device=self._device)  # concatenate goal position for getting xy position
        robots_dof_pos = torch.empty(0).to(device=self._device)
        robots_dof_vel = torch.empty(0).to(device=self._device)

        for idx, sub_envs in enumerate(self.separated_envs):
            robot_name = self.robot_list[idx]
            self.flange_pos[self.separated_envs[idx], :] = getattr(self, f'_{robot_name}')._flanges.get_local_poses()[0][self.separated_envs[idx], :]
            self.flange_rot[self.separated_envs[idx], :] = getattr(self, f'_{robot_name}')._flanges.get_local_poses()[1][self.separated_envs[idx], :]
            
            tool_pos = getattr(self, f'_{robot_name}')._tools.get_local_poses()[0][self.separated_envs[idx], :]
            tool_rot_quaternion = getattr(self, f'_{robot_name}')._tools.get_local_poses()[1][self.separated_envs[idx], :]
            tool_rot = quaternion_to_matrix(tool_rot_quaternion)
            object_pos = self._object.get_local_poses()[0][self.separated_envs[idx], :]
            object_rot_quaternion = self._object.get_local_poses()[1][self.separated_envs[idx], :]
            object_rot = quaternion_to_matrix(object_rot_quaternion)

            tool_name = robot_name.split('_')[1]
            # concat tool point cloud per robot
            tool_pcd_transformed = pcd_registration(getattr(self, f"{tool_name}_pcd")[sub_envs],
                                                    tool_pos,
                                                    tool_rot,
                                                    len(sub_envs),
                                                    device=self._device)
            tool_pcd_flattend = tool_pcd_transformed.contiguous().view(len(sub_envs), -1)
            tools_pcd_flattened = torch.cat((tools_pcd_flattened, tool_pcd_flattend), dim=0)

            # concat target point cloud per robot
            # TODO: if the environment has multiple objects, the different object pcd should be loaded in the loop
            object_pcd_transformed = pcd_registration(self.object_pcd[sub_envs],
                                                      object_pos,
                                                      object_rot,
                                                      len(sub_envs),
                                                      device=self._device)
            object_pcd_concat = torch.cat((object_pcd_concat, object_pcd_transformed), dim=0)   # concat for calculating xyz position
            object_pcd_flattend = object_pcd_transformed.contiguous().view(len(sub_envs), -1)
            objects_pcd_flattened = torch.cat((object_pcd_flattend, objects_pcd_flattened), dim=0)

            self.goal_pos = torch.cat((self.goal_pos, self._goal.get_local_poses()[0][self.separated_envs[idx]]), dim=0)

            # get robot dof position and velocity from 1st to 6th joint
            robot_joint_positions = getattr(self, f'_{robot_name}').get_joint_positions(clone=False)[sub_envs, 0:6]
            robot_joint_velocity = getattr(self, f'_{robot_name}').get_joint_velocities(clone=False)[sub_envs, 0:6]
            robots_dof_pos = torch.cat((robots_dof_pos, robot_joint_positions), dim=0)
            robots_dof_vel = torch.cat((robots_dof_vel, robot_joint_velocity), dim=0)
            # rest of the joints are not used for control. They are fixed joints at each episode.

            if self._control_space == "cartesian":
                # set jacobians from each sub-environment robot
                self.jacobians[sub_envs] = getattr(self, f'_{robot_name}').get_jacobians(clone=False)[sub_envs]

        # self.visualize_pcd(tool_pcd_transformed, object_pcd_transformed,
        #                    tool_pos, tool_rot, object_pos, object_rot,
        #                    self.goal_pos)
        self.object_pos_xyz = torch.mean(object_pcd_concat, dim=1)
        self.object_pos_xy = self.object_pos_xyz[:, [0, 1]]

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
                                  tools_pcd_flattened,                                          # [NE, N*3], point cloud
                                  objects_pcd_flattened,                                        # [NE, N*3], point cloud
                                  robots_dof_pos,                                               # [NE, 6]
                                  robots_dof_vel,                                               # [NE, 6]
                                  self.flange_pos,                                              # [NE, 3]
                                  self.flange_rot,                                              # [NE, 4]
                                  self.goal_pos_xy,                                             # [NE, 2]
                                 ), dim=1)
        return self.obs_buf


    def calculate_metrics(self) -> None:
        initialized_idx = self.progress_buf == 1    # initialized index를 통해 progress_buf가 1인 경우에만 initial distance 계산
        self.completion_reward[:] = 0.0 # reset completion reward
        current_object_goal_distance = LA.norm(self.goal_pos_xy - self.object_pos_xy, ord=2, dim=1)
        self.initial_object_goal_distance[initialized_idx] = current_object_goal_distance[initialized_idx]

        init_o_g_d = self.initial_object_goal_distance
        cur_o_g_d = current_object_goal_distance
        target_goal_distance_reward = self.relu(-(cur_o_g_d - init_o_g_d)/init_o_g_d)

        # completion reward
        self.done_envs = cur_o_g_d <= 0.1
        # completion_reward = torch.where(self.done_envs, torch.full_like(cur_t_g_d, 100.0)[self.done_envs], torch.zeros_like(cur_t_g_d))
        self.completion_reward[self.done_envs] = torch.full_like(cur_o_g_d, 300.0)[self.done_envs]

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
        reset = torch.where(self.object_pos_xy[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.object_pos_xy[:, 1] > self.y_max, ones, reset)

        # target reached
        reset = torch.where(self.done_envs, ones, reset)

        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, reset)


    def visualize_pcd(self,
                      tool_pcd_transformed,
                      object_pcd_transformed,
                      tool_pos, tool_rot,
                      object_pos, object_rot,
                      goal_pos,
                    #   farthest_idx
                      ):
        view_idx = 0

        base_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))
        tool_pos_np = tool_pos[view_idx].cpu().numpy()
        tool_rot_np = tool_rot[view_idx].cpu().numpy()
        obj_pos_np = object_pos[view_idx].cpu().numpy()
        obj_rot_np = object_rot[view_idx].cpu().numpy()
        
        tool_transformed_pcd_np = tool_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
        tool_transformed_point_cloud = o3d.geometry.PointCloud()
        tool_transformed_point_cloud.points = o3d.utility.Vector3dVector(tool_transformed_pcd_np)
        T_t = np.eye(4)
        T_t[:3, :3] = tool_rot_np
        T_t[:3, 3] = tool_pos_np
        tool_coord = copy.deepcopy(base_coord).transform(T_t)

        tool_end_point = o3d.geometry.TriangleMesh().create_sphere(radius=0.01)
        tool_end_point.paint_uniform_color([0, 0, 1])
        # farthest_pt = tool_transformed_pcd_np[farthest_idx.detach().cpu().numpy()][view_idx]
        # T_t_p = np.eye(4)
        # T_t_p[:3, 3] = farthest_pt
        # tool_tip_position = copy.deepcopy(tool_end_point).transform(T_t_p)

        obj_transformed_pcd_np = object_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
        obj_transformed_point_cloud = o3d.geometry.PointCloud()
        obj_transformed_point_cloud.points = o3d.utility.Vector3dVector(obj_transformed_pcd_np)
        T_o = np.eye(4)

        # R_b = tgt_rot_np.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        T_o[:3, :3] = obj_rot_np
        # T_o[:3, :3] = R_b
        T_o[:3, 3] = obj_pos_np
        obj_coord = copy.deepcopy(base_coord).transform(T_o)

        goal_pos_np = goal_pos[view_idx].cpu().numpy()
        goal_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.01, height=0.03)
        goal_cone.paint_uniform_color([0, 1, 0])
        T_g_p = np.eye(4)
        T_g_p[:3, 3] = goal_pos_np
        goal_position = copy.deepcopy(goal_cone).transform(T_g_p)

        # goal_pos_xy_np = copy.deepcopy(goal_pos_np)
        # goal_pos_xy_np[2] = self.target_height
        # goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # goal_sphere.paint_uniform_color([1, 0, 0])
        # T_g = np.eye(4)
        # T_g[:3, 3] = goal_pos_xy_np
        # goal_position_xy = copy.deepcopy(goal_sphere).transform(T_g)

        o3d.visualization.draw_geometries([base_coord,
                                        tool_transformed_point_cloud,
                                        obj_transformed_point_cloud,
                                        # tool_tip_position,
                                        tool_coord,
                                        obj_coord,
                                        goal_position,
                                        # goal_position_xy
                                        ],
                                            window_name=f'point cloud')