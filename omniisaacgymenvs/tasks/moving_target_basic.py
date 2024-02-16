import torch
from torch import linalg as LA
import numpy as np
import cv2
 
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")   # required by PytorchListener
# enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ur5e_tool import UR5eTool
from omniisaacgymenvs.robots.articulations.views.ur5e_view import UR5eView

import omni
from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
# from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCylinder, DynamicSphere, DynamicCuboid, VisualCone, DynamicCone
# from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
# from omni.isaac.sensor import Camera, LidarRtx, RotatingLidarPhysX
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.materials.physics_material import PhysicsMaterial

from skrl.utils import omniverse_isaacgym_utils
from pxr import Usd, UsdGeom, Gf, UsdPhysics, Semantics              # pxr usd imports used to create cube

from typing import Optional, Tuple
import asyncio

import copy


# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()    


class BasicMovingTargetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
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
        
        self.alpha = 0.4
        self.beta = 0.6
        self.scaler_a = 100
        self.scaler_b = 200
        self.eta = 0.15
        # print(self.alpha + self.beta + self.gamma + self.zeta + self.eta)
        self.punishment = -1
        self.relu = torch.nn.ReLU()

        self.stage = omni.usd.get_context().get_stage()

        # tool orientation
        self.tool_rot_x = 1.221 # 70 degree
        self.tool_rot_z = 0     # 0 degree
        self.tool_rot_y = -1.5707 # -90 degree

        # workspace 2D boundary
        self.x_min = 0.45
        self.x_max = 1.2
        self.y_min = -0.8
        self.y_max = 0.4

        # observation and action space
        # self._num_observations = 16
        # self._num_observations = 19
        self._num_observations = 29
        # self._num_observations = 24
        if self._control_space == "joint":
            self._num_actions = 6
        elif self._control_space == "cartesian":
            # self._num_actions = 3
            self._num_actions = 7   # 3 for position, 4 for rotation(quaternion)
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))

        self._flange_link = "flange_tool_rot_x"

        
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
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view")
        # goal view
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_view", reset_xform_properties=False)
        self._goals._non_root_link = True   # do not set states for kinematics

        scene.add(self._goals)
        scene.add(self._robots)
        scene.add(self._flanges)
        scene.add(self._tools)
        scene.add(self._targets)            

        self.init_data()

        return

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
        for i in range(self._num_envs):
            if scene.object_exist(f"lidar_view_{i}"):
                scene.remove(f"lidar_view_{i}", registry_only=True)

        self._robots = UR5eView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        self._flanges = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/{self._flange_link}", name="end_effector_view")
        self._tools = RigidPrimView(prim_paths_expr=f"/World/envs/.*/robot/tool", name="tool_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view")
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
        # target = DynamicCuboid(prim_path=self.default_zero_env_path + "/target",
        #                        name="target",
        #                        size=0.07,
        #                     #    size=0.1,
        #                        density=1,
        #                        color=torch.tensor([255, 0, 0]),
        #                        physics_material=PhysicsMaterial(
        #                                                         prim_path="/World/physics_materials/target_material",
        #                                                         static_friction=0.01, dynamic_friction=0.01),
        #                        )
        self.radius = 0.04
        target = DynamicCylinder(prim_path=self.default_zero_env_path + "/target",
                                 name="target",
                                 radius=self.radius,
                                 height=0.05,
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


    def get_observations(self) -> dict:
        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)

        self.flange_pos, self.flange_rot = self._flanges.get_local_poses()
        tool_pos, tool_rot = self._tools.get_local_poses()
        target_pos, target_rot = self._targets.get_local_poses()
        goal_pos, _ = self._goals.get_local_poses()
        goal_pos[:,2] = self._target_position[-1]

        ### used for is_done() method


        self.tool_pos = copy.deepcopy(tool_pos)
        self.target_pos = copy.deepcopy(target_pos)
        self.goal_pos = copy.deepcopy(goal_pos)
        self.target_goal_distance = LA.norm(target_pos - goal_pos, ord=2, dim=1)

        ### normalize flange_pos, tool_pos, target_pos, goal_pos
        self.flange_pos[:, 0] = (self.flange_pos[:, 0] - self.x_min) / (self.x_max - self.x_min)   # normalize x-axis
        self.flange_pos[:, 1] = (self.flange_pos[:, 1] - self.y_min) / (self.y_max - self.y_min)   # normalize y-axis
        tool_pos[:, 0] = (tool_pos[:, 0] - self.x_min) / (self.x_max - self.x_min)   # normalize x-axis
        tool_pos[:, 1] = (tool_pos[:, 1] - self.y_min) / (self.y_max - self.y_min)   # normalize y-axis
        target_pos[:, 0] = (target_pos[:, 0] - self.x_min) / (self.x_max - self.x_min)
        target_pos[:, 1] = (target_pos[:, 1] - self.y_min) / (self.y_max - self.y_min)
        goal_pos[:, 0] = (goal_pos[:, 0] - self.x_min) / (self.x_max - self.x_min)
        goal_pos[:, 1] = (goal_pos[:, 1] - self.y_min) / (self.y_max - self.y_min)
        ### normalize flange_pos, tool_pos, target_pos, goal_pos

        tool2target = target_pos - tool_pos
        target2goal = goal_pos - target_pos

        if self.previous_target_position == None:
            self.target_moving_distance = torch.zeros((self._num_envs), device=self._device)
            self.current_target_position = target_pos
        else:
            self.previous_target_position = self.current_target_position
            self.current_target_position = target_pos
            self.target_moving_distance = LA.norm(self.previous_target_position - self.current_target_position,
                                                  ord=2, dim=1)

        if self.previous_tool_target_distance == None:
            self.current_tool_target_distance = LA.norm(tool2target, ord=2, dim=1)
            self.initial_tool_target_distance = LA.norm(tool2target, ord=2, dim=1)
            self.previous_tool_target_distance = self.current_tool_target_distance
            self.current_target_goal_distance = LA.norm(target2goal, ord=2, dim=1)
            self.initial_target_goal_distance = LA.norm(target2goal, ord=2, dim=1)
            self.previous_target_goal_distance = self.current_target_goal_distance
        
        else:
            self.previous_tool_target_distance = self.current_tool_target_distance
            self.previous_target_goal_distance = self.current_target_goal_distance
            self.current_tool_target_distance = LA.norm(tool2target, ord=2, dim=1)
            self.current_target_goal_distance = LA.norm(target2goal, ord=2, dim=1)

        # # normalize robot_dof_pos
        # dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0   # normalized by [-1, 1]
        dof_pos_scaled = (robot_dof_pos - self.robot_dof_lower_limits) \
                        /(self.robot_dof_upper_limits - self.robot_dof_lower_limits)    # normalized by [0, 1]
        dof_pos_scaled = robot_dof_pos    # non-normalized

        # # normalize robot_dof_vel
        dof_vel_scaled = robot_dof_vel * self._dof_vel_scale
        # generalization_noise = torch.rand((dof_vel_scaled.shape[0], 6), device=self._device) + 0.5
        dof_vel_scaled = robot_dof_vel    # non-normalized

        self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        
        # robot state
        self.obs_buf[:, 1:7] = dof_pos_scaled[:, :6]
        # self.obs_buf[:, 7:13] = dof_vel_scaled[:, :6] * generalization_noise
        self.obs_buf[:, 7:13] = dof_vel_scaled[:, :6]
        self.obs_buf[:, 13:16] = self.flange_pos
        self.obs_buf[:, 16:20] = self.flange_rot

        # tool, target, goal state
        self.obs_buf[:, 20:23] = tool_pos
        # self.obs_buf[:, 23:26] = tool2target
        # self.obs_buf[:, 26:29] = target2goal
        self.obs_buf[:, 23:26] = target_pos
        self.obs_buf[:, 26:29] = goal_pos
        # self.obs_buf[:, 22] = self.current_tool_target_distance
        # self.obs_buf[:, 23] = self.current_target_goal_distance

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
            # goal_position = self.flange_pos + actions / 100.0

            # goal_position = self.flange_pos + self.actions[:, :3] / 30.0
            goal_position = self.flange_pos + self.actions[:, :3] / 10.0
            goal_orientation = self.flange_rot + self.actions[:, 3:] / 70.0
            delta_dof_pos = omniverse_isaacgym_utils.ik(
                                                        jacobian_end_effector=self.jacobians[:, self._robots.body_names.index(self._flange_link)-1, :, :],
                                                        current_position=self.flange_pos,
                                                        current_orientation=self.flange_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=goal_orientation
                                                        )
            targets = self.robot_dof_targets[:, :6] + delta_dof_pos[:, :6]

        self.robot_dof_targets[:, :6] = torch.clamp(targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])
        # self.robot_dof_targets[:, 6:] = torch.tensor(0, device=self._device, dtype=torch.float16)
        self.robot_dof_targets[:, 7] = torch.tensor(self.tool_rot_x, device=self._device)
        self.robot_dof_targets[:, 8] = torch.tensor(self.tool_rot_z, device=self._device)
        self.robot_dof_targets[:, 9] = torch.tensor(self.tool_rot_y, device=self._device)
        ### 나중에는 윗 줄을 통해 tool position이 random position으로 고정되도록 변수화. reset_idx도 확인할 것
        # self._robots.get_joint_positions()
        # self._robots.set_joint_positions(self.robot_dof_targets, indices=env_ids_int32)
        self._robots.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)
        # TOOD: self.robot_dof_targets의 나머지 value가 0인지 확인
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
        target_pos = torch.tensor(self._target_position, device=self._device)
        # ### randomize target position ###
        # # x, y randomize 는 ±0.1로 uniform random
        # z_ref = torch.abs(target_pos[2])
        # # generate uniform random values for randomizing target position
        # # reference: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        # x_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # y_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # z_rand = z_ref.repeat(len(env_ids),1)   ''' Do not randomize z position '''
        # rand = torch.cat((x_rand, y_rand, z_rand), dim=1)
        # target_pos = target_pos.repeat(len(env_ids),1) + rand
        # ### randomize target position ###
        target_pos = target_pos.repeat(len(env_ids),1)

        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        target_ori = orientation.repeat(len(env_ids),1)
        self._targets.set_world_poses(target_pos + self._env_pos[env_ids],
                                      target_ori,
                                      indices=indices)
        # self._targets.enable_rigid_body_physics()

        # reset goal
        goal_mark_pos = torch.tensor(self._goal_mark, device=self._device)
        # ### randomize goal position ###
        # # x, y randomize 는 ±0.1로 uniform random
        # z_ref = torch.abs(goal_mark_pos[2])
        # # # generate uniform random values for randomizing goal position
        # x_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # y_rand = torch.FloatTensor(len(env_ids), 1).uniform_(-0.1, 0.1).to(device=self._device)
        # z_rand = z_ref.repeat(len(env_ids),1)   ''' Do not randomize z position '''
        # rand = torch.cat((x_rand, y_rand, z_rand), dim=1)
        # goal_mark_pos = goal_mark_pos.repeat(len(env_ids),1) + rand
        # ### randomize goal position ###
        goal_mark_pos = goal_mark_pos.repeat(len(env_ids),1)
        self._goals.set_world_poses(goal_mark_pos + self._env_pos[env_ids], indices=indices)

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
        init_t_t_d = self.initial_tool_target_distance
        cur_t_t_d = self.current_tool_target_distance
        init_t_g_d = self.initial_target_goal_distance
        cur_t_g_d = self.current_target_goal_distance
        euler_num = torch.exp(torch.tensor(1., device=self._device))
        tool_target_distance_reward = self.relu(-(cur_t_t_d - init_t_t_d)/init_t_t_d)
        target_goal_distance_reward = self.relu(-(cur_t_g_d - init_t_g_d)/init_t_g_d)
        # tool_target_distance_reward = torch.exp(-(euler_num/init_t_t_d)*cur_t_t_d)
        # target_goal_distance_reward = torch.exp(-(euler_num/init_t_g_d)*cur_t_g_d)


        ## TODO: 아래 두 항은 normalize를 어떻게 해주어야 할지 감이 잘 안옴...
        ## 값이 작으므로 값을 그대로 넣어서 잘 움직일 경우 bonus로 사용하도록 함
        delta_tool_target = self.previous_tool_target_distance - self.current_tool_target_distance
        delta_target_goal = self.previous_target_goal_distance - self.current_target_goal_distance

        # # punish to go out of workspace
        # check_x_min = (self.tool_pos[:, 0] > 0.55).unsqueeze(1)
        # check_x_max = (self.tool_pos[:, 0] < 0.95).unsqueeze(1)
        # check_y_min = (self.tool_pos[:, 1] > -0.55).unsqueeze(1)
        # check_y_max = (self.tool_pos[:, 1] < 0.35).unsqueeze(1)
        # check_z_max = (self.tool_pos[:, 2] < 0.18).unsqueeze(1)
        # out_of_workspace_check = torch.cat((check_x_min, check_x_max, check_y_min, check_y_max, check_z_max), dim=1)
        # punishing_out_of_workspace = -torch.logical_not(torch.prod(out_of_workspace_check, dim=1)).float() * 3

        self.completion_reward = torch.zeros(self._num_envs).to(self._device)
        self.completion_reward[self.current_target_goal_distance<0.035] = 1
        self.rew_buf[:] = self.alpha * tool_target_distance_reward + \
                          self.beta * target_goal_distance_reward + \
                          self.completion_reward
                        #   self.scaler_a * delta_tool_target + \
                        #   self.scaler_b * delta_target_goal + \


    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        reset = torch.zeros_like(self.reset_buf)

        # # workspace regularization
        reset = torch.where(self.tool_pos[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.tool_pos[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.tool_pos[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.tool_pos[:, 1] > self.y_max, ones, reset)
        reset = torch.where(self.tool_pos[:, 2] > 0.5, ones, reset)
        reset = torch.where(self.target_pos[:, 0] < self.x_min, ones, reset)
        reset = torch.where(self.target_pos[:, 1] < self.y_min, ones, reset)
        reset = torch.where(self.target_pos[:, 0] > self.x_max, ones, reset)
        reset = torch.where(self.target_pos[:, 1] > self.y_max, ones, reset)
        reset = torch.where(self.target_pos[:, 2] > 0.5, ones, reset)

        # target reached
        reset = torch.where(self.target_goal_distance < 0.03, ones, reset)

        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, reset)