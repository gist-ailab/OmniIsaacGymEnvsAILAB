# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import os, sys
import math
import numpy as np
import torch

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
# from omni.isaac.sensor import Camera
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

class UR5eTool(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "ur5e_tool",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        rgb_cam_prim_name: Optional[str] = None,    ### BSH
        depth_cam_prim_name: Optional[str] = None,  ### BSH
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0, 0.0, 0.0]) if translation is None else translation
        # self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        if self._usd_path is None:
            isaac_root_path = os.path.join(os.path.expanduser('~'), ".local/share/ov/pkg/isaac_sim-2023.1.1")
            if isaac_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            if name == "robot":
                self._usd_path = isaac_root_path + f"/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/ur5e_tool.usd"
            else:
                self._usd_path = isaac_root_path + f"/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/{name}.usd"
            


        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "base_link/shoulder_pan_joint",
            "shoulder_link/shoulder_lift_joint",
            "upper_arm_link/elbow_joint",
            "forearm_link/wrist_1_joint",
            "wrist_1_link/wrist_2_joint",
            "wrist_2_link/wrist_3_joint",
            "gripper/grasped_position",  # grasped position on handle
            "gripper_tool_yaw/gripper_revolute_yaw",
            "gripper_tool_pitch/gripper_revolute_pitch",
            "gripper_tool_roll/gripper_revolute_roll",
            "gripper_tool_prismatic/tool_prismatic",    # prismatic along rod axis
        ]


        drive_type = ["angular"] * 6 + ["linear"] + ["angular"] * 3 + ["linear"]
        # default_dof_pos = [math.degrees(x) for x in [0.0, -1.75, 1.05, -1.57, -1.57, 1.57,
        #                                              0.0, 0.0, 0.0, 0.0]]
        default_dof_pos = [-50, -40, 50, -100, -90, 130,
                            0.1, 70, 0.0, -90]
        stiffness = [400*np.pi/180] * 6
        stiffness.extend([1000000] * 5)    # stiffness for grasped tool
        damping = [80*np.pi/180] * 6
        damping.extend([100000] * 4)   # damping for grasped tool
        max_force = [87, 87, 87, 12, 12, 12,
                     100, 100, 100, 100, 100]
                    #  0, 0, 0, 0]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.61, 2.61, 2.61,
                                                #   2.61, 2.61, 2.61, 2.61]]
                                                  0, 0, 0, 0, 0]]
        # '/World/envs/env_0/robot/ure/base_link/shoulder_pan_joint'
        for i, dof in enumerate(dof_paths):
            if i > 6: continue
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])