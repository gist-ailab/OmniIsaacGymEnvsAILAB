import torch
import numpy as np
import open3d as o3d
import os

from omni.isaac.core.objects import DynamicCylinder, DynamicCone, DynamicSphere, DynamicCuboid
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omniisaacgymenvs.robots.articulations.ur5e_tool.ur5e_tool import UR5eTool

def get_robot(robot_name,
              sim_config,
              default_zero_env_path,
              translation=torch.tensor([0.0, 0.0, 0.0])):
    ur5e_tool = UR5eTool(prim_path=default_zero_env_path + f"/{robot_name}",
                         translation=translation,
                         orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                         name=robot_name)
    sim_config.apply_articulation_settings(robot_name,
                                           get_prim_at_path(ur5e_tool.prim_path),
                                           sim_config.parse_actor_config(robot_name))

def get_object(name,
               sim_config,
               default_zero_env_path,
               translation=torch.tensor([0.0, 0.0, 0.0])):
    target = DynamicCylinder(prim_path=default_zero_env_path + f"/{name}",
                             name=f"{name}",
                            #  translation=translation,
                             radius=0.04,
                             height=0.05,
                             density=1,
                             color=torch.tensor([255, 0, 0]),
                             mass=1,
                            #  physics_material=PhysicsMaterial(
                            #                                   prim_path="/World/physics_materials/target_material",
                            #                                   static_friction=0.05, dynamic_friction=0.6)
                             physics_material=PhysicsMaterial(
                                 prim_path=f"/World/physics_materials/{name}_material",
                                 static_friction=0.02, dynamic_friction=0.3)
                            )
    
    sim_config.apply_articulation_settings(f"{name}",
                                           get_prim_at_path(target.prim_path),
                                           sim_config.parse_actor_config(f"{name}"))


def get_cube(name,
             sim_config,
             default_zero_env_path,
             translation=torch.tensor([0.0, 0.0, 0.0])):
    cube = DynamicCuboid(prim_path=default_zero_env_path + f"/{name}",
                         name=f"{name}",
                        #  translation=translation,
                         size=0.04,
                         density=1,
                         color=torch.tensor([255, 0, 0]),
                         mass=1,
                            #  physics_material=PhysicsMaterial(
                            #                                   prim_path="/World/physics_materials/target_material",
                            #                                   static_friction=0.05, dynamic_friction=0.6)
                         physics_material=PhysicsMaterial(
                            prim_path=f"/World/physics_materials/{name}_material",
                            static_friction=0.02, dynamic_friction=0.3)
                                )
    sim_config.apply_articulation_settings("cube",
                                           get_prim_at_path(cube.prim_path),
                                           sim_config.parse_actor_config("cube"))

def get_goal(name, sim_config, default_zero_env_path):
    goal = DynamicCone(prim_path=default_zero_env_path + f"/{name}",
                        name=f"{name}",
                        radius=0.015,
                        height=0.03,
                        color=torch.tensor([0, 255, 0]))
    sim_config.apply_articulation_settings(f"{name}",
                                           get_prim_at_path(goal.prim_path),
                                           sim_config.parse_actor_config(f"{name}"))
    goal.set_collision_enabled(False)