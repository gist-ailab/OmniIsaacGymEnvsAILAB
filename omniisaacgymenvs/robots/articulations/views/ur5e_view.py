from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class UR5eView(ArticulationView):
    def __init__(self,
                 prim_paths_expr: str,
                 name: str = "ur5e_tool_view") -> None:
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        robot_name = name.split("_")
        robot_name = robot_name[0] + "_" + robot_name[1]

        self._flanges = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{robot_name}/tool0",
            name=f"{robot_name}_flange_view",
            reset_xform_properties=False
        )
        
        self._tools = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{robot_name}/tool",
            name=f"{robot_name}_tool_view",
            reset_xform_properties=False
        )
