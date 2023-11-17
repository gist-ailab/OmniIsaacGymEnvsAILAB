from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class UR5eView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: str = "robot_view") -> None:
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)