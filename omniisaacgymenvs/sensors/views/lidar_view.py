from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.sensor")   # required by OIGE

from typing import Optional
from omni.isaac.sensor import RotatingLidarPhysX

class LidarView(RotatingLidarPhysX):
    def __init__(self,
                 prim_paths: str,
                 name: Optional[str] = "LidarView") -> None:
        super().__init__(prim_paths, name,
                         rotation_frequency = 0.0,
                        #  fov=[70, 50],
                         fov = [100, 50],
                         resolution = [1, 1],
                         valid_range = [0.1, 1.0]
                         )
        self._lidar = RotatingLidarPhysX(prim_path=prim_paths, name=name,
                                         rotation_frequency=0.0,
                                        #  fov=[70, 50],
                                         fov=[100, 50],
                                         resolution=[0.5, 0.5],
                                         valid_range=[0.1, 1.0])
        
        self._lidar.add_depth_data_to_frame()
        self._lidar.add_point_cloud_data_to_frame()
        self._lidar.add_semantics_data_to_frame()
        self._lidar.enable_visualization(high_lod=True,
                                         draw_points=True,
                                         draw_lines=False)
        self._lidar.initialize()