import os
import open3d as o3d
import open3d.core as o3c
import numpy as np

print("Testing IO for point cloud ...")
pcd = o3d.io.read_point_cloud("/home/bak/Documents/tool/pcd_tool.ply")
# pcd = o3d.io.read_point_cloud("/home/bak/Documents/035_power_drill/pcd_power_drill.ply")

org_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0, 0, 0]))
o3d.visualization.draw_geometries([pcd, org_coord], window_name=f'point cloud')




o3d.visualization.draw_geometries([pcd], window_name=f'point cloud')