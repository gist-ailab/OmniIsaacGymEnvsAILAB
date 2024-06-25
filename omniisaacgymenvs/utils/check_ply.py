import os
import open3d as o3d
import open3d.core as o3c
import numpy as np

print("Testing IO for point cloud ...")
name="ladle"
pcd = o3d.io.read_point_cloud(f"/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/ur5e_tool/usd/tool/{name}/{name}.ply")
# pcd = o3d.io.read_point_cloud("/home/bak/Documents/tool/pcd_tool.ply")

org_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0, 0, 0]))
o3d.visualization.draw_geometries([pcd, org_coord], window_name=f'point cloud')




o3d.visualization.draw_geometries([pcd], window_name=f'point cloud')