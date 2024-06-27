import torch
import numpy as np
import open3d as o3d
import os

from omni.isaac.core.objects import DynamicCylinder, DynamicCone, DynamicSphere, DynamicCuboid
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omniisaacgymenvs.robots.articulations.ur5e_tool.ur5e_tool import UR5eTool

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_pcd(self,
            ply_path,
            num_envs,
            pcd_sampling_num=500,
            device='cuda',
            tools=True):
    ply_path = os.path.abspath(ply_path)
    o3d_pcd = o3d.io.read_point_cloud(ply_path)
    if tools:
            pass
    else:
        # rotate with x-axis 90 degree
        R = o3d_pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        o3d_pcd.rotate(R, center=(0, 0, 0))
    o3d_downsampled_pcd = o3d_pcd.farthest_point_down_sample(pcd_sampling_num)
    np_downsampled_points = np.array(o3d_downsampled_pcd.points)
    tool_pcd = torch.from_numpy(np_downsampled_points).to(device)
    tool_pcd = tool_pcd.unsqueeze(0).repeat(num_envs, 1, 1)
    return tool_pcd.float()

''' point cloud registration for tool '''
# TODO: 아래 기능 함수화
# region point cloud registration for tool
# get transformation matrix from base to tool
T_base_to_tool = torch.eye(4, device=self._device).unsqueeze(0).repeat(self._num_envs, 1, 1)
T_base_to_tool[:, :3, :3] = tool_rot.clone().detach()
T_base_to_tool[:, :3, 3] = tool_pos.clone().detach()

B, N, _ = self.tool_pcd.shape
# Convert points to homogeneous coordinates by adding a dimension with ones
homogeneous_points = torch.cat([self.tool_pcd, torch.ones(B, N, 1, device=self.tool_pcd.device)], dim=-1)
# Perform batch matrix multiplication
transformed_points_homogeneous = torch.bmm(homogeneous_points, T_base_to_tool.transpose(1, 2))
# Convert back from homogeneous coordinates by removing the last dimension
tool_pcd_transformed = transformed_points_homogeneous[..., :3]
# endregion
''' point cloud registration for tool '''