import torch
from torch import linalg as LA
import torch.nn.functional as F
import numpy as np
import math
from sklearn.decomposition import PCA
import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix,axis_angle_to_quaternion, quaternion_multiply, matrix_to_quaternion
from copy import deepcopy


def visualize_pcd(
                  tool_pcd_transformed, cropped_tool_pcd,
                  tool_pos, tool_rot,
                  imaginary_grasping_point,
                  real_grasping_point,
                  tool_tip_point,
                  first_principal_axes,
                  approx_tool_rot,
                  object_pcd_transformed,
                  base_pos, base_rot,
                  object_pos, object_rot,
                  flange_pos, flange_rot,
                  goal_pos,
                  min_indices,
                  base_coord='flange',  # robot_base or flange
                  view_idx=0):
    
    tool_pos_np = tool_pos[view_idx].cpu().numpy()
    tool_rot_np = tool_rot[view_idx].cpu().numpy()
    base_pos_np = base_pos[view_idx].cpu().numpy()
    base_rot_np = base_rot[view_idx].cpu().numpy()
    obj_pos_np = object_pos[view_idx].cpu().numpy()
    obj_rot_np = object_rot[view_idx].cpu().numpy()
    flange_pos_np = flange_pos[view_idx].detach().cpu().numpy()
    flange_rot_np = flange_rot[view_idx].detach().cpu().numpy()

    world_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))

    if base_coord == 'flange':
        flange_coord = world_coord
        base_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))
        # Visualize base coordination
        base_rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(base_rot_np)
        T_base = np.eye(4)
        T_base[:3, :3] = base_rot_matrix
        T_base[:3, 3] = base_pos_np
        base_coord = base_coord.transform(T_base)

    else:
        base_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.15, origin=np.array([0.0, 0.0, 0.0]))
        # Visualize flange coordination
        flange_rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(flange_rot_np)
        T_flange = np.eye(4)
        T_flange[:3, :3] = flange_rot_matrix
        T_flange[:3, 3] = flange_pos_np
        flange_coord = deepcopy(base_coord).transform(T_flange)

    # Create a square plane
    flange_rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(flange_rot_np)
    plane_normal = flange_rot_matrix[:, 0]  # x-axis of flange coordinate system
    plane_center = flange_pos_np
    # Create a square plane
    plane_size = 0.2
    plane_points = [
        plane_center + plane_size * (flange_rot_matrix[:, 1] + flange_rot_matrix[:, 2]),
        plane_center + plane_size * (flange_rot_matrix[:, 1] - flange_rot_matrix[:, 2]),
        plane_center + plane_size * (-flange_rot_matrix[:, 1] - flange_rot_matrix[:, 2]),
        plane_center + plane_size * (-flange_rot_matrix[:, 1] + flange_rot_matrix[:, 2])
    ]
    plane_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    yz_plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(plane_points),
        lines=o3d.utility.Vector2iVector(plane_lines)
    )
    yz_plane.paint_uniform_color([0.5, 0.5, 0])  # Yellow for yz-plane
    # Visualize full tool point cloud
    tool_transformed_pcd_np = tool_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
    tool_transformed_point_cloud = o3d.geometry.PointCloud()
    tool_transformed_point_cloud.points = o3d.utility.Vector3dVector(tool_transformed_pcd_np)
    T_t = np.eye(4)
    T_t[:3, :3] = tool_rot_np
    T_t[:3, 3] = tool_pos_np
    tool_coord = deepcopy(world_coord).transform(T_t)

    # Visualize grasping point
    imaginary_grasping_point_np = imaginary_grasping_point[view_idx].detach().cpu().numpy()
    imaginary_grasping_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    imaginary_grasping_sphere.paint_uniform_color([0, 1, 1])  # Cyan color for grasping point
    imaginary_grasping_sphere.translate(imaginary_grasping_point_np)

    # Visualize tool tip point
    tool_tip_point_np = tool_tip_point[view_idx].detach().cpu().numpy()
    tool_tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    tool_tip_sphere.paint_uniform_color([0, 0, 1])  # Blue color for tool tip
    tool_tip_sphere.translate(tool_tip_point_np)

    # Visualize cropped point cloud
    cropped_pcd_np = cropped_tool_pcd[view_idx].squeeze(0).detach().cpu().numpy()
    cropped_point_cloud = o3d.geometry.PointCloud()
    cropped_point_cloud.points = o3d.utility.Vector3dVector(cropped_pcd_np)

    # Visualize real grasping point
    real_grasping_point_np = real_grasping_point[view_idx].detach().cpu().numpy()
    real_grasping_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    real_grasping_sphere.paint_uniform_color([1, 0, 1])  # Magenta color for real grasping point
    real_grasping_sphere.translate(real_grasping_point_np)

    # Principal axis
    principal_axis = first_principal_axes[view_idx, :].cpu().numpy()
    mean_point = cropped_pcd_np.mean(axis=0)
    start_point = mean_point - principal_axis * 0.5
    end_point = mean_point + principal_axis * 0.5
    # Create line set for the principal axis
    pca_line = o3d.geometry.LineSet()
    pca_line.points = o3d.utility.Vector3dVector([start_point, end_point])
    pca_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    pca_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Yellow color for principal axis

    # Quaternion to rotation matrix
    approx_tool_rot_np = approx_tool_rot[view_idx].cpu().numpy()
    rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(approx_tool_rot_np)
    
    # Create arrow for the principal axis
    ori_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.1, cone_height=0.02)
    ori_arrow.paint_uniform_color([1, 0, 1])  # Magenta color for orientation arrow
    T_arrow = np.eye(4)
    T_arrow[:3, :3] = rot_matrix
    ori_arrow.transform(T_arrow)
    ori_arrow.translate(real_grasping_point_np)  # Translate the arrow to the grasping point

    obj_transformed_pcd_np = object_pcd_transformed[view_idx].squeeze(0).detach().cpu().numpy()
    obj_transformed_point_cloud = o3d.geometry.PointCloud()
    obj_transformed_point_cloud.points = o3d.utility.Vector3dVector(obj_transformed_pcd_np)
    T_o = np.eye(4)

    # R_b = tgt_rot_np.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    T_o[:3, :3] = obj_rot_np
    # T_o[:3, :3] = R_b
    T_o[:3, 3] = obj_pos_np
    obj_coord = deepcopy(world_coord).transform(T_o)

    goal_pos_np = goal_pos[view_idx].cpu().numpy()
    goal_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.01, height=0.03)
    goal_cone.paint_uniform_color([0, 1, 0])
    T_g_p = np.eye(4)
    T_g_p[:3, 3] = goal_pos_np
    goal_position = deepcopy(goal_cone).transform(T_g_p)

    #Closest point between two point clouds
    min_indices_np = min_indices[view_idx].cpu().numpy()
    closest_point_tool = tool_transformed_pcd_np[min_indices_np[0]]
    closest_point_obj = obj_transformed_pcd_np[min_indices_np[1]]
    closest_points = np.array([closest_point_tool, closest_point_obj])
    closest_line = o3d.geometry.LineSet()
    closest_line.points = o3d.utility.Vector3dVector(closest_points)
    closest_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    closest_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Red for tool, Green for object

    # goal_pos_xy_np = copy.deepcopy(goal_pos_np)
    # goal_pos_xy_np[2] = self.target_height
    # goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # goal_sphere.paint_uniform_color([1, 0, 0])
    # T_g = np.eye(4)
    # T_g[:3, 3] = goal_pos_xy_np
    # goal_position_xy = copy.deepcopy(goal_sphere).transform(T_g)

    o3d.visualization.draw_geometries([base_coord,
                                        tool_transformed_point_cloud,
                                        cropped_point_cloud,
                                        tool_coord,
                                        imaginary_grasping_sphere,
                                        real_grasping_sphere,
                                        tool_tip_sphere,
                                        pca_line,
                                        ori_arrow,
                                        obj_transformed_point_cloud,                                        
                                        obj_coord,
                                        flange_coord,
                                        goal_position,
                                        closest_line,
                                        yz_plane,
                                    # goal_position_xy
                                    ],
                                        window_name=f'point cloud')



def get_imaginary_grasping_point(flange_pos, flange_rot):
    """
    Get the imaginary grasping point using approximated offset.

    Args:
    - flange_pos: Tensor of shape [num_envs, 3] representing flange positions
    - flange_rot: Tensor of shape [num_envs, 4] representing flange orientations as quaternions

    Returns:
    - imaginary_grasping_point: Tensor of imaginary grasping points with shape [num_envs, 3]
    """
    # Convert quaternion to rotation matrix
    flange_rot_matrix = quaternion_to_matrix(flange_rot)

    # Y-direction in the flange's coordinate system
    y_direction = torch.tensor([0, 1, 0], device=flange_pos.device, dtype=flange_pos.dtype)
    y_direction_flange = torch.matmul(flange_rot_matrix, y_direction)

    # Imaginary grasping point, 0.16m away in the y-direction at Robotiq 2F-85
    imaginary_grasping_point = flange_pos + 0.16 * y_direction_flange

    return imaginary_grasping_point


def crop_tool_pcd(tool_pcd, grasping_point, radius=0.05):
    """
    Crop the tool point cloud around the imaginary grasping point using a spherical region.

    Args:
    - tool_pcd: Tensor of shape [num_envs, num_points, 3]
    - grasping_point: Tensor of shape [num_envs, 3]
    - radius: Float, radius of the sphere to crop around the grasping point

    Returns:
    - cropped_pcd: Tensor of cropped points with shape [num_envs, num_cropped_points, 3]
    """
    # Calculate the distance from each point to the grasping point
    distances = torch.linalg.norm(tool_pcd - grasping_point.unsqueeze(1), dim=2)
    mask = distances <= radius  # Create a mask to select points within the sphere
    mask_int = mask.int()   # Convert mask to integer type for sorting

    # Find the maximum number of points that meet the condition for each environment
    max_cropped_points = mask_int.sum(dim=1).max().item()

    # Use the mask to select points and maintain the shape [num_envs, num_cropped_points, 3]
    sorted_idx = torch.argsort(mask_int, descending=True, dim=1)
    sorted_idx = sorted_idx[:, :max_cropped_points]
    cropped_pcd = torch.gather(tool_pcd, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))

    return cropped_pcd


def apply_pca(cropped_pcd):
    """
    Apply PCA to the cropped point cloud to get the principal axis.

    Args:
    - cropped_pcd: Tensor of shape [num_envs, num_cropped_points, 3]

    Returns:
    - principal_axes: Tensor of shape [num_envs, 3]
    """
    num_envs, num_cropped_points, _ = cropped_pcd.shape

    # Center the data
    mean = torch.mean(cropped_pcd, dim=1, keepdim=True)
    centered_data = cropped_pcd - mean

    # # Reshape the data for batched SVD
    # centered_data_flat = centered_data.view(num_envs, num_cropped_points, 3)
    # U, S, V = torch.svd(centered_data_flat) # Perform batched SVD
    # principal_axes = V[:, :, 0] # Extract the first principal component
    # return principal_axes

    # Perform PCA
    _, _, V = torch.svd(centered_data)

    return V[:, :, 0], mean # Return the first principal axis and the mean of the cropped point cloud


def get_tool_tip_position(imaginary_grasping_point, tool_pcd):
    """
    Get tool tip position by finding the farthest point from the imaginary grasping point.

    Args:
    - imaginary_grasping_point: Tensor of shape [num_envs, 3]
    - tool_pcd: Tensor of shape [num_envs, num_points, 3]

    Returns:
    - tool_end_point: Tensor of shape [num_envs, 3]
    """
    B, N, _ = tool_pcd.shape
    # calculate farthest distance and idx from the tool to the goal
    diff = tool_pcd - imaginary_grasping_point[:, None, :]
    distance = diff.norm(dim=2)  # [B, N]

    # Find the index and value of the farthest point from the base coordinate
    farthest_idx = distance.argmax(dim=1)  # [B]
    # farthest_val = distance.gather(1, farthest_idx.unsqueeze(1)).squeeze(1)  # [B]
    tool_end_point = tool_pcd.gather(1, farthest_idx.view(B, 1, 1).expand(B, 1, 3)).squeeze(1).squeeze(1)  # [B, 3]
    
    return tool_end_point



def find_intersection_with_yz_plane(flange_pos, flange_rot, principal_axis, grasping_point):
    x_direction = torch.tensor([1, 0, 0], device=flange_pos.device, dtype=flange_pos.dtype)
    x_direction_flange = torch.matmul(quaternion_to_matrix(flange_rot), x_direction)
    t = torch.dot((flange_pos - grasping_point), x_direction_flange) / torch.dot(principal_axis, x_direction_flange)
    intersection_point = grasping_point + t * principal_axis
    return intersection_point


def calculate_tool_orientation(principal_axes, tool_tip_points, grasping_points):
    """
    Calculate the tool orientation based on the principal axes and the vector from grasping point to tool tip.

    Args:
    - principal_axes: Tensor of shape [num_envs, 3]
    - tool_tip_points: Tensor of shape [num_envs, 3]
    - grasping_points: Tensor of shape [num_envs, 3]

    Returns:
    - approx_tool_rot: Tensor of shape [num_envs, 4]
    """
    # Use the first principal axis as the primary axis
    primary_axes = principal_axes
    opposite_axes = -principal_axes

    # Vector from grasping point to tool tip
    vector_grasp_to_tip = tool_tip_points - grasping_points
    vector_grasp_to_tip = F.normalize(vector_grasp_to_tip, dim=1)

    # Calculate the dot product between the primary axis and the vector from grasping point to tool tip
    dot_product_primary = torch.sum(primary_axes * vector_grasp_to_tip, dim=1)
    dot_product_opposite = torch.sum(opposite_axes * vector_grasp_to_tip, dim=1)

    selected_axes = torch.where(dot_product_primary.unsqueeze(1) > 0, primary_axes, opposite_axes)

    # Calculate rotation from [0, 0, 1] to selected_axes
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=selected_axes.device).expand_as(selected_axes)
    
    # Compute the rotation axis
    rotation_axis = torch.cross(z_axis, selected_axes, dim=1)
    rotation_axis_norm = torch.norm(rotation_axis, dim=1, keepdim=True)
    
    # Handle the case where selected_axes is already [0, 0, 1] or [0, 0, -1]
    rotation_axis = torch.where(rotation_axis_norm > 1e-6, rotation_axis / rotation_axis_norm, torch.tensor([1.0, 0.0, 0.0], device=selected_axes.device).expand_as(selected_axes))
    
    # Compute the rotation angle
    cos_angle = torch.sum(z_axis * selected_axes, dim=1)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    # Construct the quaternion
    approx_tool_rot = torch.zeros((selected_axes.shape[0], 4), device=selected_axes.device)
    approx_tool_rot[:, 0] = torch.cos(angle / 2)
    approx_tool_rot[:, 1:] = rotation_axis * torch.sin(angle / 2).unsqueeze(1)
    return approx_tool_rot


def get_real_grasping_point(flange_pos, flange_rot, principal_axes, cropped_pcd_mean):
    """
    Calculate the real grasping point as the intersection of the yz-plane 
    in the flange coordinate system and the line along the principal axis
    that passes through the mean of the cropped point cloud.

    Args:
    - flange_pos: Tensor of shape [num_envs, 3] representing flange positions
    - flange_rot: Tensor of shape [num_envs, 4] representing flange rotations as quaternions
    - principal_axes: Tensor of shape [num_envs, 3] representing the first principal axes
    - cropped_pcd_mean: Tensor of shape [num_envs, 1, 3] representing the mean of cropped point clouds

    Returns:
    - real_grasping_point: Tensor of shape [num_envs, 3] representing the real grasping points
    """
    # Convert flange rotation quaternion to rotation matrix
    flange_rot_matrix = quaternion_to_matrix(flange_rot)

    # Get the x-axis of the flange coordinate system
    flange_x_axis = flange_rot_matrix[:, :, 0]

    # Remove the extra dimension from cropped_pcd_mean
    cropped_pcd_mean = cropped_pcd_mean.squeeze(1)

    # Calculate the parameter t
    # The plane equation is: (p - flange_pos) · flange_x_axis = 0
    # The line equation is: p = cropped_pcd_mean + t * principal_axis
    # Substituting the line equation into the plane equation:
    # ((cropped_pcd_mean + t * principal_axis) - flange_pos) · flange_x_axis = 0
    # Solving for t: t = ((flange_pos - cropped_pcd_mean) · flange_x_axis) / (principal_axis · flange_x_axis)
    
    numerator = torch.sum((flange_pos - cropped_pcd_mean) * flange_x_axis, dim=1)
    denominator = torch.sum(principal_axes * flange_x_axis, dim=1)
    
    # Handle cases where the principal axis is parallel to the yz-plane
    t = torch.where(
        torch.abs(denominator) > 1e-6,
        numerator / denominator,
        torch.zeros_like(numerator)
    )

    # Calculate the intersection point
    real_grasping_point = cropped_pcd_mean + t.unsqueeze(1) * principal_axes

    return real_grasping_point


def create_ee_transform(flange_pos, flange_rot):
    """
    Create a transformation matrix from the end-effector frame to the robot base frame.
    
    Args:
    - flange_pos: Tensor of shape [num_envs, 3] representing the end-effector positions
    - flange_rot: Tensor of shape [num_envs, 4] representing the end-effector orientations as quaternions
    
    Returns:
    - transform: Tensor of shape [num_envs, 4, 4] representing the transformation matrices
    """
    num_envs = flange_pos.shape[0]
    transform = torch.eye(4, device=flange_pos.device).unsqueeze(0).repeat(num_envs, 1, 1)

    # Set the rotation part of the transform (transpose of the original rotation)
    rotation_matrix = quaternion_to_matrix(flange_rot)
    transform[:, :3, :3] = rotation_matrix.transpose(1, 2)
    # Set the translation part of the transform
    transform[:, :3, 3] = -torch.bmm(rotation_matrix.transpose(1, 2), flange_pos.unsqueeze(2)).squeeze(2)

    # rotation_matrix = quaternion_to_matrix(flange_rot)
    # transform[:, :3, :3] = rotation_matrix
    # transform[:, :3, 3] = flange_pos

    return transform


def transform_points(points, transform):
    """
    Apply transformation to points.
    
    Args:
    - points: Tensor of shape [num_envs, num_points, 3]
    - transform: Tensor of shape [num_envs, 4, 4]
    
    Returns:
    - transformed_points: Tensor of shape [num_envs, num_points, 3]
    """
    num_envs, num_points, _ = points.shape
    # Homogeneous coordinates
    points_homogeneous = torch.cat([points, torch.ones(num_envs, num_points, 1, device=points.device)], dim=-1)
    
    # Apply transformation
    transformed_points = torch.bmm(points_homogeneous, transform.transpose(1, 2))
    
    return transformed_points[:, :, :3]


def transform_pose(position, orientation, transform):
    """
    Apply transformation to a pose (position and orientation).
    
    Args:
    - position: Tensor of shape [num_envs, 3] (x, y, z)
    - orientation: Tensor of shape [num_envs, 4] (qw, qx, qy, qz)
    - transform: Tensor of shape [num_envs, 4, 4]
    
    Returns:
    - transformed_pose: Tensor of shape [num_envs, 7]
    """
    # position = pose[:, :3]
    # orientation = pose[:, 3:]
    
    # Transform position
    transformed_position = transform_points(position.unsqueeze(1), transform).squeeze(1)
    # transformed_position = -torch.bmm(transform[:, :3, :3], position.unsqueeze(2)).squeeze(2)

    # Transform orientation
    original_rot = quaternion_to_matrix(orientation)
    transformed_rot = torch.bmm(transform[:, :3, :3], original_rot)
    transformed_orientation = matrix_to_quaternion(transformed_rot)

    return transformed_position, transformed_orientation
    
def get_base_in_flange_frame(flange_pos, flange_rot):
    """
    Calculate the base coordinate in the flange frame.
    
    Args:
    - flange_pos: Tensor of shape [num_envs, 3] representing flange positions in base frame
    - flange_rot: Tensor of shape [num_envs, 4] representing flange orientations as quaternions in base frame
    
    Returns:
    - base_pose: Tensor of shape [num_envs, 7] representing base pose (position + orientation) in flange frame
    """
    # # Position of base in flange frame
    # base_pos_in_flange = -flange_pos

    transform = create_ee_transform(flange_pos, flange_rot)
    # The base position in the flange frame is just the negative of the flange position transformed
    base_pos_in_flange = -torch.bmm(transform[:, :3, :3], flange_pos.unsqueeze(2)).squeeze(2)
    # base_pos_in_flange = self.transform_points(flange_pos.unsqueeze(1), transform).squeeze(1)


    # Orientation of base in flange frame
    flange_rot_matrix = quaternion_to_matrix(flange_rot)
    base_rot_in_flange = flange_rot_matrix.transpose(1, 2)
    base_quat_in_flange = matrix_to_quaternion(base_rot_in_flange)

    return base_pos_in_flange, base_quat_in_flange


def axis_to_quaternion(axis):
    """
    Convert an axis to a quaternion representation.
    
    Args:
    - axis: Tensor of shape [num_envs, 3] representing the principal axes
    
    Returns:
    - quaternions: Tensor of shape [num_envs, 4] representing the quaternions
    """
    # Ensure the axis is normalized
    axis = F.normalize(axis, dim=1)
    
    # Our reference vector (typically the z-axis)
    reference = torch.tensor([0.0, 0.0, 1.0], device=axis.device).expand_as(axis)
    
    # Compute the rotation axis
    rotation_axis = torch.cross(reference, axis, dim=1)
    rotation_axis_norm = torch.norm(rotation_axis, dim=1, keepdim=True)
    
    # Handle cases where the axis is parallel to the reference vector
    rotation_axis = torch.where(
        rotation_axis_norm > 1e-6,
        rotation_axis / rotation_axis_norm,
        torch.tensor([1.0, 0.0, 0.0], device=axis.device).expand_as(axis)
    )
    
    # Compute the rotation angle
    cos_angle = torch.sum(reference * axis, dim=1)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    # Construct the rotation matrix
    K = torch.zeros((axis.shape[0], 3, 3), device=axis.device)
    K[:, 0, 1] = -rotation_axis[:, 2]
    K[:, 0, 2] = rotation_axis[:, 1]
    K[:, 1, 0] = rotation_axis[:, 2]
    K[:, 1, 2] = -rotation_axis[:, 0]
    K[:, 2, 0] = -rotation_axis[:, 1]
    K[:, 2, 1] = rotation_axis[:, 0]
    
    rotation_matrix = torch.eye(3, device=axis.device).unsqueeze(0).expand(axis.shape[0], 3, 3) + \
                    torch.sin(angle).unsqueeze(-1).unsqueeze(-1) * K + \
                    (1 - torch.cos(angle).unsqueeze(-1).unsqueeze(-1)) * torch.bmm(K, K)
    
    # Convert rotation matrix to quaternion
    quaternions = matrix_to_quaternion(rotation_matrix)
    
    return quaternions


def transform_principal_axis(transformation_matrix, principal_axis):
    """
    Transform a principal axis using a transformation matrix.

    Args:
    - transformation_matrix: Tensor of shape [num_envs, 4, 4] representing the transformation matrices
    - principal_axis: Tensor of shape [num_envs, 3] representing the principal axes

    Returns:
    - transformed_axis: Tensor of shape [num_envs, 3] representing the transformed principal axes
    """

    # Ensure the principal axis is a unit vector
    principal_axis = F.normalize(principal_axis, dim=1)

    # Extract the rotation part of the transformation matrix
    rotation_matrix = transformation_matrix[:, :3, :3]

    # Apply the rotation to the principal axis
    transformed_axis = torch.bmm(rotation_matrix, principal_axis.unsqueeze(-1)).squeeze(-1)

    # Normalize the result to ensure it remains a unit vector
    transformed_axis = F.normalize(transformed_axis, dim=1)

    return transformed_axis


def closest_distance_between_sets(point_cloud_A, point_cloud_B):
    """
    Compute the closest distance between two sets of point clouds for multiple environments.
    The distance is the minimum distance between any point in set A and any point in set B.
    
    Args:
    point_cloud_A: Tensor of shape [num_envs, num_points_A, 3]
    point_cloud_B: Tensor of shape [num_envs, num_points_B, 3]
    
    Returns:
    min_distances: Tensor of shape [num_envs] containing the closest distance between sets for each environment
    min_indices: Tensor of shape [num_envs, 2] containing the indices of closest points in A and B for each environment
    """
    assert point_cloud_A.dim() == 3 and point_cloud_B.dim() == 3
    assert point_cloud_A.shape[2] == 3 and point_cloud_B.shape[2] == 3
    assert point_cloud_A.shape[0] == point_cloud_B.shape[0]
    
    num_envs = point_cloud_A.shape[0]
    
    # Calculate pairwise squared distances
    A_expanded = point_cloud_A.unsqueeze(2)  # Shape: [num_envs, num_points_A, 1, 3]
    B_expanded = point_cloud_B.unsqueeze(1)  # Shape: [num_envs, 1, num_points_B, 3]
    distances = torch.sum((A_expanded - B_expanded) ** 2, dim=3)  # Shape: [num_envs, num_points_A, num_points_B]
    
    # Find the minimum distance and corresponding indices for each environment
    min_distances, min_indices_B = torch.min(distances.view(num_envs, -1), dim=1)
    min_distances = min_distances.view(num_envs, 1)
    
    # Calculate the indices in A and B
    min_indices_A = min_indices_B // point_cloud_B.shape[1]
    min_indices_B = min_indices_B % point_cloud_B.shape[1]
    
    min_indices = torch.stack((min_indices_A, min_indices_B), dim=1)
    
    
    return torch.sqrt(min_distances), min_indices



# Helper function to calculate rotation matrix from two vectors
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))
    return rotation_matrix