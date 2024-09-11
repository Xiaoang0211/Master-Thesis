import numpy as np
import torch
from torch_cluster import radius
# import open3d.ml.torch as ml3d
from matplotlib import pyplot as plt
from typing import Optional


def geodesic_distance(R1, R2, t1, t2):
    R = np.dot(R1, R2) # R2 is supposed to be the inverse matrix of R1
    cos_theta = (np.trace(R)-1)/2
    cos_theta = np.clip(cos_theta, -1, 1) # to avoid numerical errors, which will lead arccos result in nan
    angular_dist = np.arccos(cos_theta) * (180/np.pi)

    euclidean_dist = np.linalg.norm(t1 - t2)
    # print("angular distance (°): ", angular_dist)
    # print("euclidean distance (m): ", euclidean_dist)
    return euclidean_dist, angular_dist

# def compute_pcl_overlap(source, target, threshold=1e-7):
#     '''
#     Compute overlap ratio from source point cloud to target point cloud
#     '''
#     points = torch.from_numpy(np.asarray(source)).to(torch.float64)
#     queries = torch.from_numpy(np.asarray(target)).to(torch.float64)
#     radii = torch.tensor([threshold]).tile(queries.size(0)).to(torch.float64)
#     nsearch = ml3d.layers.RadiusSearch(return_distances=False)
#     ans = nsearch(points, queries, radii)
#     common_pts_idx_src = np.unique(ans[0].numpy())
    
#     overlap_ratio = round(common_pts_idx_src.shape[0] / source.shape[0], 4)
#     return overlap_ratio, common_pts_idx_src

def compute_pcl_overlap(source, target, threshold=1e-7):
    '''
    Compute overlap ratio from source point cloud to target point cloud
    '''
    source_points = torch.from_numpy(np.asarray(source)).to(torch.float64)
    target_queries = torch.from_numpy(np.asarray(target)).to(torch.float64)
    
    # Perform radius search
    edge_indices = radius(source_points, target_queries, r=threshold)

    # Get unique source indices from edge indices
    common_pts_idx_src = torch.unique(edge_indices[0]).numpy()
    
    overlap_ratio = round(len(common_pts_idx_src) / source_points.shape[0], 4)
    return overlap_ratio, common_pts_idx_src

def visualize_overlapped_plc(query_plc, top1_plc, threshold=0.05):
    _, common_pts_idx_src = compute_pcl_overlap(query_plc, top1_plc, threshold)
                
    overlapped_points = query_plc[common_pts_idx_src]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(query_plc[:, 0], query_plc[:, 1], query_plc[:, 2], s=2, c='g', label='Source Point Cloud', alpha=0.2)
    ax.scatter(top1_plc[:, 0], top1_plc[:, 1], top1_plc[:, 2], s=2, c='b', label='Reference Point Cloud', alpha=0.2)
    ax.scatter(overlapped_points[:, 0], overlapped_points[:, 1], overlapped_points[:, 2], s=5, c='r', label='overlapped_points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Visualization of the Two Point Clouds")
    plt.show()
    
def deg_to_rad(deg):
    """ Convert degrees to radians. """
    return deg * (np.pi / 180)
    
def gen_random_rot_deg(roll_range=(-45, 45), 
                       pitch_range=(-45, 45), 
                       yaw_range=(-180, 180)):
    """ Generate random rotation matrix for 3D points, input ranges in degrees. """
    # Convert degrees to radians
    roll_range = tuple(map(deg_to_rad, roll_range))
    pitch_range = tuple(map(deg_to_rad, pitch_range))
    yaw_range = tuple(map(deg_to_rad, yaw_range))

    alpha = np.random.uniform(roll_range[0], roll_range[1])  # roll
    beta = np.random.uniform(pitch_range[0], pitch_range[1])  # pitch
    gamma = np.random.uniform(yaw_range[0], yaw_range[1])    # yaw

    # Rotation matrices for each angle
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    return R

def gen_random_trans(x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), z_range=(-0.1, 0.1)):
    """ Generate random transformation for 3D points

    Args:
        trans_range (_type_): _description_
        rot_range (_type_): _description_
    """
    t_x = np.random.uniform(x_range[0], x_range[1])
    t_y = np.random.uniform(y_range[0], y_range[1])
    t_z = np.random.uniform(z_range[0], z_range[1])
        
    # Construct the translation matrix
    translation_matrix = np.array([
            [t_x],
            [t_y],
            [t_z],
        ])
    return translation_matrix

def apply_transformation(points, rotation_matrix, translation_vector):
    if points.shape[1] == 512:
        transformed_points = []
        for node in points:
            # Apply rotation and translation to each node
            transformed_node = np.matmul(node, rotation_matrix.T) + translation_vector.T
            transformed_points.append(transformed_node)
        return np.array(transformed_points)
    else:
        transformed_points = np.matmul(points, rotation_matrix.T) + translation_vector.T
        return transformed_points
