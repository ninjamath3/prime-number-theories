import numpy as np
import matplotlib.pyplot as plt

pts_robot=np.array([
    [160,301],
    [1550,130],
    [1550,1300],
    [160,1300]
    ])

pts_camera=np.array([
    [687.2,1790.8],
    [2465.9,1783.4],
    [2459.5,284.7],
    [677.3,291.0]
    ])

erreur=np.array([
    [0,0],
    [0,0],
    [0,0],
    [0,0]
    ])
    
def extract_geometric_features(points):
    """
    Extract order-invariant and normalized geometric features from a set of 4 2D points.
    Args:
        points (np.ndarray): Shape (4, 2) array of points (u, v)
    Returns:
        dict: Dictionary of normalized geometric features
    """
    from itertools import combinations

    # Compute all pairwise distances
    dists = [np.linalg.norm(points[i] - points[j]) for i, j in combinations(range(4), 2)]
    dists = np.array(dists)
    dists_sorted = np.sort(dists)

    # Normalize distances by the max distance (scale-invariance)
    norm_dists = dists_sorted / np.max(dists_sorted)

    # Compute barycenter
    center = np.mean(points, axis=0)

    # Distances to barycenter
    dist_to_center = np.linalg.norm(points - center, axis=1)
    dist_to_center_sorted = np.sort(dist_to_center)
    norm_dist_to_center = dist_to_center_sorted / np.max(dist_to_center_sorted)

    # Compute area using Shoelace formula (works even if points are unordered)
    def shoelace_area(pts):
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i, 0] * pts[j, 1]
            area -= pts[j, 0] * pts[i, 1]
        return 0.5 * abs(area)
    
    area = shoelace_area(points)
    # Normalize area by max bounding box area
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    bbox_area = np.prod(max_xy - min_xy)
    norm_area = area / bbox_area if bbox_area > 0 else 0.0

    # Compute simple moments (2nd order central moments)
    x = points[:, 0] - center[0]
    y = points[:, 1] - center[1]
    mu_20 = np.mean(x ** 2)
    mu_02 = np.mean(y ** 2)
    mu_11 = np.mean(x * y)
    # Normalize moments by (max_dist)^2
    max_dist_sq = np.max(dists_sorted) ** 2
    mu_20 /= max_dist_sq
    mu_02 /= max_dist_sq
    mu_11 /= max_dist_sq

    features = {
        "norm_dists": norm_dists.tolist(),
        "norm_dist_to_center": norm_dist_to_center.tolist(),
        "norm_area": norm_area,
        "mu_20": mu_20,
        "mu_02": mu_02,
        "mu_11": mu_11
    }
    return features

def extract_features_for_robot_and_camera(pts_robot, pts_camera):
    """
    Extract geometric features for both robot and camera points and return them as a single feature vector.
    Args:
        pts_robot (np.ndarray): Shape (4, 2) array of robot points (x, y)
        pts_camera (np.ndarray): Shape (4, 2) array of camera points (u, v)
    Returns:
        dict: Dictionary of geometric features for robot and camera
    """
    # Extract features for the camera (vision)
    camera_features = extract_geometric_features(pts_camera)
    
    # Extract features for the robot
    robot_features = extract_geometric_features(pts_robot)
    
    # Combine the two sets of features
    combined_features = {
        "camera_norm_dists": camera_features["norm_dists"],
        "camera_norm_dist_to_center": camera_features["norm_dist_to_center"],
        "camera_norm_area": camera_features["norm_area"],
        "camera_mu_20": camera_features["mu_20"],
        "camera_mu_02": camera_features["mu_02"],
        "camera_mu_11": camera_features["mu_11"],
        "robot_norm_dists": robot_features["norm_dists"],
        "robot_norm_dist_to_center": robot_features["norm_dist_to_center"],
        "robot_norm_area": robot_features["norm_area"],
        "robot_mu_20": robot_features["mu_20"],
        "robot_mu_02": robot_features["mu_02"],
        "robot_mu_11": robot_features["mu_11"],
    }
    return combined_features

import numpy as np

def compare_features(pts_robot, pts_camera):
    """
    Compare geometric features between the camera (vision) and robot (reference).
    Returns the absolute differences and ratios for key features.
    """
    features = extract_features_for_robot_and_camera(pts_robot, pts_camera)

    def diff_ratio(cam_val, rob_val):
        if isinstance(cam_val, list):
            cam_val = np.array(cam_val)
            rob_val = np.array(rob_val)
            return {
                "abs_diff": np.abs(cam_val - rob_val).tolist(),
                "rel_ratio": (cam_val / (rob_val + 1e-6)).tolist()
            }
        else:
            return {
                "abs_diff": abs(cam_val - rob_val),
                "rel_ratio": cam_val / (rob_val + 1e-6)
            }

    comparison = {
        "norm_dists": diff_ratio(features["camera_norm_dists"], features["robot_norm_dists"]),
        "norm_dist_to_center": diff_ratio(features["camera_norm_dist_to_center"], features["robot_norm_dist_to_center"]),
        "norm_area": diff_ratio(features["camera_norm_area"], features["robot_norm_area"]),
        "mu_20": diff_ratio(features["camera_mu_20"], features["robot_mu_20"]),
        "mu_02": diff_ratio(features["camera_mu_02"], features["robot_mu_02"]),
        "mu_11": diff_ratio(features["camera_mu_11"], features["robot_mu_11"]),
    }
    return comparison

comparison_results = compare_features(pts_robot, pts_camera+erreur)
for key in comparison_results.keys():
    print(key,comparison_results[key])
    
