"""
WiFi Indoor Positioning System (WIPS) - Evaluation Suite & Metrics
Calculates Building/Floor Accuracy, 2D/3D Positioning Errors, Percentiles, and Error CDFs.
"""

import numpy as np

def compute_metrics(y_true_dict, y_pred_dict, floor_height_penalty=4.0):
    """
    Computes standard indoor positioning metrics.
    
    Parameters:
    - y_true_dict: Dict with keys 'building', 'floor', 'coords' (N x 2)
    - y_pred_dict: Dict with keys 'building', 'floor', 'coords' (N x 2)
    - floor_height_penalty: meters penalty per mismatched floor (default 4.0m)
    
    Returns:
    - dict of metrics
    """
    metrics = {}
    N = len(y_true_dict["coords"])
    
    # 1. Building accuracy
    if "building" in y_true_dict and "building" in y_pred_dict:
        b_true = np.asarray(y_true_dict["building"])
        b_pred = np.asarray(y_pred_dict["building"])
        b_acc = np.mean(b_true == b_pred) * 100.0
        metrics["building_accuracy"] = round(float(b_acc), 2)
        
    # 2. Floor accuracy
    if "floor" in y_true_dict and "floor" in y_pred_dict:
        f_true = np.asarray(y_true_dict["floor"])
        f_pred = np.asarray(y_pred_dict["floor"])
        f_acc = np.mean(f_true == f_pred) * 100.0
        metrics["floor_accuracy"] = round(float(f_acc), 2)
        
    # 3. 3D Joint (Building + Floor) Hit Rate
    if "building" in metrics and "floor" in metrics:
        joint_acc = np.mean((b_true == b_pred) & (f_true == f_pred)) * 100.0
        metrics["joint_3d_accuracy"] = round(float(joint_acc), 2)
        
    # 4. 2D Euclidean Distance Error (meters)
    coords_true = np.asarray(y_true_dict["coords"])
    coords_pred = np.asarray(y_pred_dict["coords"])
    
    dist_errors = np.sqrt(np.sum((coords_pred - coords_true) ** 2, axis=1))
    
    metrics["mean_error_2d"] = round(float(np.mean(dist_errors)), 2)
    metrics["median_error_2d"] = round(float(np.median(dist_errors)), 2)
    metrics["std_error_2d"] = round(float(np.std(dist_errors)), 2)
    metrics["rmse_error_2d"] = round(float(np.sqrt(np.mean(dist_errors ** 2))), 2)
    metrics["p75_error_2d"] = round(float(np.percentile(dist_errors, 75)), 2)
    metrics["p90_error_2d"] = round(float(np.percentile(dist_errors, 90)), 2)
    metrics["p95_error_2d"] = round(float(np.percentile(dist_errors, 95)), 2)
    metrics["max_error_2d"] = round(float(np.max(dist_errors)), 2)
    
    # 5. Penalized 3D Error (accounting for vertical floor mistakes)
    if "floor" in y_true_dict and "floor" in y_pred_dict:
        f_diff = np.abs(f_true - f_pred) * floor_height_penalty
        dist_errors_3d = np.sqrt(dist_errors ** 2 + f_diff ** 2)
        metrics["mean_error_3d"] = round(float(np.mean(dist_errors_3d)), 2)
        metrics["median_error_3d"] = round(float(np.median(dist_errors_3d)), 2)
    else:
        dist_errors_3d = dist_errors
        
    return metrics, dist_errors, dist_errors_3d


def compute_cdf(errors, max_threshold=25.0, steps=100):
    """
    Computes Cumulative Distribution Function (CDF) of localization error.
    Returns:
    - thresholds: array of meter distances [0, max_threshold]
    - cdf_values: percentage of predictions with error <= threshold [0, 100%]
    """
    thresholds = np.linspace(0, max_threshold, steps)
    cdf_values = np.array([np.mean(errors <= t) * 100.0 for t in thresholds])
    return thresholds, cdf_values


def print_evaluation_summary(model_name, metrics):
    """
    Prints a formatted evaluation table.
    """
    print("=" * 60)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("=" * 60)
    if "building_accuracy" in metrics:
        print(f"  🏢 Building Classification Accuracy : {metrics['building_accuracy']:.2f}%")
    if "floor_accuracy" in metrics:
        print(f"  🏢 Floor Classification Accuracy    : {metrics['floor_accuracy']:.2f}%")
    if "joint_3d_accuracy" in metrics:
        print(f"  🎯 3D Multi-floor Hit Rate (Bld+Flr): {metrics['joint_3d_accuracy']:.2f}%")
    print(f"  📏 Mean Positioning Error (2D)      : {metrics['mean_error_2d']:.2f} meters")
    print(f"  📏 Median Positioning Error (50%)   : {metrics['median_error_2d']:.2f} meters")
    print(f"  📏 75th Percentile Error            : {metrics['p75_error_2d']:.2f} meters")
    print(f"  📏 95th Percentile Error            : {metrics['p95_error_2d']:.2f} meters")
    if "mean_error_3d" in metrics:
        print(f"  📐 Floor-Penalized 3D Mean Error    : {metrics['mean_error_3d']:.2f} meters")
    print("=" * 60)
