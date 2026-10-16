"""
WiFi Indoor Positioning System (WIPS) - Publication-Quality Visualizations
Generates presentation-ready plots: CDF curves, model comparison charts,
2D floor layout with ground truth vs. predictions, and AP signal heatmaps.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt

# Custom academic styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

def plot_cdf_comparison(model_results, output_path="outputs/plots/cdf_error_comparison.png", max_meters=20.0):
    """
    Plots Cumulative Distribution Function of positioning error across all models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 6), dpi=300)
    
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77", "#e6ab02"]
    styles = ["-", "--", "-.", ":", "-", "--"]
    
    thresholds = np.linspace(0, max_meters, 150)
    
    for idx, (model_name, res) in enumerate(model_results.items()):
        errors = res["errors_2d"]
        cdf = [np.mean(errors <= t) * 100.0 for t in thresholds]
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        median_err = np.median(errors)
        plt.plot(thresholds, cdf, label=f"{model_name} (Med: {median_err:.2f}m)", color=color, linestyle=style, linewidth=2.2)

    plt.axvline(x=3.0, color="#888888", linestyle=":", alpha=0.7, label="Room-level boundary (3m)")
    plt.axvline(x=5.0, color="#aaaaaa", linestyle=":", alpha=0.7, label="5-meter threshold")
    plt.title("Cumulative Distribution Function (CDF) of Localization Error", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Positioning Distance Error (meters)", fontsize=12)
    plt.ylabel("Probability e ≤ d (%)", fontsize=12)
    plt.xlim(0, max_meters)
    plt.ylim(0, 102)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=10, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved CDF plot to: {output_path}")


def plot_model_comparison_bar(model_results, output_path="outputs/plots/model_error_comparison.png"):
    """
    Bar chart comparing Mean Error, Median Error, and Floor Accuracy.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_names = list(model_results.keys())
    mean_errors = [res["metrics"]["mean_error_2d"] for res in model_results.values()]
    median_errors = [res["metrics"]["median_error_2d"] for res in model_results.values()]
    floor_accs = [res["metrics"]["floor_accuracy"] for res in model_results.values()]
    
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    rects1 = ax1.bar(x - width/2, mean_errors, width, label="Mean Error (m)", color="#1f77b4", alpha=0.85)
    rects2 = ax1.bar(x + width/2, median_errors, width, label="Median Error (m)", color="#2ca02c", alpha=0.85)
    
    ax1.set_ylabel("Positioning Error in Meters (Lower is Better)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Secondary axis for Floor Accuracy
    ax2 = ax1.twinx()
    line = ax2.plot(x, floor_accs, color="#d62728", marker="o", linewidth=2.5, markersize=8, label="Floor Accuracy (%)")
    ax2.set_ylabel("Floor Accuracy % (Higher is Better)", fontsize=11, fontweight="bold", color="#d62728")
    ax2.set_ylim(80, 102)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    
    plt.title("Comparative Performance Across Positioning Models", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved comparison bar chart to: {output_path}")


def plot_floor_plan_predictions(y_true, y_pred, building_id=1, floor_id=1, ap_info=None, output_path="outputs/plots/floor_prediction_map.png"):
    """
    Visualizes actual vs predicted positions on a 2D floor plane.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 7), dpi=300)
    
    # Filter test points for this specific building and floor
    mask = (y_true["building"] == building_id) & (y_true["floor"] == floor_id)
    if np.sum(mask) == 0:
        mask = (y_true["building"] == building_id)
        
    actual_coords = y_true["coords"][mask][:60]  # Plot first 60 for visual clarity
    pred_coords = y_pred["coords"][mask][:60]
    
    # Plot Access Points if available
    if ap_info is not None:
        ap_x = [ap["x"] for ap in ap_info if ap["building"] == building_id and ap["floor"] == floor_id]
        ap_y = [ap["y"] for ap in ap_info if ap["building"] == building_id and ap["floor"] == floor_id]
        if len(ap_x) > 0:
            plt.scatter(ap_x, ap_y, marker="^", color="#e6550d", s=140, edgecolor="black", label=f"WiFi APs (Floor {floor_id})", zorder=5)

    # Plot error vectors (lines connecting actual to predicted)
    for i in range(len(actual_coords)):
        plt.plot([actual_coords[i, 0], pred_coords[i, 0]], 
                 [actual_coords[i, 1], pred_coords[i, 1]], 
                 color="#999999", linestyle=":", alpha=0.7, zorder=2)
                 
    # Plot actual and predicted points
    plt.scatter(actual_coords[:, 0], actual_coords[:, 1], color="#2ca02c", s=60, marker="o", label="Actual Location (Ground Truth)", zorder=3)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], color="#d62728", s=60, marker="x", label="Predicted Location (WIPS)", zorder=4)

    plt.title(f"Indoor Localization Map: Building {building_id}, Floor {floor_id}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("X Coordinate (meters)", fontsize=11)
    plt.ylabel("Y Coordinate (meters)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved floor plan plot to: {output_path}")


def plot_top_ap_importance(top_aps, output_path="outputs/plots/ap_feature_importance.png"):
    """
    Plots bar chart of the top Access Points by feature importance.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)
    
    ap_ids = [item["ap_id"] for item in top_aps][::-1]
    importances = [item["importance"] for item in top_aps][::-1]
    
    y_pos = np.arange(len(ap_ids))
    plt.barh(y_pos, importances, color="#3182bd", edgecolor="#08519c", alpha=0.85)
    plt.yticks(y_pos, ap_ids, fontsize=10)
    plt.xlabel("Relative Gini Feature Importance", fontsize=11, fontweight="bold")
    plt.title("Top Most Discriminative WiFi Access Points (Feature Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved AP importance plot to: {output_path}")
