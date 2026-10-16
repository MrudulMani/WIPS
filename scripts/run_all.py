"""
WiFi Indoor Positioning System (WIPS) - Full Training, Benchmark & Evaluation Pipeline
Runs end-to-end data preprocessing, model training across all algorithms,
computes comparative metrics, and produces presentation graphics.
"""

import os
import sys
import json
import time
import pickle
import numpy as np

# Ensure root directory is on PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_generator import WIPSDataGenerator
from src.preprocessor import load_and_preprocess_splits
from src.evaluate import compute_metrics, print_evaluation_summary
from src.models.knn import WIPS_KNN
from src.models.random_forest import WIPS_RandomForest
from src.models.svm import WIPS_SVM
from src.models.dnn_pytorch import WIPS_PyTorch_DNN
from src.models.hierarchical import WIPS_Hierarchical
from src.visualizer import (
    plot_cdf_comparison,
    plot_model_comparison_bar,
    plot_floor_plan_predictions,
    plot_top_ap_importance
)

def run_pipeline():
    print("=" * 70)
    print("📡 WIPS: WiFi Indoor Positioning System - Stanford CS229 Reproduction & Benchmark")
    print("=" * 70)
    
    data_dir = os.path.join(BASE_DIR, "data")
    train_csv = os.path.join(data_dir, "raw", "train_data.csv")
    
    # Step 1: Ensure dataset exists
    if not os.path.exists(train_csv):
        print("Data not found. Generating dataset using ITU-R indoor propagation physics...")
        generator = WIPSDataGenerator()
        generator.save_full_dataset(output_dir=data_dir)
    else:
        print(f"Loading existing datasets from {data_dir}...")
        
    # Step 2: Load and preprocess data
    print("\n--- [Step 1/4] Preprocessing WiFi RSSI Fingerprints ---")
    preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_splits(data_dir)
    print(f"Features: {X_train.shape[1]} Access Points | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Load AP metadata for map visualization
    meta_path = os.path.join(data_dir, "building_metadata.json")
    ap_info = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            ap_info = json.load(f).get("access_points_info", [])

    # Step 3: Train and evaluate each model
    print("\n--- [Step 2/4] Training and Benchmarking ML & Deep Learning Models ---")
    
    models = {
        "k-NN (Euclidean)": WIPS_KNN(n_neighbors=5, metric="euclidean", weighted=False),
        "WKNN (Inverse Dist)": WIPS_KNN(n_neighbors=5, metric="euclidean", weighted=True),
        "Random Forest": WIPS_RandomForest(n_estimators=100, max_depth=16, random_state=42),
        "Support Vector Machine": WIPS_SVM(C=10.0),
        "PyTorch Multi-Task DNN": WIPS_PyTorch_DNN(epochs=35, batch_size=64, lr=0.002),
        "Hierarchical Multi-Stage": WIPS_Hierarchical(n_estimators=80, max_depth=14)
    }
    
    results = {}
    fitted_models = {}
    top_aps = []

    for name, model in models.items():
        print(f"\n▶ Training Model: {name} ...")
        t0 = time.time()
        
        if name == "PyTorch Multi-Task DNN":
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
            
        train_time = round(time.time() - t0, 2)
        fitted_models[name] = model
        
        # Inference on Test Set
        t_infer0 = time.time()
        preds = model.predict(X_test)
        infer_time = round((time.time() - t_infer0) * 1000 / len(X_test), 2) # ms per sample
        
        metrics, errors_2d, errors_3d = compute_metrics(y_test, preds)
        metrics["train_time_sec"] = train_time
        metrics["infer_time_ms_per_sample"] = infer_time
        
        print_evaluation_summary(name, metrics)
        
        results[name] = {
            "metrics": metrics,
            "errors_2d": errors_2d,
            "errors_3d": errors_3d,
            "sample_predictions": preds
        }
        
        # If Random Forest, extract top APs
        if name == "Random Forest":
            top_aps = model.get_top_access_points(preprocessor.active_ap_columns, top_k=15)

    # Step 4: Save metrics and models
    print("\n--- [Step 3/4] Saving Artifacts and Models ---")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    models_dir = os.path.join(outputs_dir, "models")
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save metrics JSON (converting numpy types)
    clean_summary = {}
    for m_name, res in results.items():
        clean_summary[m_name] = res["metrics"]
        
    metrics_path = os.path.join(outputs_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"Metrics written to: {metrics_path}")
    
    # Save preprocessor and best models (WKNN, Hierarchical)
    with open(os.path.join(models_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(models_dir, "wknn_model.pkl"), "wb") as f:
        pickle.dump(fitted_models["WKNN (Inverse Dist)"], f)
    with open(os.path.join(models_dir, "hierarchical_model.pkl"), "wb") as f:
        pickle.dump(fitted_models["Hierarchical Multi-Stage"], f)
    print("Model checkpoints saved in outputs/models/.")

    # Step 5: Visualizations
    print("\n--- [Step 4/4] Generating Publication-Quality Figures ---")
    plot_cdf_comparison(results, output_path=os.path.join(plots_dir, "cdf_error_comparison.png"))
    plot_model_comparison_bar(results, output_path=os.path.join(plots_dir, "model_error_comparison.png"))
    
    best_preds = results["Hierarchical Multi-Stage"]["sample_predictions"]
    plot_floor_plan_predictions(y_test, best_preds, building_id=1, floor_id=1, ap_info=ap_info,
                                output_path=os.path.join(plots_dir, "floor_prediction_map.png"))
                                
    if len(top_aps) > 0:
        plot_top_ap_importance(top_aps, output_path=os.path.join(plots_dir, "ap_feature_importance.png"))
        
    print("\n" + "=" * 70)
    print("🎉 WIPS Pipeline Execution Completed Successfully!")
    print(f"📁 Plots stored in: {plots_dir}")
    print(f"📁 Metrics saved in: {metrics_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
