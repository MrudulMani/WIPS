"""
Automated Test Suite for WIPS (WiFi Indoor Positioning System)
Verifies data generation, preprocessing, model training, and metric evaluation.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from src.data_generator import IndoorEnvironment, WIPSDataGenerator
from src.preprocessor import WIPSPreprocessor
from src.evaluate import compute_metrics, compute_cdf
from src.models.knn import WIPS_KNN
from src.models.random_forest import WIPS_RandomForest
from src.models.dnn_pytorch import MultiTaskPositioningNet, WIPS_PyTorch_DNN
from src.models.hierarchical import WIPS_Hierarchical

class TestWIPSPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.env = IndoorEnvironment(seed=42)
        self.generator = WIPSDataGenerator(environment=self.env, seed=42)

    def test_environment_topology(self):
        self.assertEqual(len(self.env.buildings), 3)
        self.assertGreater(len(self.env.aps), 50)
        for ap in self.env.aps:
            self.assertTrue(ap["ap_id"].startswith("WAP"))
            self.assertIn(ap["building"], [0, 1, 2])

    def test_data_generator_output(self):
        df = self.generator.generate_dataset(num_samples=30, split="test", seed=42)
        self.assertEqual(len(df), 30)
        self.assertIn("BUILDINGID", df.columns)
        self.assertIn("FLOOR", df.columns)
        self.assertIn("X", df.columns)
        self.assertIn("Y", df.columns)
        
        # Check RSSI bounds: either unobserved (100) or valid dBm in [-98, -20]
        ap_cols = [c for c in df.columns if c.startswith("WAP")]
        sub_rssi = df[ap_cols].values
        valid_mask = (sub_rssi == 100.0) | ((sub_rssi >= -98.0) & (sub_rssi <= -20.0))
        self.assertTrue(np.all(valid_mask))

    def test_preprocessor_pipeline(self):
        df = self.generator.generate_dataset(num_samples=60, split="train", seed=42)
        preprocessor = WIPSPreprocessor()
        X, targets = preprocessor.fit_transform(df)
        
        self.assertEqual(X.shape[0], 60)
        self.assertFalse(np.isnan(X).any())
        self.assertIn("building", targets)
        self.assertIn("floor", targets)
        self.assertIn("coords", targets)
        self.assertEqual(targets["coords"].shape, (60, 2))

    def test_knn_model(self):
        df = self.generator.generate_dataset(num_samples=40, split="train", seed=42)
        preprocessor = WIPSPreprocessor()
        X, targets = preprocessor.fit_transform(df)
        
        knn = WIPS_KNN(n_neighbors=3, weighted=True)
        knn.fit(X, targets)
        preds = knn.predict(X[:5])
        
        self.assertEqual(len(preds["building"]), 5)
        self.assertEqual(len(preds["floor"]), 5)
        self.assertEqual(preds["coords"].shape, (5, 2))

    def test_random_forest_and_importance(self):
        df = self.generator.generate_dataset(num_samples=40, split="train", seed=42)
        preprocessor = WIPSPreprocessor()
        X, targets = preprocessor.fit_transform(df)
        
        rf = WIPS_RandomForest(n_estimators=10, max_depth=6, random_state=42)
        rf.fit(X, targets)
        preds = rf.predict(X[:5])
        
        self.assertEqual(preds["coords"].shape, (5, 2))
        top_aps = rf.get_top_access_points(preprocessor.active_ap_columns, top_k=5)
        self.assertEqual(len(top_aps), 5)
        self.assertGreater(top_aps[0]["importance"], 0.0)

    def test_pytorch_dnn_architecture(self):
        net = MultiTaskPositioningNet(input_dim=50, num_buildings=3, num_floors=4)
        dummy_input = torch.randn(4, 50)
        out_b, out_f, out_c = net(dummy_input)
        
        self.assertEqual(out_b.shape, (4, 3))
        self.assertEqual(out_f.shape, (4, 4))
        self.assertEqual(out_c.shape, (4, 2))

    def test_metrics_and_cdf(self):
        y_true = {
            "building": np.array([0, 1, 2, 0]),
            "floor": np.array([1, 2, 0, 1]),
            "coords": np.array([[10.0, 20.0], [30.0, 40.0], [5.0, 5.0], [12.0, 18.0]])
        }
        y_pred = {
            "building": np.array([0, 1, 2, 0]),
            "floor": np.array([1, 2, 0, 1]),
            "coords": np.array([[10.0, 21.0], [30.0, 40.0], [5.0, 9.0], [12.0, 18.0]])
        }
        
        metrics, errors, _ = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["building_accuracy"], 100.0)
        self.assertEqual(metrics["floor_accuracy"], 100.0)
        self.assertAlmostEqual(metrics["mean_error_2d"], (1.0 + 0.0 + 4.0 + 0.0) / 4.0, places=2)
        
        thresholds, cdf_vals = compute_cdf(errors, max_threshold=10.0, steps=10)
        self.assertEqual(len(thresholds), 10)
        self.assertEqual(len(cdf_vals), 10)
        self.assertGreaterEqual(cdf_vals[-1], cdf_vals[0])

if __name__ == "__main__":
    unittest.main()
