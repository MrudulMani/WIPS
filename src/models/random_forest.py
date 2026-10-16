"""
WiFi Indoor Positioning System (WIPS) - Random Forest Classifier & Regressor Suite
Uses ensemble tree models to predict Building, Floor, and (X, Y) coordinates,
and extracts Access Point importance rankings.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class WIPS_RandomForest:
    """
    Random Forest ensemble architecture for WiFi Indoor Positioning.
    """
    def __init__(self, n_estimators=100, max_depth=16, random_state=42, n_jobs=-1):
        self.building_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
        )
        self.floor_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
        )
        self.coords_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
        )
        self.feature_importances_ = None

    def fit(self, X, targets):
        """
        Trains the ensemble trees on WiFi fingerprints.
        """
        X = np.asarray(X)
        print("🌲 Training Random Forest Building Classifier...")
        self.building_model.fit(X, targets["building"])
        
        print("🌲 Training Random Forest Floor Classifier...")
        self.floor_model.fit(X, targets["floor"])
        
        print("🌲 Training Random Forest Coordinate Regressor (X, Y)...")
        self.coords_model.fit(X, targets["coords"])
        
        # Aggregate feature importance across the models
        imp_b = self.building_model.feature_importances_
        imp_f = self.floor_model.feature_importances_
        imp_c = np.mean([tree.feature_importances_ for tree in self.coords_model.estimators_], axis=0) if hasattr(self.coords_model, "estimators_") else self.coords_model.feature_importances_
        
        self.feature_importances_ = (imp_b + imp_f + imp_c) / 3.0
        return self

    def predict(self, X):
        """
        Predicts Building, Floor, and continuous (X, Y) coordinates.
        """
        X = np.asarray(X)
        pred_b = self.building_model.predict(X)
        pred_f = self.floor_model.predict(X)
        pred_c = self.coords_model.predict(X)
        
        return {
            "building": pred_b,
            "floor": pred_f,
            "coords": pred_c
        }

    def get_top_access_points(self, ap_names, top_k=15):
        """
        Returns the top most informative Access Points based on tree impurity reduction.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first!")
            
        indices = np.argsort(self.feature_importances_)[::-1][:top_k]
        top_aps = []
        for rank, idx in enumerate(indices):
            top_aps.append({
                "rank": rank + 1,
                "ap_id": ap_names[idx] if idx < len(ap_names) else f"AP_{idx}",
                "importance": float(self.feature_importances_[idx])
            })
        return top_aps
