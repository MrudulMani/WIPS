"""
WiFi Indoor Positioning System (WIPS) - Hierarchical Multi-Stage Positioning Engine
State-of-the-art hierarchical architecture:
Stage 1: Building Classification -> Stage 2: Floor Classification -> Stage 3: Floor-conditioned Coordinate Regression
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

class WIPS_Hierarchical:
    """
    Hierarchical Positioning Pipeline to eliminate cross-building/cross-floor errors.
    """
    def __init__(self, n_estimators=80, max_depth=14, n_neighbors=5):
        self.building_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.floor_models = {}      # dict mapping b_id -> floor classifier
        self.coord_regressors = {}  # dict mapping (b_id, floor_id) -> regressor
        self.fallback_regressor = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.n_neighbors = n_neighbors

    def fit(self, X, targets):
        X = np.asarray(X)
        b_targets = targets["building"]
        f_targets = targets["floor"]
        c_targets = targets["coords"]

        print("🏛️ [Stage 1] Training Global Building Classifier...")
        self.building_model.fit(X, b_targets)

        # Stage 2 & 3: Fit per building and floor
        unique_buildings = np.unique(b_targets)
        for b_id in unique_buildings:
            b_mask = (b_targets == b_id)
            X_b = X[b_mask]
            f_b = f_targets[b_mask]
            c_b = c_targets[b_mask]

            print(f"  🏢 [Stage 2] Training Floor Classifier for Building {b_id} ({len(X_b)} samples)...")
            clf_floor = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42)
            clf_floor.fit(X_b, f_b)
            self.floor_models[b_id] = clf_floor

            unique_floors = np.unique(f_b)
            for f_id in unique_floors:
                bf_mask = (f_b == f_id)
                X_bf = X_b[bf_mask]
                c_bf = c_b[bf_mask]

                if len(X_bf) >= 10:
                    reg_coord = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42)
                    reg_coord.fit(X_bf, c_bf)
                    self.coord_regressors[(b_id, f_id)] = reg_coord
                    
        print("🌐 Training fallback coordinate regressor...")
        self.fallback_regressor.fit(X, c_targets)
        return self

    def predict(self, X):
        X = np.asarray(X)
        N = X.shape[0]
        
        pred_b = self.building_model.predict(X)
        pred_f = np.zeros(N, dtype=int)
        pred_c = np.zeros((N, 2), dtype=np.float32)

        for i in range(N):
            b_pred = pred_b[i]
            
            # Predict floor using building-specific model
            if b_pred in self.floor_models:
                f_pred = self.floor_models[b_pred].predict(X[i:i+1])[0]
            else:
                f_pred = 0
            pred_f[i] = f_pred

            # Predict coordinate using (building, floor)-specific regressor
            bf_key = (b_pred, f_pred)
            if bf_key in self.coord_regressors:
                pred_c[i] = self.coord_regressors[bf_key].predict(X[i:i+1])[0]
            else:
                pred_c[i] = self.fallback_regressor.predict(X[i:i+1])[0]

        return {
            "building": pred_b,
            "floor": pred_f,
            "coords": pred_c
        }
