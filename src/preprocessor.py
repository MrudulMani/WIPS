"""
WiFi Indoor Positioning System (WIPS) - Data Preprocessor & Feature Engineering
Handles RSSI value transformation, missing AP imputation, scaling, and feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class WIPSPreprocessor:
    """
    Transforms raw WiFi RSSI fingerprints into normalized feature vectors
    suitable for classification and regression models.
    """
    def __init__(self, unobserved_val=100.0, min_rssi=-105.0, max_rssi=-20.0, scaling="minmax"):
        self.unobserved_val = unobserved_val
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi
        self.scaling = scaling
        self.ap_columns = []
        self.active_ap_columns = []
        self.scaler = MinMaxScaler() if scaling == "minmax" else StandardScaler()
        self.is_fitted = False

    def fit(self, df):
        """
        Learns feature parameters and active Access Points from training dataframe.
        """
        self.ap_columns = [col for col in df.columns if col.startswith("WAP")]
        
        # Transform RSSI for detection checking
        rssi_matrix = df[self.ap_columns].values.copy()
        
        # Identify APs that have at least one detection
        is_detected = (rssi_matrix != self.unobserved_val) & (rssi_matrix <= 0.0)
        detection_counts = is_detected.sum(axis=0)
        active_indices = np.where(detection_counts > 0)[0]
        
        self.active_ap_columns = [self.ap_columns[i] for i in active_indices]
        print(f"Total APs in dataset: {len(self.ap_columns)} | Active detected APs: {len(self.active_ap_columns)}")
        
        # Fit scaler on transformed active AP features
        transformed = self._transform_rssi(df[self.active_ap_columns].values)
        self.scaler.fit(transformed)
        self.is_fitted = True
        return self

    def _transform_rssi(self, rssi_array):
        """
        Mathematical Transformation:
        1. Replace unobserved values (100.0 or > 0) with min_rssi (-105.0 dBm).
        2. Clip values to [min_rssi, max_rssi].
        3. Shift values so that min_rssi maps to 0.0, and stronger signals are positive.
           Shifted = RSSI - min_rssi  -> [0, 85]
        4. Optional exponential weighting to emphasize stronger close-range APs.
        """
        x = rssi_array.copy().astype(np.float32)
        
        # Mask for unobserved APs
        unobs_mask = (x == self.unobserved_val) | (x > 0.0) | (x < self.min_rssi)
        
        # Set unobserved to min_rssi
        x[unobs_mask] = self.min_rssi
        x = np.clip(x, self.min_rssi, self.max_rssi)
        
        # Positive shift: 0 (undetected/weakest) to 85 (strongest at -20 dBm)
        shifted = x - self.min_rssi
        
        # Exponential scaling: pow(shifted / 85.0, 2.0) or linear normalized
        # Gives higher weight to high-confidence nearby access points
        intensity = shifted / (self.max_rssi - self.min_rssi)
        return intensity

    def transform(self, df):
        """
        Transforms dataframe into feature matrix X and target dictionaries.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted on training data before calling transform()!")
            
        raw_rssi = df[self.active_ap_columns].values
        transformed = self._transform_rssi(raw_rssi)
        
        if self.scaling == "minmax":
            X = self.scaler.transform(transformed)
        else:
            X = transformed
            
        targets = {}
        if "BUILDINGID" in df.columns:
            targets["building"] = df["BUILDINGID"].values.astype(int)
        if "FLOOR" in df.columns:
            targets["floor"] = df["FLOOR"].values.astype(int)
        if "X" in df.columns and "Y" in df.columns:
            targets["coords"] = df[["X", "Y"]].values.astype(np.float32)
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            targets["gps"] = df[["LATITUDE", "LONGITUDE"]].values.astype(np.float32)
        if "SPACEID" in df.columns:
            targets["space"] = df["SPACEID"].values.astype(int)
            
        return X, targets

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def load_and_preprocess_splits(data_dir="data"):
    """
    Convenience function to load and preprocess train, val, test datasets.
    """
    import os
    train_path = os.path.join(data_dir, "raw", "train_data.csv")
    val_path = os.path.join(data_dir, "raw", "val_data.csv")
    test_path = os.path.join(data_dir, "raw", "test_data.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    preprocessor = WIPSPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    return preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test)
