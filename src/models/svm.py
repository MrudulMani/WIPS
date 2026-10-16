"""
WiFi Indoor Positioning System (WIPS) - Support Vector Machine (SVM & SVR)
Core method referenced in Stanford CS229 paper (Dan Li, Le Wang, Shiqi Wu).
Uses RBF kernel SVM for building/floor classification and multi-output SVR for coordinate regression.
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor

class WIPS_SVM:
    """
    Support Vector Machine and Regression architecture for WiFi Indoor Localization.
    """
    def __init__(self, C=10.0, gamma="scale"):
        self.C = C
        self.gamma = gamma
        self.building_model = SVC(C=C, kernel="rbf", gamma=gamma)
        self.floor_model = SVC(C=C, kernel="rbf", gamma=gamma)
        self.coords_model = MultiOutputRegressor(SVR(C=C, kernel="rbf", gamma=gamma, epsilon=0.5))

    def fit(self, X, targets):
        """
        Trains the SVM/SVR models.
        """
        X = np.asarray(X)
        print("⚡ Training Support Vector Classifier for Building...")
        self.building_model.fit(X, targets["building"])
        
        print("⚡ Training Support Vector Classifier for Floor...")
        self.floor_model.fit(X, targets["floor"])
        
        print("⚡ Training Support Vector Regressor for (X, Y) Coordinates...")
        self.coords_model.fit(X, targets["coords"])
        return self

    def predict(self, X):
        """
        Predicts Building, Floor, and (X, Y) coordinates.
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
