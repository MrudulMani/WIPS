"""
WiFi Indoor Positioning System (WIPS) - k-Nearest Neighbors (k-NN) & Weighted k-NN (WKNN)
Classic fingerprint pattern matching baselines using Euclidean and Manhattan distance metrics.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode

class WIPS_KNN:
    """
    Implements k-NN and Weighted k-NN for joint Building, Floor, and (X, Y) estimation.
    """
    def __init__(self, n_neighbors=5, metric="euclidean", weighted=True, epsilon=1e-5):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighted = weighted
        self.epsilon = epsilon
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.train_X = None
        self.train_targets = None

    def fit(self, X, targets):
        """
        Stores the training fingerprints radio map.
        targets is a dict with keys: 'building', 'floor', 'coords'
        """
        self.train_X = np.asarray(X)
        self.train_targets = targets
        self.nn_model.fit(self.train_X)
        return self

    def predict(self, X):
        """
        Predicts Building, Floor, and (X, Y) coordinates for query RSSI vectors.
        """
        X = np.asarray(X)
        distances, indices = self.nn_model.kneighbors(X)
        
        N = X.shape[0]
        pred_building = np.zeros(N, dtype=int)
        pred_floor = np.zeros(N, dtype=int)
        pred_coords = np.zeros((N, 2), dtype=np.float32)

        train_building = self.train_targets["building"]
        train_floor = self.train_targets["floor"]
        train_coords = self.train_targets["coords"]

        for i in range(N):
            neighbor_indices = indices[i]
            neighbor_dists = distances[i]

            if self.weighted:
                # Inverse distance weights: w = 1 / (d + eps)
                weights = 1.0 / (neighbor_dists + self.epsilon)
                weights /= np.sum(weights)

                # Weighted coordinate interpolation
                pred_coords[i] = np.sum(train_coords[neighbor_indices] * weights[:, None], axis=0)

                # Distance-weighted voting for Building and Floor
                b_votes = {}
                f_votes = {}
                for idx, w in zip(neighbor_indices, weights):
                    b = train_building[idx]
                    f = train_floor[idx]
                    b_votes[b] = b_votes.get(b, 0.0) + w
                    f_votes[f] = f_votes.get(f, 0.0) + w
                    
                pred_building[i] = max(b_votes, key=b_votes.get)
                pred_floor[i] = max(f_votes, key=f_votes.get)
            else:
                # Unweighted coordinate average
                pred_coords[i] = np.mean(train_coords[neighbor_indices], axis=0)
                
                # Majority vote for categorical attributes
                pred_building[i] = mode(train_building[neighbor_indices], keepdims=False)[0]
                pred_floor[i] = mode(train_floor[neighbor_indices], keepdims=False)[0]

        return {
            "building": pred_building,
            "floor": pred_floor,
            "coords": pred_coords
        }
