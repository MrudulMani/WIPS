"""
WiFi Indoor Positioning System (WIPS) - Multi-Task Deep Neural Network (PyTorch)
Shared representation learning with multi-task prediction heads for Building, Floor, and Coordinates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MultiTaskPositioningNet(nn.Module):
    """
    PyTorch Deep Neural Network architecture with shared backbone and 3 specialized output heads.
    """
    def __init__(self, input_dim, num_buildings=3, num_floors=4):
        super().__init__()
        
        # Shared Representation Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Output Head 1: Building Classification (Categorical Cross Entropy)
        self.building_head = nn.Linear(64, num_buildings)
        
        # Output Head 2: Floor Classification (Categorical Cross Entropy)
        self.floor_head = nn.Linear(64, num_floors)
        
        # Output Head 3: Coordinate Regression (Continuous MSE for X, Y)
        self.coord_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        shared_feat = self.backbone(x)
        out_b = self.building_head(shared_feat)
        out_f = self.floor_head(shared_feat)
        out_c = self.coord_head(shared_feat)
        return out_b, out_f, out_c


class WIPS_PyTorch_DNN:
    """
    Wrapper for training, evaluation, and inference of the Multi-Task DNN.
    """
    def __init__(self, epochs=40, batch_size=64, lr=0.002, weight_decay=1e-4, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, targets, X_val=None, targets_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        input_dim = X.shape[1]
        num_buildings = int(np.max(targets["building"])) + 1
        num_floors = int(np.max(targets["floor"])) + 1
        
        self.model = MultiTaskPositioningNet(input_dim, num_buildings, num_floors).to(self.device)
        
        # Prepare DataLoaders
        t_X = torch.tensor(X, dtype=torch.float32)
        t_b = torch.tensor(targets["building"], dtype=torch.long)
        t_f = torch.tensor(targets["floor"], dtype=torch.long)
        t_c = torch.tensor(targets["coords"], dtype=torch.float32)
        
        dataset = TensorDataset(t_X, t_b, t_f, t_c)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss criteria
        criterion_b = nn.CrossEntropyLoss()
        criterion_f = nn.CrossEntropyLoss()
        criterion_c = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        
        print(f"🧠 Training Multi-Task PyTorch DNN on {self.device} ({self.epochs} epochs)...")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            for bx, bb, bf, bc in loader:
                bx, bb, bf, bc = bx.to(self.device), bb.to(self.device), bf.to(self.device), bc.to(self.device)
                
                optimizer.zero_grad()
                pred_b, pred_f, pred_c = self.model(bx)
                
                loss_b = criterion_b(pred_b, bb)
                loss_f = criterion_f(pred_f, bf)
                loss_c = criterion_c(pred_c, bc)
                
                # Multi-task composite loss with task balancing weights
                loss = loss_c + 15.0 * loss_f + 25.0 * loss_b
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0 or epoch == self.epochs:
                print(f"  Epoch [{epoch:02d}/{self.epochs:02d}] -> Composite Loss: {avg_loss:.4f}")
                
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained before calling predict()!")
            
        self.model.eval()
        t_X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            out_b, out_f, out_c = self.model(t_X)
            pred_b = torch.argmax(out_b, dim=1).cpu().numpy()
            pred_f = torch.argmax(out_f, dim=1).cpu().numpy()
            pred_c = out_c.cpu().numpy()
            
        return {
            "building": pred_b,
            "floor": pred_f,
            "coords": pred_c
        }
