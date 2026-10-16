"""
WiFi Indoor Positioning System (WIPS) - Realistic Synthetic & Benchmark Data Generator
Based on Log-Distance Path Loss Model, ITU-R P.1238 Indoor Propagation Recommendation,
and UJIIndoorLoc benchmark schema.
"""

import os
import json
import numpy as np
import pandas as pd

# Fix random seed for reproducible research
DEFAULT_SEED = 42

class IndoorEnvironment:
    """
    Defines multi-building indoor campus topology and AP placement.
    """
    def __init__(self, seed=DEFAULT_SEED):
        np.random.seed(seed)
        self.buildings = {
            0: {
                "name": "Engineering & Technology Hall",
                "width": 80.0,    # meters (X)
                "depth": 60.0,    # meters (Y)
                "floors": 4,      # Floors 0, 1, 2, 3
                "num_aps": 40,
                "lat_base": 37.4275,
                "lon_base": -122.1697
            },
            1: {
                "name": "Science & Library Complex",
                "width": 100.0,
                "depth": 70.0,
                "floors": 4,
                "num_aps": 50,
                "lat_base": 37.4285,
                "lon_base": -122.1680
            },
            2: {
                "name": "Student Innovation Center",
                "width": 60.0,
                "depth": 45.0,
                "floors": 3,
                "num_aps": 30,
                "lat_base": 37.4265,
                "lon_base": -122.1710
            }
        }
        self.floor_height = 3.8  # meters per floor
        self.aps = self._deploy_access_points()

    def _deploy_access_points(self):
        """
        Deploy Access Points systematically across building ceilings.
        """
        aps = []
        ap_global_idx = 1
        for b_id, b_info in self.buildings.items():
            num_aps = b_info["num_aps"]
            floors = b_info["floors"]
            aps_per_floor = int(np.ceil(num_aps / floors))
            
            for f in range(floors):
                # Place APs in an optimal grid pattern with slight jitter
                grid_cols = int(np.ceil(np.sqrt(aps_per_floor)))
                grid_rows = int(np.ceil(aps_per_floor / grid_cols))
                
                dx = b_info["width"] / (grid_cols + 1)
                dy = b_info["depth"] / (grid_rows + 1)
                
                count = 0
                for r in range(1, grid_rows + 1):
                    for c in range(1, grid_cols + 1):
                        if count >= aps_per_floor:
                            break
                        x = c * dx + np.random.uniform(-2.0, 2.0)
                        y = r * dy + np.random.uniform(-2.0, 2.0)
                        z = f * self.floor_height + 2.8 # ceiling mounted
                        ap_name = f"WAP{ap_global_idx:03d}"
                        aps.append({
                            "ap_id": ap_name,
                            "index": ap_global_idx,
                            "building": b_id,
                            "floor": f,
                            "x": np.clip(x, 2.0, b_info["width"] - 2.0),
                            "y": np.clip(y, 2.0, b_info["depth"] - 2.0),
                            "z": z,
                            "tx_power": -35.0  # RSSI at 1 meter in dBm
                        })
                        ap_global_idx += 1
                        count += 1
        return aps


class WIPSDataGenerator:
    """
    Simulates WiFi Fingerprints using Log-Distance Path Loss with Shadow Fading,
    Floor Attenuation Factor (FAF), Wall Attenuation (WAF), and Device Heterogeneity.
    """
    def __init__(self, environment=None, seed=DEFAULT_SEED):
        self.env = environment if environment is not None else IndoorEnvironment(seed=seed)
        self.seed = seed
        np.random.seed(seed)
        self.total_aps = len(self.env.aps)
        self.ap_names = [ap["ap_id"] for ap in self.env.aps]
        
        # Phone models and antenna sensitivities
        self.phone_profiles = {
            1: {"name": "iPhone 13 Pro", "bias": 0.0, "noise_scale": 1.0},
            2: {"name": "Samsung Galaxy S22", "bias": 1.5, "noise_scale": 1.1},
            3: {"name": "Google Pixel 7", "bias": -1.0, "noise_scale": 0.9},
            4: {"name": "OnePlus 9", "bias": 0.5, "noise_scale": 1.2},
            5: {"name": "iPad Pro", "bias": 2.0, "noise_scale": 0.8}
        }

    def compute_rssi(self, x, y, floor, building_id, phone_id=1, user_orientation=0.0):
        """
        Computes the RSSI vector for all Access Points at given 3D coordinate.
        """
        rssi_vector = np.full(self.total_aps, 100.0, dtype=np.float32) # 100 = unobserved (UJI standard)
        phone = self.phone_profiles.get(phone_id, self.phone_profiles[1])
        z_user = floor * self.env.floor_height + 1.2  # phone held at hand level (~1.2m)
        
        for i, ap in enumerate(self.env.aps):
            ap_building = ap["building"]
            ap_floor = ap["floor"]
            
            # Cross-building attenuation is immense (> 60 dB drop due to exterior concrete walls)
            if ap_building != building_id:
                # Slight chance of detection if buildings are close, but highly attenuated
                dist_xy = np.hypot(x - ap["x"], y - ap["y"]) + 40.0
                ext_wall_attenuation = 45.0
            else:
                dist_xy = np.hypot(x - ap["x"], y - ap["y"])
                ext_wall_attenuation = 0.0

            dist_z = abs(z_user - ap["z"])
            dist_3d = np.sqrt(dist_xy**2 + dist_z**2)
            dist_3d = max(dist_3d, 0.8) # avoid singularity at d=0

            # Path loss exponent: 2.0 (free space) to 3.2 (cluttered indoor)
            floor_diff = abs(floor - ap_floor)
            if ap_building == building_id and floor_diff == 0:
                n = 2.4
            else:
                n = 3.0

            # Reference power at 1 meter
            p0 = ap["tx_power"]

            # Floor Attenuation Factor (FAF): ~16 dB for 1 floor, ~25 dB for 2 floors
            if floor_diff == 0:
                faf = 0.0
            elif floor_diff == 1:
                faf = 17.0
            elif floor_diff == 2:
                faf = 26.0
            else:
                faf = 34.0

            # Wall Attenuation (estimated by distance)
            num_walls = int(dist_xy / 12.0)
            waf = num_walls * 3.5

            # Human Body Shadowing (user holding phone attenuates rearward signals by 3-5 dB)
            angle_to_ap = np.arctan2(ap["y"] - y, ap["x"] - x)
            angle_diff = abs((user_orientation - angle_to_ap + np.pi) % (2 * np.pi) - np.pi)
            body_shadowing = 4.0 if angle_diff > (np.pi / 2.0) else 0.0

            # Log-normal shadowing noise N(0, sigma^2)
            sigma = 3.5 * phone["noise_scale"]
            shadowing_noise = np.random.normal(0, sigma)

            # Total expected RSSI
            rssi = (p0 
                    - (10 * n * np.log10(dist_3d)) 
                    - faf 
                    - waf 
                    - ext_wall_attenuation 
                    - body_shadowing 
                    + phone["bias"] 
                    + shadowing_noise)

            # Receiver Sensitivity Threshold (~ -98 dBm)
            # Anything below -98 dBm is not registered by client Wi-Fi chipset
            if rssi >= -98.0 and rssi <= -20.0:
                rssi_vector[i] = np.round(rssi, 1)
            else:
                rssi_vector[i] = 100.0  # UJIIndoorLoc convention for not detected
                
        return rssi_vector

    def generate_dataset(self, num_samples=4000, split="train", seed=None):
        """
        Generates structured dataset of fingerprint records.
        """
        if seed is not None:
            np.random.seed(seed)
            
        records = []
        base_timestamp = 1697443200 # Fixed base epoch
        
        for sample_idx in range(num_samples):
            # Select building according to size/activity
            b_id = np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
            b_info = self.env.buildings[b_id]
            
            floor = np.random.randint(0, b_info["floors"])
            
            # Position within building
            x = np.random.uniform(1.0, b_info["width"] - 1.0)
            y = np.random.uniform(1.0, b_info["depth"] - 1.0)
            
            # Simulated space / room
            room_x = int(x / 10.0)
            room_y = int(y / 10.0)
            space_id = room_y * 10 + room_x + 100
            
            # Global GPS estimation
            lat = b_info["lat_base"] + (y / 111320.0)
            lon = b_info["lon_base"] + (x / (111320.0 * np.cos(np.radians(b_info["lat_base"]))))
            
            phone_id = np.random.choice(list(self.phone_profiles.keys()))
            user_id = np.random.randint(1, 15)
            orientation = np.random.uniform(-np.pi, np.pi)
            timestamp = base_timestamp + sample_idx * 15 + np.random.randint(0, 10)
            
            rssi = self.compute_rssi(x, y, floor, b_id, phone_id, orientation)
            
            # Create row dict
            row = {self.ap_names[i]: rssi[i] for i in range(self.total_aps)}
            row["LONGITUDE"] = lon
            row["LATITUDE"] = lat
            row["FLOOR"] = floor
            row["BUILDINGID"] = b_id
            row["SPACEID"] = space_id
            row["RELATIVEPOSITION"] = 1 if np.random.rand() > 0.5 else 2
            row["USERID"] = user_id
            row["PHONEID"] = phone_id
            row["TIMESTAMP"] = timestamp
            row["X"] = round(x, 2)
            row["Y"] = round(y, 2)
            
            records.append(row)
            
        df = pd.DataFrame(records)
        return df

    def save_full_dataset(self, output_dir="data"):
        """
        Creates train, validation, and test splits with metadata.
        """
        os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
        
        print(f"Generating WIPS WiFi Fingerprint Dataset ({self.total_aps} Access Points across 3 Buildings)...")
        train_df = self.generate_dataset(num_samples=3500, split="train", seed=42)
        val_df = self.generate_dataset(num_samples=800, split="val", seed=100)
        test_df = self.generate_dataset(num_samples=1200, split="test", seed=200)
        
        train_path = os.path.join(output_dir, "raw", "train_data.csv")
        val_path = os.path.join(output_dir, "raw", "val_data.csv")
        test_path = os.path.join(output_dir, "raw", "test_data.csv")
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        meta = {
            "dataset_name": "WIPS-Campus-WiFi-Fingerprints",
            "total_aps": self.total_aps,
            "ap_columns": self.ap_names,
            "buildings": self.env.buildings,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "rssi_range": "[-98 dBm, -20 dBm], unobserved = 100",
            "access_points_info": self.env.aps
        }
        
        meta_path = os.path.join(output_dir, "building_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"Data generation complete!")
        print(f"Train set: {train_path} ({len(train_df)} records)")
        print(f"Val set:   {val_path} ({len(val_df)} records)")
        print(f"Test set:  {test_path} ({len(test_df)} records)")
        print(f"Metadata:  {meta_path}")
        return train_path, val_path, test_path, meta_path

if __name__ == "__main__":
    generator = WIPSDataGenerator()
    generator.save_full_dataset(output_dir="/Users/mrudul1206/.gemini/antigravity/scratch/WIPS/data")
