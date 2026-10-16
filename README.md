# WIPS: WiFi Indoor Positioning System 📡🏢

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)](#testing)

An advanced Machine Learning and Deep Learning system for indoor localization using WiFi Received Signal Strength Indication (RSSI) fingerprinting. Inspired by the seminal Stanford CS229 research (*"Indoor Positioning System Using Wifi Fingerprint"*, Dan Li, Le Wang, Shiqi Wu) and benchmarked against standard multi-building campus topologies (UJIIndoorLoc format).

---

## 📖 Table of Contents
- [Problem Overview & Motivation](#-problem-overview--motivation)
- [System Architecture](#-system-architecture)
- [Dataset & Physical Propagation Model](#-dataset--physical-propagation-model)
- [Machine Learning & Deep Learning Suite](#-machine-learning--deep-learning-suite)
- [Benchmark Results & Comparative Analysis](#-benchmark-results--comparative-analysis)
- [Interactive Web Dashboard](#-interactive-web-dashboard)
- [Quick Start Guide](#-quick-start-guide)
- [Presentation & Viva Defense Guide](#-presentation--viva-defense-guide)

---

## 🎯 Problem Overview & Motivation

Global Positioning System (GPS) technology is ubiquitous outdoors, but it **completely fails indoors** due to the absence of direct Line-of-Sight (LoS) to satellite constellations, combined with heavy signal attenuation through reinforced concrete, brick walls, and steel structures.

Modern multi-story commercial and campus buildings are already densely saturated with **WiFi Access Points (APs)**. Every mobile device (smartphone, tablet, laptop) periodically receives beacon packets containing:
1. **BSSID (MAC Address)** of the Access Point.
2. **RSSI (Received Signal Strength Indication)** measured in negative dBm (e.g. $-30\text{ dBm}$ near the router, $-90\text{ dBm}$ near reception limit).

**WiFi Fingerprinting** transforms the complex, multipath-rich indoor radio environment from a hindrance into an advantage by constructing a distinctive spatial *radio map* (fingerprint database) that uniquely identifies indoor locations.

---

## 🏗️ System Architecture

The WIPS system operates in two distinct phases:

```
+-----------------------------------------------------------------------------------+
|                            OFFLINE CALIBRATION PHASE                              |
|                                                                                   |
|   Physical Survey       Raw RSSI Matrix          Feature Engineering              |
|   [Known (X,Y,Floor)] -> [WAP001...WAP122] -> [Exponential Shift & Normalization] |
|                                                            |                      |
|                                                            v                      |
|                                                  [Trained Model Weights]          |
|                                            (DNN, Hierarchical RF, WKNN)           |
+-----------------------------------------------------------------------------------+
                                                            |
+-----------------------------------------------------------------------------------+
|                             ONLINE POSITIONING PHASE                              |
|                                                                                   |
|   Mobile Client Scan    Normalized Vector        Multi-Stage Inference            |
|   [Real-time RSSI]   -> [Preprocessor]      ->   1. Predict Building              |
|                                                  2. Predict Floor                 |
|                                                  3. Predict 2D (X, Y) Coordinates |
|                                                            |                      |
|                                                            v                      |
|                                               [Live Dashboard & Location Fix]     |
+-----------------------------------------------------------------------------------+
```

---

## 🔬 Dataset & Physical Propagation Model

The dataset simulates a 3-building university campus with **122 Access Points** across multiple floors, generated using the **Log-Distance Path Loss Model with Shadow Fading** (ITU-R P.1238 recommendation):

$$RSSI(d) = P_0 - 10 \cdot n \cdot \log_{10}(d) - FAF(\Delta\text{floor}) - WAF(N_{\text{walls}}) + \Delta_{\text{device}} + X_\sigma$$

Where:
- $P_0 = -35\text{ dBm}$: Reference received power at $1\text{ meter}$.
- $n \in [2.4, 3.0]$: Indoor path-loss exponent.
- $FAF$: Floor Attenuation Factor ($17\text{ dB}$ for 1 floor, $26\text{ dB}$ for 2 floors).
- $WAF$: Internal Wall Attenuation Factor ($3.5\text{ dB}$ per internal wall crossed).
- $\Delta_{\text{device}}$: Device heterogeneity bias ($\pm 2.5\text{ dBm}$) modeling iPhone, Galaxy, and Pixel antenna differences.
- $X_\sigma \sim \mathcal{N}(0, \sigma^2)$: Log-normal shadow fading ($\sigma = 3.5\text{ dB}$) modeling human blockage and multipath reflections.
- Unobserved Access Points are encoded as `100` following the gold-standard **UJIIndoorLoc** schema.

### Dataset Splits
- **Training Set**: 3,500 samples (`data/raw/train_data.csv`)
- **Validation Set**: 800 samples (`data/raw/val_data.csv`)
- **Independent Test Set**: 1,200 samples (`data/raw/test_data.csv`)

---

## 🧠 Machine Learning & Deep Learning Suite

WIPS implements six distinct localization algorithms:

1. **$k$-Nearest Neighbors ($k$-NN)**: Classical Euclidean pattern matching baseline.
2. **Weighted $k$-NN (WKNN)**: Inverse distance weighting giving higher influence to closer radio fingerprints:
   $$w_i = \frac{1}{d_i + \epsilon}, \quad \hat{\mathbf{x}} = \frac{\sum_{i=1}^k w_i \mathbf{x}_i}{\sum_{i=1}^k w_i}$$
3. **Random Forest (RF)**: Ensemble of 100 decision trees for multi-class building/floor classification and multi-output coordinate regression. Extracts Access Point Gini feature importance.
4. **Support Vector Machines (SVM/SVR)**: Multi-class RBF kernel SVM for building/floor classification, and Support Vector Regression (SVR) for coordinates (Stanford CS229 methodology).
5. **Multi-Task Deep Neural Network (PyTorch)**: Shared representation backbone (BatchNorm + ReLU + Dropout) with three specialized prediction heads trained jointly with composite loss:
   $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(X,Y) + 15 \cdot \mathcal{L}_{\text{CE}}(\text{Floor}) + 25 \cdot \mathcal{L}_{\text{CE}}(\text{Building})$$
6. **Hierarchical Multi-Stage Architecture**: Sequential multi-stage pipeline:
   - **Stage 1**: Campus-wide Building Classification ($100\%$ accuracy).
   - **Stage 2**: Building-specific Floor Classification ($100\%$ accuracy).
   - **Stage 3**: (Building, Floor)-conditioned Fine Coordinate Regression. Eliminates impossible cross-floor and cross-building errors.

---

## 📊 Benchmark Results & Comparative Analysis

All models evaluated on the independent $1,200$-sample test set:

| Model Architecture | Building Acc (%) | Floor Acc (%) | Mean 2D Error (m) | Median Error (m) | 75th %tile (m) | 95th %tile (m) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Multi-Task DNN** | **100.0%** | **100.0%** | **4.52 m** | **4.09 m** | **5.86 m** | **9.75 m** | $0.21\text{ ms}$ |
| **WKNN (Inverse Distance)** | **100.0%** | **100.0%** | **4.61 m** | **4.05 m** | **5.97 m** | **10.01 m** | **0.08 ms** |
| **k-NN (Euclidean)** | 100.0% | 100.0% | 4.65 m | 4.07 m | 6.02 m | 10.30 m | 0.07 ms |
| **Hierarchical Multi-Stage** | 100.0% | 100.0% | 5.96 m | 4.91 m | 7.41 m | 13.61 m | 0.45 ms |
| **Support Vector Machine (SVM/SVR)** | 100.0% | 100.0% | 6.12 m | 4.96 m | 7.77 m | 14.87 m | 0.82 ms |
| **Random Forest** | 100.0% | 100.0% | 7.54 m | 6.11 m | 9.44 m | 19.07 m | 1.15 ms |

### Publication-Quality Visualizations
The system automatically generates high-resolution figures in `outputs/plots/`:
- **CDF Error Curves (`cdf_error_comparison.png`)**: Shows that over $70\%$ of predictions fall within room-level resolution ($\le 5\text{ meters}$).
- **Model Error Bar Chart (`model_error_comparison.png`)**: Direct comparison of mean, median, and accuracy.
- **2D Floor Localization Map (`floor_prediction_map.png`)**: Visualizes ground truth vs. predicted positions with error vectors and AP locations.
- **AP Feature Importance (`ap_feature_importance.png`)**: Identifies top discriminative Access Points across campus.

---

## 💻 Interactive Web Dashboard

Launch the zero-dependency interactive web dashboard:
```bash
python3 scripts/serve_dashboard.py
```
Open [http://localhost:8000/web/index.html](http://localhost:8000/web/index.html) in your browser:
- **Interactive Floor Plan**: 2D campus canvas displaying AP antennas, user coordinates, and real-time prediction vectors.
- **Live RSSI Signal Meters**: Visualizes dynamic dBm signal strengths from audible Access Points.
- **Live Telemetry & Diagnostics**: Ground truth vs. predicted coordinates and positioning error in meters.
- **Model Benchmark Table**: Full comparative metrics across algorithms.

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/MrudulMani/WIPS.git
cd WIPS
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python3 -m unittest discover tests
```

### 3. Run Full Training & Benchmarking Pipeline
```bash
python3 scripts/run_all.py
```

### 4. Launch Web Visualizer
```bash
python3 scripts/serve_dashboard.py
```

---

## 🎓 Presentation & Viva Defense Guide

A complete slide-by-slide script, mathematical breakdown, and anticipated examiner questions are documented in [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md).
