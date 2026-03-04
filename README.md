# iot-anomaly-detection-ml
Detection of Anomalous Behavior in IoT Network Traffic Using Machine Learning Methods
# IoT Anomaly Detection using Machine Learning

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.12-blue)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange)
  ![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-green)
  ![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-brightgreen)
  ![License](https://img.shields.io/badge/License-MIT-yellow)
  ![Research](https://img.shields.io/badge/Research-NIRS%20%26%20NIRM-red)
  
</div>

## 📋 Overview

This repository contains the complete code for detecting anomalous behavior in IoT network traffic using machine learning methods. The research was conducted for the **NIRS & NIRM University Research Competition 2026** and achieved state-of-the-art results on the TON_IoT dataset.

### 🏆 Key Achievements
- **Best Model:** SVM with 97.95% accuracy
- **F1-Score:** 0.9795
- **Dataset:** TON_IoT (460,000 records, 78 features, 10 classes)
- **Algorithms Compared:** 5 (Logistic Regression, Random Forest, XGBoost, LightGBM, SVM)

## 📊 Dataset: TON_IoT

The TON_IoT dataset was developed by the Cyber Range Lab at UNSW Canberra and contains real IoT network traffic with various cyberattacks.

**Dataset Characteristics:**
- 📦 **Size:** 460,000 records
- 🔢 **Features:** 78 (network, transport, application layers)
- 🎯 **Classes:** 10 (normal + 9 attack types)
- ⚖️ **Class Balance:** Perfectly balanced (1.00:1 ratio)
- 📁 **File:** `train_test_network.csv`

**Attack Types Included:**
- DDoS (Distributed Denial of Service)
- DoS (Denial of Service)
- Backdoor
- Injection (SQLi, XSS, Command Injection)
- Scanning (Port scan, vulnerability scan)
- MitM (Man-in-the-Middle)
- Ransomware
- Password attacks (Brute force, dictionary)
- XSS (Cross-site scripting)

## 🚀 Results

### Model Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | Training Time |
|-----------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 81.54% | 0.8162 | 0.8154 | 0.8152 | 18 sec |
| Random Forest | 93.56% | 0.9357 | 0.9356 | 0.9356 | 52 sec |
| XGBoost | 95.31% | 0.9531 | 0.9531 | 0.9531 | 75 sec |
| LightGBM | 94.86% | 0.9486 | 0.9486 | 0.9486 | 38 sec |
| **SVM (Best)** | **97.95%** | **0.9796** | **0.9795** | **0.9795** | 145 sec |

### Feature Importance (Top 10)

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | dst_bytes | 0.124 | Bytes from destination to source |
| 2 | src_bytes | 0.118 | Bytes from source to destination |
| 3 | duration | 0.097 | Connection duration |
| 4 | dst_pkts | 0.085 | Packets from destination |
| 5 | src_pkts | 0.079 | Packets from source |
| 6 | proto | 0.068 | Transport protocol |
| 7 | service | 0.054 | Application service |
| 8 | dst_port | 0.048 | Destination port |
| 9 | src_port | 0.042 | Source port |
| 10 | conn_state | 0.036 | Connection state |

## 📁 Repository Structure
```
├── 📄 iot_anomaly_detection.py    # Main code
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
├── 📄 LICENSE                       # MIT License
├── 📁 results/                      # Output
│   ├── 📄 model_results.csv
│   └── 📁 figures/
└── 📁 models/                       # Saved models
    └── 📄 best_model_svm.pkl
```
