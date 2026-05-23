# Firewall Log Classifier

A machine learning system for automated classification of firewall log entries into four action categories: allow, deny, drop, and reset-both. Built as part of CSAI 801 — Artificial Intelligence and Machine Learning.

**Live Application:** https://huggingface.co/spaces/yomnafarag95/Log_Classifier

---

## Overview

Enterprise firewalls generate thousands of log entries per hour, making manual review impractical. This project trains a tuned Random Forest classifier on real network traffic data to automate that review process, achieving 99.56% test accuracy across four action classes.

---

## Model Performance

| Model                   | Test Accuracy | Macro F1 |
|-------------------------|--------------|----------|
| Random Forest (baseline)| 98.32%       | 0.981    |
| Logistic Regression     | 99.75%       | 0.997    |
| KNN                     | 99.23%       | 0.990    |
| Random Forest (tuned)   | **99.56%**   | **0.803**|

Tuned hyperparameters: `n_estimators=200`, `max_depth=20`, `min_samples_split=2`

---

## Dataset

- **Source:** UCI Machine Learning Repository — Internet Firewall Data
- **URL:** https://archive.ics.uci.edu/dataset/542/internet+firewall+data
- **Raw records:** 65,532
- **After deduplication:** 57,170
- **Class distribution:** allow (37,439) · drop (11,635) · deny (8,042) · reset-both (54)

**Input features (11):**

| # | Feature              |
|---|----------------------|
| 1 | Source Port          |
| 2 | Destination Port     |
| 3 | NAT Source Port      |
| 4 | NAT Destination Port |
| 5 | Bytes                |
| 6 | Bytes Sent           |
| 7 | Bytes Received       |
| 8 | Packets              |
| 9 | Elapsed Time (sec)   |
|10 | pkts_sent            |
|11 | pkts_received        |

---

## Preprocessing Pipeline

1. Duplicate removal (65,532 → 57,170 records)
2. Stratified 70/30 train/test split
3. SMOTE oversampling on training set to balance minority classes
4. StandardScaler normalization

---

## Try the Application

Paste any of the following lines into the application input and click Classify.
Each line contains 11 comma-separated values matching the feature order above.

**Allow**
```
51465,443,39975,443,3961,1595,2366,21,16,12,9
```

**Deny**
```
34086,25174,0,0,62,62,0,1,0,1,0
```

**Drop**
```
51125,445,0,0,66,66,0,1,0,1,0
```

**Reset-Both**
```
64461,31652,0,0,62,62,0,1,0,1,0
```

---

## Run Locally

```bash
git clone https://github.com/yomnafarag95/Log_Classifier.git
cd Log_Classifier
pip install -r requirements.txt
streamlit run app.py
```

---

## Retrain the Model

```bash
pip install scikit-learn imbalanced-learn pandas joblib
python retrain.py
```

Outputs: `model.joblib`, `scaler.joblib`, `label_encoder.joblib`

---

## Repository Structure

```
Log_Classifier/
├── app.py                    Streamlit web application
├── retrain.py                Model retraining script
├── model.joblib              Trained Random Forest model
├── scaler.joblib             Fitted StandardScaler
├── label_encoder.joblib      Label encoder for action classes
├── requirements.txt          Python dependencies
└── The_Report.pdf            Full project report
```

---

## Authors

Yasmeen Algendy, Yomna Algendy, Zahraa Mohamed
Supervisor: Dr. Marwa Elsayed
CSAI 801 — Artificial Intelligence and Machine Learning
