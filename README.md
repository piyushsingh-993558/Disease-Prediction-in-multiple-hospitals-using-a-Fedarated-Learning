# Federated Diabetes Prediction with Flower + FedAvg

## 📊 Project Overview

Train a diabetes prediction model collaboratively across simulated hospitals without sharing raw patient data. Each hospital trains on its local data and only sends model parameters to a central server. The server combines them using **Federated Averaging (FedAvg)** and sends the improved global model back out.

## 🏗️ Week 1 Foundation - What's Included

This repository contains the complete Week 1 foundation for the federated learning project:

### 📈 Data Pipeline (Completed)
- **Dataset:** `diabetes_binary_health_indicators_BRFSS2015.csv` (253,680 rows originally)
- **Deduplication:** 24,206 exact duplicate rows removed (~10%)
- **Cleaned data:** 229,474 rows saved to `data/processed/diabetes_cleaned.csv`
- **Global test holdout:** 20% (45,895 rows), stratified, server-only
- **5 client partitions:** Equal IID splits from 80% training pool
- **Local train/val splits:** 80/20 per client
- **Feature scaling:** Fixed-range min-max [0,1] on 7 continuous features, fitted on global train only

### 📊 Baselines Trained & Recorded

| Baseline | Metrics (on Global Test Set) |
|----------|------------------------------|
| **Centralized** (all data pooled) | Accuracy: 84.72% | AUC-ROC: 0.7544 |
| **Local-only** (each hospital alone) | Avg AUC-ROC: 0.4342 | Range: 0.4237 – 0.4471 |

### 🔧 Flower Framework
- **Version:** `flwr==1.32.1`
- **Smoke test:** Verified import and basic simulation
- **FedAvg strategy:** Configured with `fraction_fit=1.0`, `min_fit_clients=2`

### 🧪 Reproducibility
- **Seed helper:** `src/utils/seed.py` with `set_seed(42)`
- **Random seeds:** `random`, `numpy`, `torch` all seeded

## 🚀 Quick Start - How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/piyushsingh-993558/Disease-Prediction-in-multiple-hospitals-using-a-Fedarated-Learning.git
cd Disease-Prediction-in-multiple-hospitals-using-a-Fedarated-Learning
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** If torch installation fails with SSL errors (Python 3.14 on Windows), use:
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.python.org torch
```

### 4. Verify Installation
```bash
python -c "import flwr as fl; print(f'Flower {fl.__version__} OK')"
python -c "import torch; print(f'Torch {torch.__version__} OK')"
```

### 5. Explore the Data Pipeline
```bash
python verify_dataset.py
```
This shows:
- Cleaned dataset shape and class balance
- Global test split statistics
- 5 client partitions with label distributions
- Scaler information

### 6. Run Centralized Baseline
```bash
python src\centralized_baseline.py
```
**Output:** Accuracy 84.72%, AUC-ROC 0.7544 (upper bound)

### 7. Run Local-Only Baseline
```bash
python src\local_only_baseline.py
```
**Output:** Average AUC-ROC 0.4342 (lower bound, each hospital alone)

## 📁 Repository Structure

```
Disease-Prediction-in-multiple-hospitals-using-a-Fedarated-Learning/
├── README.md                    ← This file
├── requirements.txt             ← Pinned dependencies
├── pyproject.toml               ← (optional)
├── data/
│   ├── processed/diabetes_cleaned.csv   ← Deduped BRFSS data (229,474 rows)
│   ├── global_test_set.csv              ← Server-only holdout (45,895 rows)
│   └── partitions/                      ← 5 client partitions
│       ├── client_0_train.csv
│       ├── client_0_val.csv
│       ├── client_1_train.csv
│       ├── client_1_val.csv
│       ├── client_2_train.csv
│       ├── client_2_val.csv
│       ├── client_3_train.csv
│       ├── client_3_val.csv
│       ├── client_4_train.csv
│       └── client_4_val.csv
├── src/
│   ├── model.py                ← LogisticRegression + SmallMLP, get_parameters/set_parameters
│   ├── train_eval.py           ← train_one_epoch, evaluate, create_dataloader
│   ├── centralized_baseline.py ← Upper-bound baseline (already trained)
│   ├── local_only_baseline.py  ← Lower-bound baseline (already trained)
│   ├── utils/seed.py           ← set_seed(42) reproducibility helper
│   ├── *(8 empty dirs for Phases 2-11)*
├── experiments/                ← (Week 3-4: experiment scripts)
├── results/                    ← (Week 3-4: logs and figures)
├── notebooks/                 ← (EDA exploration)
├── flower_simulation_test.py   ← Flower simulation verification
├── generate_dataset.py         ← Custom dataset generator
├── phase2_split.py             ← Train/test split and preprocessing
├── process_dataset.py          ← Dataset cleaning pipeline
├── verify_dataset.py           ← Dataset verification script
└── verify_phase2.py            ← Phase 2 verification
```

## 🛠️ Development Phases (Roadmap)

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data exploration, cleaning, dedup | ✅ Complete |
| **Phase 2** | Train/test split, preprocessing | ✅ Complete |
| **Phase 3-4** | Partition code, IID/non-IID splits | ✅ Complete |
| **Phase 5** | Model definition (PyTorch) | ✅ Complete |
| **Phase 6** | Centralized baseline | ✅ Complete |
| **Phase 7** | Local-only baseline | ✅ Complete |
| **Phase 8** | Flower NumPyClient with real partitions | 🟡 Week 2 |
| **Phase 9** | FedAvg strategy + FedProx comparison | 🔜 Week 2-3 |
| **Phase 10** | Real separate processes (server + clients) | 🔜 Week 2-3 |
| **Phase 11** | Experiments E1-E8 (IID vs non-IID, epochs, clients) | 🔜 Week 3-4 |
| **Phase 12-14** | Evaluation, plots, report, optional extensions | 🔜 Week 4+ |

## 📊 Key Metrics Summary

### Centralized Baseline (Upper Bound)
- Trained on: **All pooled training data** (183,579 rows)
- Tested on: **Global test holdout** (45,895 rows)
- Accuracy: **84.72%**
- Precision: **0.5938**
- Recall: **0.0027** (class imbalance)
- F1: **0.0054**
- **AUC-ROC: 0.7544** ← Target for FL

### Local-Only Baseline (Lower Bound)
- Each hospital trains on its partition only
- No collaboration
- Tested on: **Global test holdout** (45,895 rows)
- Average AUC-ROC: **0.4342**
- Per-client range: **0.4237 – 0.4471**
- Recall: **0.0000** (default 0.5 threshold on imbalanced data)
- F1: **0.0000**

**The gap between 0.7544 and 0.4342 demonstrates the potential benefit of federation.**

## 🔍 Important Notes

### Dataset Information
- **Source:** BRFSS 2015 (`diabetes_binary_health_indicators_BRFSS2015.csv`)
- **Original rows:** 253,680
- **After dedup:** 229,474 (24,206 duplicates removed)
- **Features:** 21 (HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income)
- **Class balance:** 15.29% positive / 84.71% negative
- **No missing values** in original dataset

### Dependencies (requirements.txt)
```
torch==2.14.1
flwr==1.32.1
pandas==3.0.6
scikit-learn==1.9.1
matplotlib==3.11.2
numpy==2.5.4
```

### Torch SSL Workaround (Windows + Python 3.14)
If `pip install torch` fails with SSL errors:
```bash
pip install --trusted-host pypi.org \
              --trusted-host pypi.python.org \
              --trusted-host files.python.org torch
```

### Privacy Motivation
- Raw patient data **never leaves** a client hospital
- Only model parameters (weights) are exchanged
- Each client sees only its own partition
- Fixed-range min-max scaling means no client needs another's statistics

### FedAvg Update Rule
```
w_global(t+1) = Σ_k ( n_k / n ) · w_k(t+1)
```
- `w_k` = weights from hospital k after local training
- `n_k` = number of training samples at hospital k
- `n` = total training samples across participating hospitals

## 🎯 Next Steps (Week 2-4)

1. **Flower Federation:** Run server + 5 client processes
2. **FedAvg Strategy:** Configure and run federated training
3. **Experiments E1-E8:** IID vs non-IID, local epochs, client participation
4. **FedProx Comparison:** Proximal term with μ sweep
5. **Model Analysis:** Threshold tuning for higher recall
6. **Plots & Report:** Convergence curves, final results tables
7. **Documentation:** Full README, slides, demo script

## 👥 Team & Contact

- **Repository:** GitHub
- **Project Duration:** 1 month
- **Team Size:** 3 members
- **Deadline:** 1 month from start

---

## 📜 License

This project is for educational and research purposes. 

---

**Happy Federated Learning! 🏥🤖**