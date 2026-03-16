# Federated Fraud Detection with XGBoost

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Thesis-purple)
![Privacy](https://img.shields.io/badge/Privacy-Federated%20Learning-teal)
![Accuracy](https://img.shields.io/badge/Accuracy-EXCLUDED%20%E2%80%94%200.13%25%20imbalance-red)

A privacy-preserving **Federated Learning** framework for early
financial fraud detection using **XGBoost** and the novel
**JSON Tree Concatenation Algorithm**.

---

## Thesis

| Field | Detail |
|---|---|
| **Title** | A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy |
| **Author** | Sothun Darachhat |
| **Institution** | Royal University of Phnom Penh |
| **Degree** | Bachelor of Engineering in Information Technology Engineering |
| **Supervisor** | Mr. Chhim Bunchhun |
| **Dataset** | PaySim Synthetic Financial Dataset |

---

## The Problem

Financial institutions that serve low-risk customer segments
accumulate transaction histories with **zero confirmed fraud cases**.
A supervised fraud detection model trained on such data achieves:
```
Bank 2 (Retail) — Local-Only Baseline:
  AUPRC    = 0.0000   ← Total detection failure
  F1-Score = 0.0000   ← Total detection failure
```

This is the **blind spot problem** — data isolation condemns
retail banks to zero fraud detection capability.

---

## The Solution

The **JSON Tree Concatenation Algorithm** enables three banks
to collaboratively train a fraud detection model without
transferring any raw transaction data:
```
Bank 1 ──► local XGBoost model ──► JSON trees ──┐
Bank 2 ──► local XGBoost model ──► JSON trees ──┼──► Global Server
Bank 3 ──► local XGBoost model ──► JSON trees ──┘         │
                                                    Concatenate trees
                                                           │
Bank 1 ◄── federated global model ◄────────────────────────┤
Bank 2 ◄── federated global model ◄────────────────────────┤
Bank 3 ◄── federated global model ◄────────────────────────┘

Result: Bank 2 AUPRC = 0.0000 → [FL Round 5 score]
        No raw data ever leaves any institution.
```

---

## Architecture
```
federated-fraud-detection/
│
├── config/                     # YAML configuration files
│   ├── base_config.yaml        # Global hyperparameters & paths
│   ├── bank1_config.yaml       # Bank 1 — High-Risk profile
│   ├── bank2_config.yaml       # Bank 2 — Retail / Blind Spot
│   └── bank3_config.yaml       # Bank 3 — Mixed profile
│
├── data/
│   ├── raw/                    # Original PaySim CSV
│   ├── processed/              # Preprocessed per-bank CSVs
│   └── partitions/             # Partition metadata
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py     # Feature engineering pipeline
│   │   └── partitioner.py      # Non-IID partitioning logic
│   ├── models/
│   │   └── xgboost_trainer.py  # Local XGBoost training
│   ├── federated/
│   │   ├── server.py           # Global Server coordinator
│   │   ├── client.py           # Federated client logic
│   │   └── aggregator.py       # JSON Tree Concatenation Algorithm
│   ├── evaluation/
│   │   └── metrics.py          # AUPRC + F1 only (no Accuracy)
│   └── utils/
│       ├── logger.py           # Structured per-component logging
│       ├── seed.py             # Reproducibility seed control
│       └── io_utils.py         # File I/O and sync utilities
│
├── scripts/
│   ├── prepare_data.py         # One-time data preparation
│   ├── run_baseline_local.py   # Local-Only baseline experiment
│   ├── run_baseline_central.py # Centralized baseline experiment
│   ├── run_server.py           # Launch Global Server (Terminal 0)
│   ├── run_client.py           # Launch client bank (Terminals 1-3)
│   └── run_simulation.py       # Single-process simulation mode
│
├── experiments/results/        # Saved JSON results per round
├── models/                     # Serialized XGBoost model files
├── logs/                       # Per-component structured logs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_federated_results_visualization.ipynb
│
├── CLAUDE.md                   # Instructions for Claude AI
├── SKILL.md                    # Reusable project skill procedures
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Federated Learning Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                     FEDERATED ROUND N                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   BANK 1     │  │   BANK 2     │  │   BANK 3     │         │
│  │  High-Risk   │  │   Retail     │  │    Mixed     │         │
│  │              │  │  (Blind Spot)│  │              │         │
│  │ Local Train  │  │ Local Train  │  │ Local Train  │         │
│  │  XGBoost     │  │  XGBoost     │  │  XGBoost     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │ JSON trees      │ JSON trees       │ JSON trees      │
│         │ (no raw data)   │ (no raw data)    │ (no raw data)   │
│         └─────────────────▼──────────────────┘                 │
│                    ┌──────────────┐                             │
│                    │ GLOBAL SERVER│                             │
│                    │              │                             │
│                    │  JSON Tree   │                             │
│                    │Concatenation │                             │
│                    │  Algorithm   │                             │
│                    └──────┬───────┘                             │
│                           │ Federated global model              │
│         ┌─────────────────┼──────────────────┐                 │
│         ▼                 ▼                  ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   BANK 1     │  │   BANK 2     │  │   BANK 3     │         │
│  │  Evaluate    │  │  Evaluate    │  │  Evaluate    │         │
│  │  AUPRC + F1  │  │  AUPRC + F1  │  │  AUPRC + F1  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘

Privacy Guarantee: NO raw transaction data ever leaves any bank.
```

---

## Setup
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/federated-fraud-detection.git
cd federated-fraud-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install as editable package
pip install -e .

# 4. Download PaySim dataset
# Visit: https://www.kaggle.com/datasets/ealaxi/paysim1
# Place the downloaded CSV at: data/raw/paysim.csv
```

---

## Quickstart

### Step 1 — Prepare Data
```bash
# Run once before any experiments
python scripts/prepare_data.py
```

### Step 2 — Run Baselines
```bash
# Local-Only: each bank trains in isolation (blind spot demo)
python scripts/run_baseline_local.py

# Centralized: all data pooled (privacy-violated upper bound)
python scripts/run_baseline_central.py
```

### Step 3 — Run Federated Learning

**Option A — Single-process simulation (recommended)**
```bash
python scripts/run_simulation.py --rounds 5 --seed 42
```

**Option B — Distributed 4-terminal mode**
```bash
# Open 4 terminals. Start server first, then all three clients.

# Terminal 0 — Global Server
python scripts/run_server.py

# Terminal 1 — Bank 1 (High-Risk)
python scripts/run_client.py --bank bank1

# Terminal 2 — Bank 2 (Retail / Blind Spot)
python scripts/run_client.py --bank bank2

# Terminal 3 — Bank 3 (Mixed)
python scripts/run_client.py --bank bank3
```

### Step 4 — Visualize Results
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_baseline_analysis.ipynb
jupyter notebook notebooks/03_federated_results_visualization.ipynb
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **XGBoost over neural networks** | Superior performance on structured tabular financial data |
| **AUPRC + F1 only — no Accuracy** | Accuracy is 99.87% for a zero-detection classifier under 0.13% fraud |
| **JSON Tree Concatenation** | Non-invasive FL aggregation grounded in XGBoost additive scoring |
| **Non-IID partitioning** | Reflects realistic heterogeneity across bank risk profiles |
| **Bank 2 zero fraud labels** | Demonstrates the blind spot problem quantitatively |
| **File-based communication** | Simple, auditable, portable across environments |
| **YAML configuration** | All hyperparameters externalized — no hardcoded values |
| **Structured logging** | Per-component logs with rotation for full experiment traceability |

---

## Evaluation Metrics

| Metric | Used | Reason |
|---|---|---|
| **AUPRC** | ✅ Primary | Handles extreme class imbalance — measures fraud class directly |
| **F1-Score** | ✅ Primary | Threshold-specific precision-recall balance |
| **Precision** | ✅ Supporting | Reported alongside F1 for operational context |
| **Recall** | ✅ Supporting | Reported alongside F1 for operational context |
| **Accuracy** | ❌ Excluded | Degenerate classifier scores 99.87% by predicting all-legitimate |

---

## Experimental Results

| Condition | Bank 1 AUPRC | Bank 2 AUPRC | Bank 3 AUPRC |
|---|---|---|---|
| **Local-Only** | 0.9514 | **0.0000** ⚠ | 0.9263 |
| **Centralized** | 0.9442 | 0.9442 | 0.9442 |
| **FL Round 5** | TBD | TBD | TBD |

> ⚠ Bank 2 AUPRC = 0.0000 represents total fraud detection failure
> under data isolation — the **blind spot problem**.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
```bash
# Ensure package is installed in editable mode
pip install -e .
# OR add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### `FileNotFoundError: data/raw/paysim.csv`
```bash
# Download PaySim from Kaggle and place at correct path
# https://www.kaggle.com/datasets/ealaxi/paysim1
mkdir -p data/raw
mv ~/Downloads/PS_20174392719_1491204439457_log.csv data/raw/paysim.csv
```

### `TimeoutError: waiting for client model at round N`
```bash
# In distributed mode, ensure all 4 terminals are running
# before any round begins. Start server first.
# Increase timeout in config/base_config.yaml:
#   federated.round_timeout_seconds: 600
```

### `KeyError in JSON tree concatenation`
```bash
# XGBoost version mismatch — ensure all clients use same version
pip install xgboost==1.7.6
# Verify: python -c "import xgboost; print(xgboost.__version__)"
```

### `AUPRC = 0.0 for Bank 2 after federation`
```bash
# This is expected at Round 0 (Local-Only baseline).
# If it persists after Round 1, verify:
# 1. Global model file exists in models/global/
# 2. Bank 1 and Bank 3 models contain fraud-discriminative trees
# 3. scale_pos_weight is set correctly in base_config.yaml
```

### Jupyter kernel not finding project modules
```bash
# Register the virtual environment as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=federated-fraud --display-name "Federated Fraud"
# Then select "Federated Fraud" kernel in Jupyter
```

---

## Contribution Guidelines

This is a research thesis project. Contributions that improve
reproducibility, documentation, or experimental coverage are welcome.

### How to Contribute
```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-improvement

# 3. Make changes following the code style below
# 4. Run existing experiments to confirm nothing is broken
python scripts/run_simulation.py --rounds 2 --seed 42

# 5. Submit a pull request with a clear description
```

### Code Style Guidelines

- Follow **PEP 8** for all Python files
- Use **type hints** for all function signatures
- Write **docstrings** for all classes and public methods
- Use the **structured logger** from `src/utils/logger.py`
  — never use bare `print()` in `src/` modules
- All evaluation code must use **AUPRC and F1-Score only**
  — never introduce Accuracy as a metric
- All hyperparameters must be defined in **YAML config files**
  — never hardcode values in source files

### What Not to Change

- Do not modify the `scale_pos_weight` default without
  documenting the new imbalance ratio calculation
- Do not add Accuracy as an evaluation metric under any
  circumstances — it is architecturally excluded by design
- Do not commit files from `data/`, `models/`, or `logs/`
  — these are in `.gitignore` by design

---

## License

MIT License — see `LICENSE` for details.

---

## Citation
```bibtex
@thesis{darachhat2025federated,
  title     = {A Federated Learning Framework for Early Fraud
               Detection using XGBoost: Balancing Model
               Performance and Data Privacy},
  author    = {Sothun Darachhat},
  year      = {2025},
  school    = {Royal University of Phnom Penh},
  type      = {Bachelor's Thesis},
  supervisor = {Chhim Bunchhun}
}
```