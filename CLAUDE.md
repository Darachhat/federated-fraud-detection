# CLAUDE.md — Project Context for AI Assistants

## Project Identity

**Title:** A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy

**Author:** Sothun Darachhat
**Institution:** Royal University of Phnom Penh (សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ)
**Degree:** Bachelor of Engineering in Information Technology Engineering
**Supervisor:** Mr. Chhim Bunchhun
**Date:** June 2025
**License:** MIT

---

## What This Project Is

This is a **research thesis implementation** — a privacy-preserving Federated Learning framework for financial fraud detection. The core innovation is the **JSON Tree Concatenation Algorithm**, a novel aggregation method that enables multiple banks to collaboratively train an XGBoost fraud detection model **without sharing any raw transaction data**.

The project demonstrates that a retail bank with **zero fraud-labeled training data** (the "blind spot problem") can achieve near-centralized fraud detection performance through federated collaboration.

---

## The Problem Being Solved

Financial institutions are legally prohibited from sharing raw customer transaction data (GDPR, PDPA). This creates **Data Silos** where each bank trains fraud detection models on only its own data. Banks serving low-risk customers may have **zero confirmed fraud cases** in their entire transaction history, making supervised fraud detection models completely non-functional.

**Quantitative evidence of the problem:**
- Bank 2 (Retail, 2.27M transactions, 0 fraud labels) → AUPRC = 0.5006 (random classifier), F1 = 0.0000

---

## The Solution: JSON Tree Concatenation Algorithm

XGBoost prediction is additive: `ŷ = Σ f_k(x)` for all trees `k`. This means concatenating tree ensembles from multiple clients preserves the full discriminative capacity of each local model.

**Algorithm steps (server-side):**
1. Each client trains a local XGBoost model and serializes it to JSON
2. Server extracts tree arrays from `learner.gradient_booster.model.trees`
3. Server concatenates all tree arrays into a single unified list
4. Server updates metadata: `num_trees`, `tree_info`, `iteration_indptr`
5. Server saves and redistributes the federated global model

**No raw data is ever transferred** — only JSON model structures containing tree split points and leaf scores.

---

## Key Experimental Results

| Condition | Bank 1 AUPRC | Bank 2 AUPRC | Bank 3 AUPRC |
|---|---|---|---|
| **Local-Only** | 0.9343 | **0.5006** ⚠ | 0.9932 |
| **Centralized** ⚠ | 0.9976 | 0.9976 | 0.9976 |
| **FL Round 5** | **0.9830** | **0.9830** | **0.9830** |

- **Privacy Tax:** 0.0146 AUPRC (1.46%) — negligible
- **Bank 2 recovery:** 0.5006 → 0.9830 in **a single round** — blind spot resolved
- ⚠ Centralized is privacy-violating — evaluated as theoretical ceiling only

---

## Dataset: PaySim

- **Source:** PaySim Synthetic Financial Dataset (López-Rojas et al., 2016)
- **Size:** 6,362,620 transactions
- **Fraud prevalence:** 0.13% (8,213 fraud cases)
- **Class imbalance ratio:** 773:1
- **Fraud transaction types:** CASH-OUT and TRANSFER only
- **Train/Test split:** 80/20 stratified (test set: 1,272,524 records, 1,643 fraud)
- **Download:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **Local path:** `data/raw/paysim.csv`

---

## Non-IID Client Partitioning

| Client | Bank Profile | Transaction Types | Training Records | Fraud Records | Fraud % |
|---|---|---|---|---|---|
| **Bank 1** | High-Risk | TRANSFER, CASH-OUT | 1,064,011 | 3,077 | 0.2892% |
| **Bank 2** | Retail / Blind Spot | PAYMENT, CASH-IN | 2,272,208 | **0** | **0.0000%** |
| **Bank 3** | Mixed | All remaining | 735,859 | 2,129 | 0.2893% |

---

## Architecture and Code Structure

```
federated-fraud-detection/
│
├── config/                          # YAML configuration (all hyperparameters externalized)
│   ├── base_config.yaml             # Global: XGBoost params, federation settings, paths
│   ├── bank1_config.yaml            # Bank 1 data paths and profile
│   ├── bank2_config.yaml            # Bank 2 data paths and profile
│   └── bank3_config.yaml            # Bank 3 data paths and profile
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py          # PaySimPreprocessor — feature engineering pipeline
│   │   └── partitioner.py           # NonIIDPartitioner — 3-bank Non-IID split
│   ├── models/
│   │   └── xgboost_trainer.py       # XGBoostTrainer — local training, save/load JSON
│   ├── federated/
│   │   ├── aggregator.py            # JSONTreeConcatenator — THE CORE ALGORITHM
│   │   ├── server.py                # FederatedServer — round coordination
│   │   └── client.py                # FederatedClient — local training + evaluation
│   ├── evaluation/
│   │   └── metrics.py               # AUPRC + F1 evaluation (Accuracy excluded)
│   └── utils/
│       ├── logger.py                # Structured per-component logging (loguru)
│       ├── seed.py                  # Global seed control for reproducibility
│       └── io_utils.py              # File I/O, config loading, file polling
│
├── scripts/
│   ├── prepare_data.py              # One-time data preprocessing and partitioning
│   ├── run_baseline_local.py        # Local-Only baseline (3 isolated banks)
│   ├── run_baseline_central.py      # Centralized baseline (pooled, privacy-violated)
│   ├── run_server.py                # Launch Global Server (Terminal 0)
│   ├── run_client.py                # Launch client bank (Terminals 1-3)
│   ├── run_simulation.py            # Single-process federated simulation
│   └── generate_figures.py          # Thesis figure generation (matplotlib/seaborn)
│
├── data/raw/                        # Original PaySim CSV (not committed)
├── data/processed/                  # Preprocessed per-bank CSVs (not committed)
├── experiments/results/             # JSON results per round (not committed)
├── models/                          # Serialized XGBoost models (not committed)
├── logs/                            # Per-component structured logs (not committed)
├── notebooks/                       # Jupyter notebooks for exploration/visualization
├── documents/thesis_document.md     # Complete thesis text in Markdown
│
├── requirements.txt
├── setup.py                         # Editable install: pip install -e .
├── CLAUDE.md                        # This file
├── SKILL.md                         # Reusable procedure instructions
└── README.md
```

---

## Core Components — Detailed Reference

### 1. `JSONTreeConcatenator` (`src/federated/aggregator.py`)
The **primary technical contribution**. Methods:
- `extract_trees(model_path)` → extracts tree array from XGBoost JSON at `learner.gradient_booster.model.trees`
- `concatenate(model_paths)` → concatenates tree arrays from all clients, re-indexes tree IDs
- `build_federated_model(base_model_path, concatenated_trees, output_path)` → injects concatenated trees into a base model structure, updates `num_trees`, `tree_info`, and `iteration_indptr`
- `aggregate(client_model_paths, base_model_path, output_path)` → full pipeline entry point

### 2. `FederatedServer` (`src/federated/server.py`)
Coordinates federation rounds. Waits for all client model submissions, runs `JSONTreeConcatenator.aggregate()`, distributes global model. Uses file-based communication via `models/exchange/`.

### 3. `FederatedClient` (`src/federated/client.py`)
Represents a single bank. Loads local data, trains local XGBoost model each round, submits to exchange directory, waits for global model, evaluates on global test set. Supports warm-starting from previous global model (Rounds 2-5).

### 4. `XGBoostTrainer` (`src/models/xgboost_trainer.py`)
Wraps `xgb.XGBClassifier`. Handles `build_model()`, `train()` (with optional warm-start via `init_model_path`), `save()` (JSON serialization), and `load()`.

### 5. `PaySimPreprocessor` (`src/data/preprocessor.py`)
Pipeline: load CSV → drop identifiers (`nameOrig`, `nameDest`, `isFlaggedFraud`) → engineer `errorBalanceOrig` and `errorBalanceDest` features → one-hot encode `type` → stratified 80/20 split.

### 6. `NonIIDPartitioner` (`src/data/partitioner.py`)
Splits training data into 3 Non-IID subsets by transaction type. Bank 2's fraud labels are forced to 0 to create the blind spot condition.

### 7. `evaluate_model()` (`src/evaluation/metrics.py`)
Evaluates using **AUPRC and F1-Score only**. Accuracy is explicitly excluded with a comment explaining why. Also reports precision and recall as supporting metrics.

---

## XGBoost Hyperparameter Configuration

| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 100 | Sufficient for convergence on PaySim |
| `max_depth` | 6 | Standard for tabular financial data |
| `learning_rate` | 0.1 | Moderate shrinkage |
| `scale_pos_weight` | 773 | Matches the 773:1 class imbalance ratio |
| `eval_metric` | `aucpr` | Optimizes AUPRC during training |
| `use_label_encoder` | false | Suppresses deprecation warnings |
| `random_state` | 42 | Full reproducibility |
| `n_jobs` | -1 | Use all CPU cores |

---

## Engineered Features (Critical for Performance)

These two features are the **highest-importance features** in all trained models:

1. **`errorBalanceOrig`** = `newbalanceOrig - (oldbalanceOrg - amount)`
   - Measures discrepancy in originating account balance
   - Fraudulent transactions systematically produce non-zero values

2. **`errorBalanceDest`** = `newbalanceDest - (oldbalanceDest + amount)`
   - Measures discrepancy in destination account balance
   - Complements originating account signal for TRANSFER fraud

---

## Critical Rules: DO NOT Violate These

### 1. NEVER Use Accuracy as a Metric
Accuracy is **architecturally excluded** from this project. Under 0.13% fraud prevalence, a classifier predicting all-legitimate achieves 99.87% accuracy while detecting **zero fraud**. Every evaluation must use **AUPRC and F1-Score only**.

### 2. NEVER Introduce Raw Data Transfer in Federated Code
The entire privacy guarantee depends on the fact that **only JSON model structures** (tree splits and leaf scores) are transmitted. Never add code that transfers `DataFrame`s, feature values, or transaction records between clients or to the server.

### 3. All Hyperparameters Must Live in YAML Config Files
No hardcoded values in source files. All parameters are externalized in `config/base_config.yaml` and per-bank config files.

### 4. Never Use `print()` in `src/` Modules for Production Logging
Use the structured logger from `src/utils/logger.py` (loguru-based). The existing `print()` calls in the codebase are legacy debug outputs.

### 5. Do Not Commit Data, Models, or Logs
`data/`, `models/`, `logs/`, and `experiments/results/` are in `.gitignore`. These contain large binary/CSV files and per-run artifacts.

---

## How to Run Experiments

### Prerequisites
```bash
pip install -e .     # Editable install (enables src.* imports)
# Place PaySim CSV at: data/raw/paysim.csv
```

### Step 1 — Data Preparation (run once)
```bash
python scripts/prepare_data.py
```

### Step 2 — Baselines
```bash
python scripts/run_baseline_local.py     # Local-Only (demonstrates blind spot)
python scripts/run_baseline_central.py   # Centralized (privacy-violated ceiling)
```

### Step 3 — Federated Learning
```bash
# Option A: Single-process simulation (recommended)
python scripts/run_simulation.py --rounds 5 --seed 42

# Option B: Distributed 4-terminal mode
# Terminal 0: python scripts/run_server.py
# Terminal 1: python scripts/run_client.py --bank bank1
# Terminal 2: python scripts/run_client.py --bank bank2
# Terminal 3: python scripts/run_client.py --bank bank3
```

---

## Communication Architecture

```
Federated Round N:

  Bank 1 ──► trains local XGBoost ──► saves JSON ──► models/exchange/bank1_round_N.json
  Bank 2 ──► trains local XGBoost ──► saves JSON ──► models/exchange/bank2_round_N.json
  Bank 3 ──► trains local XGBoost ──► saves JSON ──► models/exchange/bank3_round_N.json
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │   Global Server  │
                                                    │   Concatenates   │
                                                    │   tree arrays    │
                                                    └────────┬─────────┘
                                                             │
                                                             ▼
                                               models/global/global_model_round_N.json
                                                             │
                                        ┌────────────────────┼────────────────────┐
                                        ▼                    ▼                    ▼
                                     Bank 1              Bank 2              Bank 3
                                   (Evaluate)          (Evaluate)          (Evaluate)
                                   AUPRC + F1          AUPRC + F1          AUPRC + F1

  Privacy Guarantee: NO raw transaction data ever leaves any bank.
  Only JSON tree structures (split features, thresholds, leaf scores) are transmitted.
```

---

## Federated Training Protocol

- **5 communication rounds** (convergence achieved in Round 1)
- **Round 1:** Each client trains from scratch, submits local model
- **Rounds 2-5:** Each client warm-starts from previous global model
- **File-based communication:** Shared `models/exchange/` directory
- **Timeout:** 300 seconds per client per round (configurable)
- **Server cleans up exchange files** after each round's aggregation

---

## Key Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| XGBoost over neural networks | Superior on structured tabular data; JSON serialization enables tree concatenation |
| AUPRC + F1 only | Accuracy gives 99.87% for a zero-detection classifier — completely misleading |
| JSON Tree Concatenation | Non-invasive: operates at serialization layer, no XGBoost training modification needed |
| `scale_pos_weight = 773` | Matches exact negative-to-positive ratio to address 773:1 class imbalance |
| Non-IID partitioning | Reflects realistic heterogeneity across bank risk profiles |
| Bank 2 zero fraud labels | Deliberately creates the blind spot condition to demonstrate FL's value |
| File-based communication | Simple, auditable, no network stack required for simulation |
| Stratified train/test split | Preserves 0.13% fraud prevalence in both partitions |
| Global test set shared across all evaluations | Ensures full comparability |
| `errorBalanceOrig` / `errorBalanceDest` features | Exploit PaySim's fraud encoding mechanism; highest-importance features |

---

## Dependencies

```
xgboost>=1.7.0       # Core ML algorithm (JSON serialization support required)
scikit-learn>=1.2.0   # Train/test split, evaluation metrics
numpy>=1.23.0         # Numerical operations
pandas>=1.5.0         # Data loading and manipulation
pyyaml>=6.0           # YAML configuration loading
matplotlib>=3.6.0     # Figure generation
seaborn>=0.12.0       # Statistical visualization
loguru>=0.6.0         # Structured logging
tqdm>=4.64.0          # Progress bars
joblib>=1.2.0         # Parallel utilities
jupyter>=1.0.0        # Notebooks
ipykernel>=6.0.0      # Jupyter kernel
python-dotenv>=0.21.0 # Environment variable management
```

**Python version:** 3.9+
**Platform:** Developed and tested on Windows

---

## Thesis Document

The complete thesis text (all 6 chapters, front matter, references) is available at:
`documents/thesis_document.md`

### Chapter Structure:
1. **Introduction** — Background, problem statement, aims, rationale, limitations
2. **Literature Review** — Rule-based systems, ML fraud detection, XGBoost, Data Silos, FL foundations, Non-IID challenges, SecureBoost/FedTree comparison, PaySim
3. **Methodology** — Research design, dataset, feature engineering, Non-IID partitioning, JSON Tree Concatenation Algorithm, federated protocol, baselines, metrics
4. **Results** — Local-Only, Centralized, Federated results with full metric tables
5. **Discussion** — Blind spot validation, algorithm effectiveness, privacy tax, comparison with prior work, limitations
6. **Conclusion** — 6 formal conclusions, 7 future work directions

---

## Common Tasks for AI Assistants

### "Run the full experiment"
```bash
python scripts/prepare_data.py
python scripts/run_baseline_local.py
python scripts/run_baseline_central.py
python scripts/run_simulation.py --rounds 5 --seed 42
```

### "Generate thesis figures"
```bash
python scripts/generate_figures.py
```

### "Add a new evaluation metric"
Edit `src/evaluation/metrics.py` — but **never add Accuracy**.

### "Change XGBoost hyperparameters"
Edit `config/base_config.yaml` under the `xgboost:` section. Never hardcode in source.

### "Add a new client bank"
1. Create `config/bank4_config.yaml`
2. Update `NonIIDPartitioner.partition()` in `src/data/partitioner.py`
3. Update `num_clients` in `config/base_config.yaml`
4. Update `FederatedServer.run()` default `bank_ids` list

### "Understand the JSON tree structure"
XGBoost JSON model path: `learner → gradient_booster → model → trees` (array of tree dicts). Each tree has `id`, node splits, leaf values. The `gbtree_model_param.num_trees` must equal `len(trees)` and `len(tree_info)`.
