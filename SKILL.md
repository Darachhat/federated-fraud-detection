# SKILL.md — Reusable Project Skill Procedures

This file documents the reusable, step-by-step procedures
for the four core skill areas of this project. Each skill
is a complete, self-contained procedure that can be
followed independently to reproduce a specific aspect
of the experimental pipeline.

---

## Skill 1 — Data Preparation

**Purpose:** Transform raw PaySim CSV into Non-IID
partitioned datasets ready for federated training.

**When to use:**
- First time setting up the project
- After changing partitioning strategy
- After modifying feature engineering logic

### Prerequisites
```bash
pip install -e .
# PaySim CSV must exist at: data/raw/paysim.csv
```

### Procedure

**Step 1 — Verify raw data**
```python
import pandas as pd
df = pd.read_csv("data/raw/paysim.csv")
assert len(df) > 6_000_000, "Unexpected row count"
assert "isFraud" in df.columns, "Target column missing"
assert df["isFraud"].sum() > 8000, "Insufficient fraud cases"
print(f"Raw data verified: {len(df):,} records")
```

**Step 2 — Run preprocessing pipeline**
```bash
python scripts/prepare_data.py
```

**Step 3 — Verify outputs**
```python
from pathlib import Path
required_files = [
    "data/processed/global_test.csv",
    "data/processed/bank1_train.csv",
    "data/processed/bank1_test.csv",
    "data/processed/bank2_train.csv",
    "data/processed/bank2_test.csv",
    "data/processed/bank3_train.csv",
    "data/processed/bank3_test.csv",
]
for f in required_files:
    assert Path(f).exists(), f"Missing: {f}"
print("All partition files verified. ✓")
```

**Step 4 — Verify Non-IID property**
```python
import pandas as pd

for bank_id in ["bank1", "bank2", "bank3"]:
    df = pd.read_csv(f"data/processed/{bank_id}_train.csv")
    fraud_pct = df["isFraud"].mean() * 100
    print(f"{bank_id}: {fraud_pct:.4f}% fraud prevalence")

# Expected output:
# bank1: > 0.1%  (fraud-enriched)
# bank2: 0.0000% (zero fraud — blind spot)
# bank3: ~0.05%  (moderate fraud)
```

**Step 5 — Verify engineered features**
```python
df = pd.read_csv("data/processed/bank1_train.csv")
assert "errorBalanceOrig" in df.columns
assert "errorBalanceDest" in df.columns
assert "isFlaggedFraud" not in df.columns   # Must be dropped
assert "nameOrig" not in df.columns          # Must be dropped
print("Engineered features verified. ✓")
```

### Expected Outputs
```
data/processed/
├── global_test.csv          ~1.27M rows, 0.13% fraud
├── bank1_train.csv          Fraud-enriched (> 0.1%)
├── bank1_test.csv           Local test set
├── bank2_train.csv          Zero fraud (0.0%)
├── bank2_test.csv           Local test set
├── bank3_train.csv          Moderate fraud (~0.05%)
└── bank3_test.csv           Local test set
```

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: paysim.csv` | Raw data missing | Download from Kaggle |
| `AssertionError: isFraud` | Wrong CSV file | Verify column names match |
| `MemoryError` | Insufficient RAM | Use chunked loading or reduce dataset |
| `bank2 has fraud cases` | Partitioning bug | Check NonIIDPartitioner logic |

---

## Skill 2 — Federated Training

**Purpose:** Execute the complete federated learning
protocol across N communication rounds using the
JSON Tree Concatenation Algorithm.

**When to use:**
- Running federated experiments
- Testing a new aggregation strategy
- Reproducing thesis results

### Prerequisites
```bash
# Data preparation must be complete (Skill 1)
python scripts/prepare_data.py
python scripts/run_baseline_local.py   # Required for comparison
```

### Procedure — Simulation Mode (Recommended)

**Step 1 — Verify configuration**
```python
import yaml
with open("config/base_config.yaml") as f:
    cfg = yaml.safe_load(f)

assert cfg["project"]["seed"] == 42
assert cfg["xgboost"]["scale_pos_weight"] == 773
assert cfg["federated"]["num_rounds"] == 5
print("Configuration verified. ✓")
```

**Step 2 — Run simulation**
```bash
python scripts/run_simulation.py --rounds 5 --seed 42
```

**Step 3 — Monitor output**

Expected terminal output pattern per round:
```
══════════════════════════════════════════════════════════════
  FEDERATION ROUND 1 / 5
══════════════════════════════════════════════════════════════
[bank1] Round 1 local training...
[bank2] Round 1 local training...
[bank3] Round 1 local training...
[SERVER] Applying JSON Tree Concatenation Algorithm...
[AGGREGATOR] Extracted 100 trees from bank1_round_1.json
[AGGREGATOR] Extracted 100 trees from bank2_round_1.json
[AGGREGATOR] Extracted 100 trees from bank3_round_1.json
[AGGREGATOR] Total concatenated trees: 300

──────────────────────────────────────────────────────
  [BANK2] Round 1 Evaluation
──────────────────────────────────────────────────────
  AUPRC     : X.XXXX       ← Should be > 0.0000
  F1-Score  : X.XXXX
  [Accuracy : EXCLUDED]
──────────────────────────────────────────────────────
```

**Step 4 — Verify Bank 2 recovery**
```python
import json
from pathlib import Path

trajectory_path = Path(
    "experiments/results/federated/bank2_full_trajectory.json"
)
with open(trajectory_path) as f:
    data = json.load(f)

for r in data["trajectory"]:
    print(
        f"Round {r['round']}: "
        f"AUPRC = {r['auprc']:.4f} | "
        f"F1 = {r['f1_score']:.4f}"
    )

# Round 0 MUST be 0.0000 (blind spot baseline)
assert data["trajectory"][0]["auprc"] == 0.0
# Round 5 MUST be > 0.0 (federation resolved blind spot)
assert data["trajectory"][-1]["auprc"] > 0.0
print("Bank 2 recovery verified. ✓")
```

**Step 5 — Save results for thesis**
```python
# Results are automatically saved to:
# experiments/results/federated/round_N/
# experiments/results/federated/[bank_id]_full_trajectory.json
# experiments/results/federated/all_banks_full_trajectory.json

import json
with open(
    "experiments/results/federated/all_banks_full_trajectory.json"
) as f:
    all_results = json.load(f)

# Extract thesis table values
for bank_id in ["bank1", "bank2", "bank3"]:
    final = all_results[bank_id][-1]
    print(
        f"{bank_id} Round 5: "
        f"AUPRC = {final['auprc']:.4f} | "
        f"F1 = {final['f1_score']:.4f}"
    )
```

### Procedure — Distributed 4-Terminal Mode
```bash
# Terminal 0 (start first — blocks waiting for clients)
python scripts/run_server.py

# Terminal 1
python scripts/run_client.py --bank bank1

# Terminal 2
python scripts/run_client.py --bank bank2

# Terminal 3
python scripts/run_client.py --bank bank3
```

### Expected Round-by-Round Behavior

| Round | Bank 2 AUPRC | Expected Behavior |
|---|---|---|
| 0 (Local-Only) | 0.0000 | Total failure — no fraud labels |
| 1 | > 0.0000 | First fraud intelligence received |
| 2–4 | Increasing | Progressive improvement |
| 5 | Peak value | Maximum federated performance |

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| `TimeoutError at round N` | Client crashed or slow | Increase `round_timeout_seconds` in config |
| `KeyError: gradient_booster` | XGBoost version mismatch | Pin `xgboost==1.7.6` |
| `Bank 2 still 0.0 at Round 1` | Empty exchange directory | Check `models/exchange/` path exists |
| `Concatenated trees: 0` | No client models found | Verify client model paths in exchange dir |

---

## Skill 3 — Evaluation and Metrics

**Purpose:** Correctly evaluate any XGBoost model
using AUPRC and F1-Score. Never use Accuracy.

**When to use:**
- Evaluating any trained model
- Comparing baseline vs federated performance
- Generating thesis results tables

### The Golden Rule
```
ACCURACY IS EXCLUDED.
Under 0.13% fraud prevalence, a degenerate classifier
achieves 99.87% accuracy while detecting zero fraud.
Accuracy conveys no information about fraud detection.

USE ONLY: AUPRC and F1-Score (+ Precision, Recall as support)
```

### Standard Evaluation Procedure

**Step 1 — Load model and test data**
```python
import xgboost as xgb
import pandas as pd

model = xgb.XGBClassifier()
model.load_model("path/to/model.json")

test_df = pd.read_csv("data/processed/global_test.csv")
X_test  = test_df.drop(columns=["isFraud"])
y_test  = test_df["isFraud"]
```

**Step 2 — Generate probability scores**
```python
# Always use predict_proba — not predict
# Column 1 = probability of fraud class
y_prob = model.predict_proba(X_test)[:, 1]
```

**Step 3 — Compute AUPRC**
```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds = precision_recall_curve(
    y_test, y_prob
)
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.4f}")

# Interpretation:
# 0.0000 = Total failure (Bank 2 Local-Only)
# 0.50   = Moderate performance
# 0.90+  = Strong performance
# 1.0000 = Perfect (theoretical maximum)
```

**Step 4 — Compute F1-Score**
```python
from sklearn.metrics import f1_score

THRESHOLD = 0.5   # From base_config.yaml
y_pred    = (y_prob >= THRESHOLD).astype(int)
f1        = f1_score(y_test, y_pred, zero_division=0)
print(f"F1-Score: {f1:.4f}")
```

**Step 5 — Use the project evaluation utility**
```python
# Preferred: use the project's evaluate_model function
# which enforces the Accuracy exclusion
from src.evaluation.metrics import evaluate_model

results = evaluate_model(
    model=model,
    test_df=test_df,
    threshold=0.5,
    bank_id="bank_name",
    round_num=0
)
# Returns: auprc, f1_score, precision, recall
# Does NOT return accuracy — by design
```

### Interpreting Results
```
AUPRC Reference Points:
  0.0000 → 0.0999  : Detection failure (no useful signal)
  0.1000 → 0.4999  : Poor detection
  0.5000 → 0.7999  : Moderate detection
  0.8000 → 0.8999  : Good detection
  0.9000 → 0.9999  : Strong detection (thesis target range)
  1.0000            : Perfect (theoretical)

Thesis Baselines:
  Bank 1 Local-Only  : AUPRC = 0.9514  (strong)
  Bank 2 Local-Only  : AUPRC = 0.0000  (failure)
  Bank 3 Local-Only  : AUPRC = 0.9263  (strong)
  Centralized        : AUPRC = 0.9442  (upper bound)
```

### Thesis Results Table Template
```python
# Generate a formatted results table for the thesis
results_data = [
    ("Bank 1", "Local-Only",    0.9514, 0.9356),
    ("Bank 2", "Local-Only",    0.0000, 0.0000),
    ("Bank 3", "Local-Only",    0.9263, 0.8555),
    ("All",    "Centralized",   0.9442, 0.9170),
    # Add FL Round 5 results here after experiments
    ("Bank 2", "FL Round 5",    None,   None),
]

print(f"{'Bank':<10} {'Condition':<15} {'AUPRC':<10} {'F1-Score':<10}")
print("-" * 45)
for bank, cond, auprc, f1 in results_data:
    auprc_str = f"{auprc:.4f}" if auprc is not None else "TBD"
    f1_str    = f"{f1:.4f}"    if f1    is not None else "TBD"
    print(f"{bank:<10} {cond:<15} {auprc_str:<10} {f1_str:<10}")

print("\n[NOTE] Accuracy excluded — misleading under 0.13% imbalance")
```

---

## Skill 4 — Visualization

**Purpose:** Generate all thesis figures from
experimental results using the project's
visualization notebooks.

**When to use:**
- After completing all experiments
- When updating thesis figures with new results
- When generating figures for presentations

### Prerequisites
```bash
# All experiments must be complete
python scripts/prepare_data.py
python scripts/run_baseline_local.py
python scripts/run_baseline_central.py
python scripts/run_simulation.py --rounds 5 --seed 42
```

### Figure Generation Procedure

**Step 1 — Launch Jupyter**
```bash
jupyter notebook
# Select kernel: "Federated Fraud" (or your venv kernel)
```

**Step 2 — Run notebooks in order**
```
1. notebooks/01_data_exploration.ipynb
   → Generates: class_imbalance.png,
                accuracy_exclusion_justification.png,
                fraud_by_transaction_type.png,
                amount_distribution.png,
                temporal_analysis.png,
                engineered_features_validation.png,
                correlation_heatmap.png

2. notebooks/02_baseline_analysis.ipynb
   → Generates: partition_statistics.png,
                partition_size_comparison.png,
                baseline_performance_comparison.png,
                baseline_pr_curves.png,
                baseline_feature_importance.png,
                baseline_results_summary.csv

3. notebooks/03_federated_results_visualization.ipynb
   → Generates: auprc_trajectory.png,
                f1_trajectory.png,
                auprc_comparison_bar.png,
                precision_recall_curves.png,
                feature_importance.png,
                bank2_recovery.png,
                full_results_table.csv
```

**Step 3 — Verify all figures generated**
```python
from pathlib import Path

expected_figures = [
    "class_imbalance.png",
    "accuracy_exclusion_justification.png",
    "fraud_by_transaction_type.png",
    "amount_distribution.png",
    "temporal_analysis.png",
    "engineered_features_validation.png",
    "correlation_heatmap.png",
    "partition_statistics.png",
    "partition_size_comparison.png",
    "baseline_performance_comparison.png",
    "baseline_pr_curves.png",
    "baseline_feature_importance.png",
    "auprc_trajectory.png",
    "f1_trajectory.png",
    "auprc_comparison_bar.png",
    "precision_recall_curves.png",
    "feature_importance.png",
    "bank2_recovery.png",
]

figures_dir = Path("notebooks/figures")
missing     = []

for fig in expected_figures:
    path = figures_dir / fig
    if not path.exists():
        missing.append(fig)
    else:
        print(f"  ✓ {fig}")

if missing:
    print(f"\n  ✗ Missing: {missing}")
else:
    print(f"\n  All {len(expected_figures)} figures verified. ✓")
```

**Step 4 — Update thesis table placeholders**
```python
# After running all notebooks, extract final values:
import json

with open(
    "experiments/results/federated/all_banks_full_trajectory.json"
) as f:
    results = json.load(f)

print("=== REPLACE THESE VALUES IN YOUR THESIS ===\n")
for bank_id in ["bank1", "bank2", "bank3"]:
    final = results[bank_id][-1]
    print(
        f"[REPLACE] {bank_id} FL Round 5:\n"
        f"  AUPRC    = {final['auprc']:.4f}\n"
        f"  F1-Score = {final['f1_score']:.4f}\n"
    )
```

### Figure Naming Convention
```
Format: [content]_[type].png

Examples:
  auprc_trajectory.png          ← Line plot over rounds
  baseline_pr_curves.png        ← Precision-Recall curves
  partition_statistics.png      ← Pie charts per bank
  bank2_recovery.png            ← Bank 2 focused trajectory
  feature_importance.png        ← Horizontal bar chart
  auprc_comparison_bar.png      ← Grouped bar chart
```

### Thesis Figure Mapping

| Thesis Figure | File | Chapter |
|---|---|---|
| Figure 3.1 | `partition_statistics.png` | Ch. 3 |
| Figure 3.2 | `class_imbalance.png` | Ch. 3 |
| Figure 3.3 | `accuracy_exclusion_justification.png` | Ch. 3 |
| Figure 4.1 | `baseline_performance_comparison.png` | Ch. 4 |
| Figure 4.2 | `baseline_pr_curves.png` | Ch. 4 |
| Figure 4.3 | `auprc_trajectory.png` | Ch. 4 |
| Figure 4.4 | `f1_trajectory.png` | Ch. 4 |
| Figure 4.5 | `bank2_recovery.png` | Ch. 4 |
| Figure 4.6 | `auprc_comparison_bar.png` | Ch. 4 |
| Figure 4.7 | `feature_importance.png` | Ch. 4 |
| Figure 4.8 | `precision_recall_curves.png` | Ch. 4 |