# MMRec-FAIR

Fairness in Multimodal Recommender Systems.

This repository contains the **core pipeline code** used in our RecSys 2026 work:
data preprocessing, dataset annotation, running MMRec models (multimodal + ID-only),
and computing accuracy + multi-class / multi-group fairness metrics.

We deliberately **omit** paper-specific table builders and heavy analysis scripts here,
and only keep the code you need to reproduce model training and evaluation.

---

## Repository structure

All core logic lives in the `pipeline/` directory:

- `pipeline/config.yaml`  
  Global configuration: dataset names, paths, and which models/datasets to run.
  This is the single source of truth for the pipeline.

- `pipeline/requirements.txt`  
  Minimal Python dependencies for the pipeline (e.g. `pandas`, `numpy`, `PyYAML`).

### Preprocessing and annotations

- `pipeline/data_preprocess.py`  
  Converts raw datasets into the internal interaction format used by MMRec and the
  fairness evaluation. Produces preprocessed data under `pipeline_output/` (paths
  are controlled via `config.yaml`).

- `pipeline/annotate.py`  
  Builds **data annotations** for fairness:
  - `relevance.csv`: multi-class relevance labels (Low / Medium / High) per interaction
  - `item_groups.csv`: item popularity tiers (head / mid / tail)
  - `user_groups.csv`: user activity tiers (low / mid / high)  
  These are written under `pipeline_output/data_annotations/<dataset>/` and are
  used by `evaluate.py` to compute fairness metrics.

### Running MMRec models

- `pipeline/run_mmrec_baselines.py`  
  Main experiment driver for **MMRec** models (multimodal and ID-only):

  - Ensures MMRec data exists (under the external MMRec repo, configured via `config.yaml`)
  - Runs `annotate` so that `data_annotations/` is in sync with MMRec’s user/item IDs
  - For each `(dataset, model)`:
    - optionally trains MMRec via its `main.py`
    - collects top‑K recommendation CSVs
    - calls `evaluate.py` to compute metrics

  Results are written to:
  - `pipeline_output/baselines/<dataset>_<model>/runs_metrics.csv` (per‑run accuracy + item fairness)
  - `pipeline_output/user_fairness/<dataset>_<model>/runs_metrics.csv` (per‑run user fairness)

  Useful flags (run with `python3 pipeline/run_mmrec_baselines.py --help`):

  - `--baseline` : run **ID-only** baseline models (e.g. `*_IDOnly`)
  - `--all`      : run both multimodal and ID-only variants (except BPR)
  - `--evaluate-only` : skip MMRec training, evaluate existing top‑K CSVs only
  - `--resume`   : in evaluate‑only mode, skip run_ids already present in `runs_metrics.csv`
  - `--binary-only` : compute only binary fairness metrics (optional)
  - `--fill-user-fairness` : backfill **user_fairness** runs from existing baselines
    without duplicating baseline results (used to complete ID-only user fairness).

### Evaluation and fairness metrics

- `pipeline/evaluate.py`  
  Core evaluation entrypoint. Given top‑K recommendation CSVs and the annotations
  from `annotate.py`, it computes:

  - Traditional ranking metrics: Recall@K, Precision@K, NDCG@K, MAP@K
  - Item-side fairness:
    - Exposure per popularity tier (head / mid / tail)
    - Exposure gap and variance
    - Group-conditioned NDCG@K by popularity tier
  - Relevance-class fairness:
    - Group-conditioned NDCG@K by relevance class (Low / Medium / High)
  - User-side fairness (activity tiers):
    - NDCG@K for low / mid / high activity users
    - Binary user fairness: low vs. (mid + high)

  It is used both stand‑alone and from `run_mmrec_baselines.py`.

- `pipeline/fairness_metrics.py`  
  Implements the actual fairness metric computations, including:

  - Exposure per group and derived statistics (max–min gap, variance)
  - Group-conditioned nDCG@K for:
    - item popularity groups (head / mid / tail)
    - relevance classes (Low / Medium / High)
    - user activity tiers (low / mid / high)
  - Binary views when needed (e.g., tail vs. head, niche vs. mainstream users).

- `pipeline/fairness.py`  
  Light wrapper / helper for fairness evaluation that orchestrates
  calling the lower-level metrics in `fairness_metrics.py`.

---

## Installation

Use Python 3.9+ (we developed with 3.9). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows PowerShell

pip install -r pipeline/requirements.txt
