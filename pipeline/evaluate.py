"""
Load top-K recommendations (from MMRec CSV or numpy) and annotations;
compute accuracy + multi-group / multi-class fairness metrics.
Saves results to pipeline_output/results_fairness/<dataset>_<model>/ with clear names.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Support both package and script usage:
# - When run as `python -m pipeline.evaluate`, use relative import.
# - When run directly as `python pipeline/evaluate.py`, fall back to absolute.
try:  # package context (preferred)
    from .fairness_metrics import (
        accuracy_metrics,
        compute_fairness_report,
        compute_fairness_report_binary,
        exposure_per_group,
        max_min_exposure_gap,
        variance_exposure,
        ndcg_at_k,
    )
except ImportError:  # script context
    from fairness_metrics import (  # type: ignore
        accuracy_metrics,
        compute_fairness_report,
        compute_fairness_report_binary,
        exposure_per_group,
        max_min_exposure_gap,
        variance_exposure,
        ndcg_at_k,
    )

DEFAULT_BASE = Path(__file__).resolve().parent.parent


def load_topk_from_csv(csv_path: Path, user_col: str = "id", top_col_prefix: str = "top_") -> tuple:
    """MMRec format: id, top_0, top_1, ... Returns (topk, user_ids)."""
    df = pd.read_csv(csv_path, sep="\t")
    user_ids = np.asarray(df[user_col].fillna(0).values, dtype=np.float64)
    user_ids = np.nan_to_num(user_ids, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
    top_cols = [c for c in df.columns if c.startswith(top_col_prefix)]
    top_cols = sorted(top_cols, key=lambda x: int(x.split("_")[1]))
    # Fill NaN so astype(int) does not fail (e.g. dbbook FREEDOM sometimes has NaN in top-K)
    topk = np.nan_to_num(df[top_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
    return topk, user_ids


def load_topk_from_elliot_tsv(tsv_path: Path, top_k: int = 20) -> tuple:
    """Elliot format: UserID, ItemID, Score (one row per recommendation). Pivot to top-K per user.
    Elliot writes TSVs without a header (user, item, score columns only)."""
    df = pd.read_csv(tsv_path, sep="\t")
    # If first row was interpreted as header and looks numeric (Elliot no-header format), re-read with explicit names
    known_user = {"user_id", "UserId", "u"}
    known_item = {"item_id", "ItemId", "i"}
    known_score = {"score", "Score", "s"}
    has_known = (
        any(c in df.columns for c in known_user)
        and any(c in df.columns for c in known_item)
        and any(c in df.columns for c in known_score)
    )
    if not has_known and len(df.columns) == 3:
        # Headerless: columns are 0,1,2 or first data row values; re-read with no header
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["user_id", "item_id", "score"])
    # Resolve column names
    user_col = "user_id" if "user_id" in df.columns else ("UserId" if "UserId" in df.columns else "u")
    item_col = "item_id" if "item_id" in df.columns else ("ItemId" if "ItemId" in df.columns else "i")
    score_col = "score" if "score" in df.columns else ("Score" if "Score" in df.columns else "s")
    if user_col not in df.columns or item_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"Elliot TSV must have user/item/score columns. Found: {list(df.columns)}")
    df = df.sort_values([user_col, score_col], ascending=[True, False])
    users = df[user_col].unique()
    topk = np.zeros((len(users), top_k), dtype=np.int64)
    for i, u in enumerate(users):
        items = df.loc[df[user_col] == u, item_col].values[:top_k]
        topk[i, : len(items)] = items
        if len(items) < top_k:
            topk[i, len(items) :] = items[-1] if len(items) else 0
    return topk, users


def load_annotations(annotations_dir: Path, dataset_name: str) -> tuple:
    """Load relevance, item_groups, and user_groups for a dataset (from data_annotations/<dataset_name>/)."""
    dataset_dir = annotations_dir / dataset_name
    rel_path = dataset_dir / "relevance.csv"
    grp_path = dataset_dir / "item_groups.csv"
    inter_path = dataset_dir / "inter.csv"
    user_groups_path = dataset_dir / "user_groups.csv"
    if not rel_path.exists() or not grp_path.exists():
        raise FileNotFoundError(f"Run annotate first: {annotations_dir}")
    relevance = pd.read_csv(rel_path)
    item_groups = pd.read_csv(grp_path)
    inter = pd.read_csv(inter_path) if inter_path.exists() else None

    item_to_group = {int(row["item"]): row["popularity_tier"] for _, row in item_groups.iterrows()}

    user_to_group = {}
    if user_groups_path.exists():
        user_groups_df = pd.read_csv(user_groups_path)
        if "user" in user_groups_df.columns and "activity_tier" in user_groups_df.columns:
            user_to_group = {int(row["user"]): row["activity_tier"] for _, row in user_groups_df.iterrows()}

    user_pos_items = {}
    user_pos_relevance = {}
    rel_lookup = relevance.set_index(["user", "item"])["relevance_class"]
    if inter is not None:
        test = inter[inter["x_label"] == 2]
        for _, row in test.iterrows():
            u, i = int(row["user"]), int(row["item"])
            user_pos_items.setdefault(u, []).append(i)
            if (row["user"], row["item"]) in rel_lookup.index:
                user_pos_relevance[(u, i)] = rel_lookup.loc[(row["user"], row["item"])]
            else:
                user_pos_relevance[(u, i)] = "High"
    for (u, i), r in rel_lookup.items():
        user_pos_relevance[(int(u), int(i))] = r

    return item_to_group, user_pos_items, user_pos_relevance, user_to_group


def _compute_user_group_ndcg(
    topk: np.ndarray,
    user_pos_items: Dict[int, List[int]],
    user_to_group: Dict[int, str],
    k: int,
) -> Dict[str, Dict[str, float]]:
    """
    Compute nDCG@k per user activity group.
    Returns mapping group -> {"ndcg": mean_ndcg, "count": n_users_in_group_with_pos}.
    """
    if not user_to_group:
        return {}
    n_users = topk.shape[0]
    max_k = min(k, topk.shape[1])
    pos_len = np.array([len(user_pos_items.get(u, [])) for u in range(n_users)])
    pos_index = np.zeros((n_users, max_k))
    for u in range(n_users):
        pos_set = set(int(i) for i in user_pos_items.get(u, []))
        for p in range(max_k):
            if int(topk[u, p]) in pos_set:
                pos_index[u, p] = 1
    ndcg_u = ndcg_at_k(pos_index, pos_len, k)
    per_group: Dict[str, Dict[str, float]] = {}
    for u in range(n_users):
        g = user_to_group.get(u)
        if g is None:
            continue
        if pos_len[u] == 0:
            continue
        stats = per_group.setdefault(g, {"sum": 0.0, "count": 0.0})
        stats["sum"] += float(ndcg_u[u])
        stats["count"] += 1.0
    for g, stats in per_group.items():
        c = stats["count"] or 1.0
        stats["ndcg"] = stats["sum"] / c
    return per_group


def _binary_activity_groups(user_to_group: Dict[int, str]) -> Dict[int, str]:
    """
    Collapse activity_tier into a binary view for user-side fairness:
      - low  -> low
      - mid/high/other -> high
    """
    if not user_to_group:
        return {}
    out: Dict[int, str] = {}
    for u, g in user_to_group.items():
        out[u] = "low" if g == "low" else "high"
    return out


def evaluate(
    topk: np.ndarray,
    user_pos_items: Dict[int, List[int]],
    user_pos_relevance: Dict[tuple, str],
    item_to_group: Dict[int, str],
    k_list: List[int],
    include_binary_attributes: bool = True,
    user_to_group: Optional[Dict[int, str]] = None,
) -> Dict:
    """Compute accuracy and fairness for given top-K and annotations.
    If include_binary_attributes is True (default), also compute binary-view metrics for RQ1 (head/tail, relevant/not_relevant).
    If user_to_group is provided, also compute user-side fairness (activity_tier)."""
    results = {}
    results["accuracy"] = accuracy_metrics(topk, user_pos_items, k_list)
    for k in k_list:
        report = compute_fairness_report(
            topk, user_pos_items, user_pos_relevance, item_to_group, k
        )
        results[f"fairness_k{k}"] = {
            "exposure_per_group": report["exposure_per_group"],
            "max_min_exposure_gap": report["max_min_exposure_gap"],
            "variance_exposure": report["variance_exposure"],
            "group_conditioned_ndcg": report["group_conditioned_ndcg"],
        }
        if include_binary_attributes:
            binary_report = compute_fairness_report_binary(
                topk, user_pos_items, user_pos_relevance, item_to_group, k
            )
            results[f"fairness_k{k}_binary"] = {
                "exposure_per_group": binary_report["exposure_per_group"],
                "max_min_exposure_gap": binary_report["max_min_exposure_gap"],
                "variance_exposure": binary_report["variance_exposure"],
                "gini_exposure": binary_report.get("gini_exposure"),
                "avg_pairwise_exposure_gap": binary_report.get("avg_pairwise_exposure_gap"),
                "group_conditioned_ndcg": binary_report["group_conditioned_ndcg"],
            }
        if user_to_group:
            # Multi-group user-side fairness (activity_tier: low/mid/high)
            ug = _compute_user_group_ndcg(topk, user_pos_items, user_to_group, k)
            if ug:
                ndcg_per_group = {g: stats["ndcg"] for g, stats in ug.items()}
                count_per_group = {g: stats["count"] for g, stats in ug.items()}
                results[f"user_fairness_k{k}"] = {
                    "ndcg_per_group": ndcg_per_group,
                    "count_per_group": count_per_group,
                }
            # Binary user-side fairness: low vs (mid+high)
            binary_user_groups = _binary_activity_groups(user_to_group)
            if binary_user_groups:
                ug_bin = _compute_user_group_ndcg(topk, user_pos_items, binary_user_groups, k)
                if ug_bin:
                    ndcg_per_group_bin = {g: stats["ndcg"] for g, stats in ug_bin.items()}
                    count_per_group_bin = {g: stats["count"] for g, stats in ug_bin.items()}
                    results[f"user_fairness_k{k}_binary"] = {
                        "ndcg_per_group": ndcg_per_group_bin,
                        "count_per_group": count_per_group_bin,
                    }
    return results


def print_all_metrics(results: Dict, dataset_name: str = "", model_name: str = "") -> None:
    """Print all evaluation metrics: accuracy (Recall, Precision, NDCG) and fairness (Erasmo-style exposure, gap, variance, group nDCG)."""
    header = f"========== Evaluation metrics {dataset_name} {model_name}".strip() + " =========="
    print(header)
    # Accuracy
    acc = results.get("accuracy", {})
    if acc:
        print("  Accuracy:")
        for k, v in sorted(acc.items(), key=lambda x: (x[0].split("@")[-1], x[0])):
            print(f"    {k}: {v:.6f}")
    # Fairness (Erasmo-style: exposure, parity, group-conditioned nDCG)
    for key in sorted(results.keys()):
        if not key.startswith("fairness_k") or key == "accuracy":
            continue
        k_val = key.replace("fairness_k", "").replace("_binary", "")
        if "_binary" in key:
            k_val = k_val.replace("_", "")
        suffix = " (binary attributes)" if "_binary" in key else ""
        print(f"  Fairness @ K={k_val}{suffix} (exposure & group nDCG):")
        fair = results[key]
        exp = fair.get("exposure_per_group", {})
        for g, v in sorted(exp.items()):
            print(f"    exposure_{g}: {v:.6f}")
        gap = fair.get("max_min_exposure_gap")
        if gap is not None:
            print(f"    max_min_exposure_gap: {gap:.6f}")
        var = fair.get("variance_exposure")
        if var is not None:
            print(f"    variance_exposure: {var:.6f}")
        gndcg = fair.get("group_conditioned_ndcg", {})
        for gk, gv in sorted(gndcg.items()):
            if isinstance(gv, float):
                print(f"    {gk}: {gv:.6f}")
            else:
                print(f"    {gk}: {gv}")
    print("=" * len(header))


def run_evaluate(
    dataset_name: str,
    topk_path: Optional[Path] = None,
    topk_array: Optional[np.ndarray] = None,
    base: Optional[Path] = None,
    config: Optional[Dict] = None,
    k_list: Optional[List[int]] = None,
    model_name: Optional[str] = None,
    save_results: bool = True,
    run_id: Optional[str] = None,
    include_binary_attributes: bool = True,
    binary_only: bool = False,
    save_baseline_results: bool = True,
) -> Dict:
    """
    Run fairness evaluation.
    Either provide topk_path (CSV from MMRec/Elliot) or topk_array (n_users, max_k).
    If save_results and model_name: saves to results_fairness/<dataset>_<model>/.
    When run_id is set, appends one row to runs_metrics.csv (all runs in one table per folder).
    When run_id is None, overwrites report.json, accuracy.csv, metrics_<dataset>_<model>.csv (single-run legacy).
    binary_only: if True, compute only binary fairness and write only to binary_fairness/ (do not touch baselines/).
    save_baseline_results: if False, do not write to baselines/ or binary_fairness/ (only user_fairness/). Use when filling user_fairness from existing baseline runs.
    """
    base = Path(base or DEFAULT_BASE)
    if config is None:
        import yaml
        with open(Path(__file__).parent / "config.yaml") as f:
            config = yaml.safe_load(f)
    ann_dir = base / config["data_annotations_dir"]
    k_list = k_list or config.get("topk_list", [5, 10, 20])

    item_to_group, user_pos_items, user_pos_relevance, user_to_group = load_annotations(ann_dir, dataset_name)

    if topk_path is not None:
        df_peek = pd.read_csv(topk_path, sep="\t", nrows=0)
        if any(c.startswith("top_") for c in df_peek.columns):
            topk, user_ids = load_topk_from_csv(topk_path)
        else:
            top_k = max(k_list) if k_list else 20
            topk, user_ids = load_topk_from_elliot_tsv(topk_path, top_k=top_k)
        if not model_name and hasattr(topk_path, "name"):
            model_name = topk_path.stem.split("-")[0] if "-" in topk_path.stem else topk_path.stem
    elif topk_array is not None:
        topk = topk_array
        user_ids = np.arange(topk.shape[0])
        model_name = model_name or "model"
    else:
        raise ValueError("Provide topk_path or topk_array")

    # Reindex so row u = recs for user id u (fairness code assumes topk[u] = user u)
    max_uid = max(user_pos_items.keys()) if user_pos_items else (int(np.max(user_ids)) if len(user_ids) else 0)
    user_ids = np.asarray(user_ids).ravel()
    topk_reindexed = np.zeros((max_uid + 1, topk.shape[1]), dtype=topk.dtype)
    for i, u in enumerate(user_ids):
        u = int(u)
        if u <= max_uid:
            topk_reindexed[u] = topk[i]

    # When binary_only, we compute only accuracy + binary fairness and skip multi-class/multi-group item-side fairness.
    if binary_only:
        results = {"accuracy": accuracy_metrics(topk_reindexed, user_pos_items, k_list)}
        for k in k_list:
            binary_report = compute_fairness_report_binary(
                topk_reindexed, user_pos_items, user_pos_relevance, item_to_group, k
            )
            results[f"fairness_k{k}_binary"] = {
                "exposure_per_group": binary_report["exposure_per_group"],
                "max_min_exposure_gap": binary_report["max_min_exposure_gap"],
                "variance_exposure": binary_report["variance_exposure"],
                "gini_exposure": binary_report.get("gini_exposure"),
                "avg_pairwise_exposure_gap": binary_report.get("avg_pairwise_exposure_gap"),
                "group_conditioned_ndcg": binary_report["group_conditioned_ndcg"],
            }
    else:
        results = evaluate(
            topk_reindexed,
            user_pos_items,
            user_pos_relevance,
            item_to_group,
            k_list,
            include_binary_attributes=include_binary_attributes,
            user_to_group=user_to_group,
        )

    # Print all metrics (accuracy + Erasmo-style fairness) to stdout
    print_all_metrics(results, dataset_name=dataset_name, model_name=model_name or "model")

    if save_results and model_name:
        def _flatten(d: dict, prefix: str = "") -> dict:
            out = {}
            for k, v in d.items():
                key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict) and v:
                    out.update(_flatten(v, key))
                elif not isinstance(v, (dict, list)):
                    out[key] = v
            return out
        flat = {"dataset": dataset_name, "model": model_name}
        for k, v in results.get("accuracy", {}).items():
            flat[k] = v
        for key, val in results.items():
            if key == "accuracy":
                continue
            flat.update(_flatten({key: val}, ""))

        if binary_only:
            binary_dir = base / config.get("binary_fairness_dir", "pipeline_output/binary_fairness")
            binary_out = binary_dir / f"{dataset_name}_{model_name}"
            binary_out.mkdir(parents=True, exist_ok=True)
            binary_flat = {"dataset": dataset_name, "model": model_name}
            if run_id is not None:
                binary_flat["run_id"] = run_id
            for k, v in flat.items():
                if k in ("dataset", "model", "run_id"):
                    continue
                if "_binary" in k or k.startswith("Recall") or k.startswith("Precision") or k.startswith("NDCG") or k.startswith("MAP"):
                    binary_flat[k] = v
            binary_path = binary_out / "runs_metrics.csv"
            row_df = pd.DataFrame([binary_flat])
            if binary_path.exists() and run_id is not None:
                row_df.to_csv(binary_path, mode="a", header=False, index=False)
            else:
                row_df.to_csv(binary_path, index=False)
            print("Binary-only results written to", binary_out)

        # Write full results (including user_fairness_*) to baselines/ and user_fairness/
        if not binary_only:
            out_dir = base / config["results_fairness_dir"] / f"{dataset_name}_{model_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if run_id is not None:
                # Multi-run: append one row to runs_metrics.csv (all runs per folder)
                flat["run_id"] = run_id
                runs_path = out_dir / "runs_metrics.csv"
                row_df = pd.DataFrame([flat])
                if save_baseline_results:
                    if runs_path.exists():
                        row_df.to_csv(runs_path, mode="a", header=False, index=False)
                    else:
                        row_df.to_csv(runs_path, index=False)
                    print("Fairness results appended to", out_dir, "| run_id:", run_id)
            else:
                # Single-run (legacy): overwrite report.json, accuracy.csv, metrics_*.csv
                if save_baseline_results:
                    def _to_serializable(obj):
                        if isinstance(obj, dict):
                            return {k: _to_serializable(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return [_to_serializable(x) for x in obj]
                        if hasattr(obj, "item"):
                            return obj.item()
                        return obj
                    report_path = out_dir / "report.json"
                    with open(report_path, "w") as f:
                        json.dump(_to_serializable(results), f, indent=2)
                    accuracy_path = out_dir / "accuracy.csv"
                    pd.DataFrame([results["accuracy"]]).to_csv(accuracy_path, index=False)
                    metrics_csv_path = out_dir / f"metrics_{dataset_name}_{model_name}.csv"
                    pd.DataFrame([flat]).to_csv(metrics_csv_path, index=False)
                    print("Fairness results saved to", out_dir, "| CSV:", metrics_csv_path.name)

            # Binary fairness: write binary-only columns to pipeline_output/binary_fairness/
            if save_baseline_results and include_binary_attributes and any("_binary" in k for k in flat):
                binary_dir = base / config.get("binary_fairness_dir", "pipeline_output/binary_fairness")
                binary_out = binary_dir / f"{dataset_name}_{model_name}"
                binary_out.mkdir(parents=True, exist_ok=True)
                binary_flat = {"dataset": dataset_name, "model": model_name}
                if run_id is not None:
                    binary_flat["run_id"] = run_id
                for k, v in flat.items():
                    if "_binary" in k:
                        binary_flat[k] = v
                binary_path = binary_out / "runs_metrics.csv"
                row_df = pd.DataFrame([binary_flat])
                if binary_path.exists() and run_id is not None:
                    row_df.to_csv(binary_path, mode="a", header=False, index=False)
                else:
                    row_df.to_csv(binary_path, index=False)
                print("Binary fairness results written to", binary_out)

            # User fairness: write user_fairness_* columns to pipeline_output/user_fairness/
            user_cols = [k for k in flat.keys() if k.startswith("user_fairness_k")]
            if user_cols:
                user_dir = base / config.get("user_fairness_dir", "pipeline_output/user_fairness")
                user_out = user_dir / f"{dataset_name}_{model_name}"
                user_out.mkdir(parents=True, exist_ok=True)
                user_flat = {"dataset": dataset_name, "model": model_name}
                if run_id is not None:
                    user_flat["run_id"] = run_id
                for k in user_cols:
                    user_flat[k] = flat[k]
                user_path = user_out / "runs_metrics.csv"
                row_df = pd.DataFrame([user_flat])
                if user_path.exists() and run_id is not None:
                    row_df.to_csv(user_path, mode="a", header=False, index=False)
                else:
                    row_df.to_csv(user_path, index=False)
                print("User fairness results written to", user_out)

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="movielens_1m")
    p.add_argument("--topk_csv", type=Path, default=None)
    p.add_argument("--base", type=Path, default=None)
    args = p.parse_args()
    if not args.topk_csv:
        print("Pass --topk_csv (path to MMRec/Elliot top-K file) after running Elliot or MMRec.")
        exit(1)
    run_evaluate(args.dataset, topk_path=args.topk_csv, base=args.base)
