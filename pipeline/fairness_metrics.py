"""
Multi-group and multi-class fairness metrics for top-K recommendation.
Designed for RecSys 2026 evaluation: exposure by group, group-conditioned nDCG by relevance class,
variance/min-max, fairness-accuracy trade-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Position discount for exposure (IR-style): 1 / log2(rank+1)
def _position_discount(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(1, k + 1) + 1)


def exposure_per_group(
    topk: np.ndarray,
    item_to_group: Dict[int, str],
    k: int,
    position_weighted: bool = True,
) -> Dict[str, float]:
    """
    Expected exposure per item group (e.g. head/mid/tail).
    topk: (n_users, max_k) item indices.
    item_to_group: item_id -> group label.
    """
    n_users = topk.shape[0]
    disc = _position_discount(k) if position_weighted else np.ones(k) / k
    group_exposure: Dict[str, List[float]] = {}
    for g in set(item_to_group.values()):
        group_exposure[g] = []
    for u in range(n_users):
        for pos in range(min(k, topk.shape[1])):
            i = topk[u, pos]
            g = item_to_group.get(int(i), "unknown")
            if g not in group_exposure:
                group_exposure[g] = []
            w = disc[pos] if position_weighted else 1.0 / k
            group_exposure[g].append(w)
    out = {}
    for g, vals in group_exposure.items():
        out[g] = np.sum(vals) / n_users if n_users else 0.0
    return out


def max_min_exposure_gap(exposure_per_group: Dict[str, float]) -> float:
    """Max group exposure - min group exposure."""
    vals = list(exposure_per_group.values())
    return float(np.max(vals) - np.min(vals)) if vals else 0.0


def variance_exposure(exposure_per_group: Dict[str, float]) -> float:
    """Variance of exposure across groups."""
    vals = np.array(list(exposure_per_group.values()))
    return float(np.var(vals)) if len(vals) > 1 else 0.0


def gini_exposure(exposure_per_group: Dict[str, float]) -> float:
    """
    Gini coefficient of exposure across groups.
    Standard definition:
      G = sum_{i,j} |x_i - x_j| / (2 * n^2 * mean(x))
    where x_i are group exposures (non-negative).
    """
    vals = np.array(list(exposure_per_group.values()), dtype=float)
    n = len(vals)
    if n == 0:
        return 0.0
    mean = vals.mean()
    if mean == 0:
        return 0.0
    diff_matrix = np.abs(vals[:, None] - vals[None, :])
    return float(diff_matrix.sum() / (2.0 * n * n * mean))


def avg_pairwise_exposure_gap(exposure_per_group: Dict[str, float]) -> float:
    """
    Average absolute pairwise exposure difference across groups.
    This is a disparity-style measure complementary to max-min gap.
    """
    vals = np.array(list(exposure_per_group.values()), dtype=float)
    n = len(vals)
    if n <= 1:
        return 0.0
    diff_matrix = np.abs(vals[:, None] - vals[None, :])
    iu, ju = np.triu_indices(n, k=1)
    return float(diff_matrix[iu, ju].mean())


def ndcg_at_k(pos_index: np.ndarray, pos_len: np.ndarray, k: int) -> np.ndarray:
    """Per-user nDCG@k. pos_index: (n_users, max_k) bool hit matrix; pos_len: (n_users,) num positives."""
    n_users = pos_index.shape[0]
    idcg_len = np.minimum(pos_len.astype(int), k)
    iranks = np.arange(1, pos_index.shape[1] + 1, dtype=np.float64)
    idcg_full = np.cumsum(1.0 / np.log2(iranks + 1))
    idcg_per_user = np.array([idcg_full[min(int(l) - 1, k - 1)] if l > 0 else 1.0 for l in idcg_len])
    ranks = np.arange(1, pos_index.shape[1] + 1, dtype=np.float64)
    dcg = np.cumsum(np.where(pos_index, 1.0 / np.log2(ranks + 1), 0), axis=1)
    dcg_at_k = dcg[:, min(k, dcg.shape[1]) - 1]
    ndcg_u = np.where(idcg_per_user > 0, dcg_at_k / idcg_per_user, 0)
    return ndcg_u


def group_conditioned_ndcg(
    topk: np.ndarray,
    user_pos_items: Dict[int, List[int]],
    user_pos_relevance: Dict[Tuple[int, int], str],
    item_to_group: Dict[int, str],
    k: int,
    group: Optional[str] = None,
    relevance_class: Optional[str] = None,
) -> Tuple[float, int]:
    """
    nDCG@k conditioned on (group, relevance_class).
    If group is set, only consider test items in that item group.
    If relevance_class is set, only consider test items with that relevance.
    Returns (mean nDCG, count of users with at least one such positive).
    """
    n_users = topk.shape[0]
    pos_index_list = []
    pos_len_list = []
    for u in range(n_users):
        pos_items = list(user_pos_items.get(u, []))
        if group is not None:
            pos_items = [i for i in pos_items if item_to_group.get(int(i), "") == group]
        if relevance_class is not None:
            pos_items = [i for i in pos_items if user_pos_relevance.get((u, int(i)), "") == relevance_class]
        if not pos_items:
            continue
        pos_set = set(pos_items)
        hits = np.zeros(k)
        for pos in range(min(k, topk.shape[1])):
            if int(topk[u, pos]) in pos_set:
                hits[pos] = 1
        pos_index_list.append(hits)
        pos_len_list.append(len(pos_items))
    if not pos_index_list:
        return 0.0, 0
    pos_index = np.array(pos_index_list)
    pos_len = np.array(pos_len_list)
    ndcg_u = ndcg_at_k(pos_index, pos_len, k)
    return float(ndcg_u.mean()), len(ndcg_u)


def _map_at_k(pos_index: np.ndarray, pos_len: np.ndarray, k: int) -> np.ndarray:
    """Per-user MAP@k. AP_u = (1/|rel_u|) * sum_{r=1..k} P(r) * rel(r); P(r)=hits_up_to_r/r."""
    n_users = pos_index.shape[0]
    map_u = np.zeros(n_users)
    for u in range(n_users):
        n_rel = int(pos_len[u])
        if n_rel == 0:
            continue
        hits = pos_index[u, :k]
        prec_at_r = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float64)
        map_u[u] = np.sum(prec_at_r * hits) / n_rel
    return map_u


def accuracy_metrics(
    topk: np.ndarray,
    user_pos_items: Dict[int, List[int]],
    k_list: List[int],
) -> Dict[str, float]:
    """Recall@K, Precision@K, nDCG@K, and MAP@K (macro over users). Aligns with Elliot + evaluation needs."""
    max_k = max(k_list)
    n_users = topk.shape[0]
    pos_len = np.array([len(user_pos_items.get(u, [])) for u in range(n_users)])
    pos_index = np.zeros((n_users, max_k))
    for u in range(n_users):
        pos_set = set(int(i) for i in user_pos_items.get(u, []))
        for p in range(min(max_k, topk.shape[1])):
            if int(topk[u, p]) in pos_set:
                pos_index[u, p] = 1
    results = {}
    for k in k_list:
        hits = np.cumsum(pos_index[:, :k], axis=1)[:, -1]
        rec = hits / np.maximum(pos_len, 1)
        results[f"Recall@{k}"] = float(rec.mean())
        results[f"Precision@{k}"] = float(hits.mean() / k)  # macro over users: (1/n)*sum(hits_u/k)
        ndcg_u = ndcg_at_k(pos_index[:, :k], pos_len, k)
        results[f"NDCG@{k}"] = float(ndcg_u.mean())
        map_u = _map_at_k(pos_index[:, :k], pos_len, k)
        results[f"MAP@{k}"] = float(map_u.mean())
    return results


def compute_fairness_report(
    topk: np.ndarray,
    user_pos_items: Dict[int, List[int]],
    user_pos_relevance: Dict[Tuple[int, int], str],
    item_to_group: Dict[int, str],
    k: int,
) -> Dict:
    """Single K: accuracy + exposure fairness + group-conditioned nDCG by relevance."""
    report = {}
    report["accuracy"] = accuracy_metrics(topk, user_pos_items, [k])
    report["exposure_per_group"] = exposure_per_group(topk, item_to_group, k)
    report["max_min_exposure_gap"] = max_min_exposure_gap(report["exposure_per_group"])
    report["variance_exposure"] = variance_exposure(report["exposure_per_group"])
    # Additional disparity / inequality style metrics (Erasmo-inspired)
    report["gini_exposure"] = gini_exposure(report["exposure_per_group"])
    report["avg_pairwise_exposure_gap"] = avg_pairwise_exposure_gap(report["exposure_per_group"])
    report["group_conditioned_ndcg"] = {}
    for rel in ["Low", "Medium", "High"]:
        ndcg_val, count = group_conditioned_ndcg(
            topk, user_pos_items, user_pos_relevance, item_to_group, k, relevance_class=rel
        )
        report["group_conditioned_ndcg"][f"nDCG@{k}_relevance_{rel}"] = ndcg_val
        report["group_conditioned_ndcg"][f"count_{rel}"] = count
    for grp in ["head", "mid", "tail"]:
        ndcg_val, count = group_conditioned_ndcg(
            topk, user_pos_items, user_pos_relevance, item_to_group, k, group=grp
        )
        report["group_conditioned_ndcg"][f"nDCG@{k}_group_{grp}"] = ndcg_val
        report["group_conditioned_ndcg"][f"count_group_{grp}"] = count
    return report


def _binary_item_group(tier: str) -> str:
    """Map multi-group popularity_tier to binary: tail vs head (head+mid -> head)."""
    if tier == "tail":
        return "tail"
    return "head"  # head, mid, or unknown


def _binary_relevance(relevance_class: str) -> str:
    """Map multi-class relevance to binary: High -> relevant, Low/Medium -> not_relevant."""
    if relevance_class == "High":
        return "relevant"
    return "not_relevant"


def compute_fairness_report_binary(
    topk: np.ndarray,
    user_pos_items: Dict[int, List[int]],
    user_pos_relevance: Dict[Tuple[int, int], str],
    item_to_group: Dict[int, str],
    k: int,
) -> Dict:
    """
    Same as compute_fairness_report but with binary attributes for RQ1 comparison.
    - Group: tail vs head (mid merged into head).
    - Relevance: relevant (High) vs not_relevant (Low+Medium).
    Returns same-shaped dict with only binary groups/classes; keys compatible with flattening.
    """
    binary_item_to_group = {i: _binary_item_group(g) for i, g in item_to_group.items()}
    binary_relevance = {
        (u, i): _binary_relevance(r) for (u, i), r in user_pos_relevance.items()
    }
    report = {}
    report["exposure_per_group"] = exposure_per_group(topk, binary_item_to_group, k)
    report["max_min_exposure_gap"] = max_min_exposure_gap(report["exposure_per_group"])
    report["variance_exposure"] = variance_exposure(report["exposure_per_group"])
    report["gini_exposure"] = gini_exposure(report["exposure_per_group"])
    report["avg_pairwise_exposure_gap"] = avg_pairwise_exposure_gap(report["exposure_per_group"])
    report["group_conditioned_ndcg"] = {}
    for rel in ["relevant", "not_relevant"]:
        ndcg_val, count = group_conditioned_ndcg(
            topk,
            user_pos_items,
            binary_relevance,
            binary_item_to_group,
            k,
            relevance_class=rel,
        )
        report["group_conditioned_ndcg"][f"nDCG@{k}_relevance_{rel}"] = ndcg_val
        report["group_conditioned_ndcg"][f"count_{rel}"] = count
    for grp in ["head", "tail"]:
        ndcg_val, count = group_conditioned_ndcg(
            topk,
            user_pos_items,
            binary_relevance,
            binary_item_to_group,
            k,
            group=grp,
        )
        report["group_conditioned_ndcg"][f"nDCG@{k}_group_{grp}"] = ndcg_val
        report["group_conditioned_ndcg"][f"count_group_{grp}"] = count
    return report
