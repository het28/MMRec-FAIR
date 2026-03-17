"""
Build relevance classes and group memberships for fairness evaluation.
Uses same core-5 + multimodal filtering logic as the repo; adds multi-class relevance
and item groups (popularity tier). Outputs sidecar files for evaluate.py.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

# Default base = RecSys 2026 folder (parent of pipeline/)
DEFAULT_BASE = Path(__file__).resolve().parent.parent


def _core_k_filter(interactions: pd.DataFrame, user_col: str, item_col: str, k: int) -> pd.DataFrame:
    while True:
        ucount = interactions[user_col].value_counts()
        icount = interactions[item_col].value_counts()
        valid_u = ucount[ucount >= k].index
        valid_i = icount[icount >= k].index
        core = interactions[interactions[user_col].isin(valid_u) & interactions[item_col].isin(valid_i)]
        if len(core) == len(interactions):
            break
        interactions = core
    return core


def _build_user_activity_groups(train: pd.DataFrame) -> pd.DataFrame:
    """
    Build user-side groups (activity_tier) based on training interaction count per user.
    Levels: low | mid | high using 33% / 67% quantiles, symmetric to popularity_tier.
    """
    if train.empty or "user" not in train.columns:
        return pd.DataFrame(columns=["user", "activity_tier"])
    user_counts = train["user"].value_counts()
    # Ensure stable ordering
    users = sorted(user_counts.index.tolist())
    counts = user_counts.reindex(users, fill_value=0)
    if len(counts) == 1:
        q33 = q67 = counts.values[0]
    else:
        q33, q67 = np.quantile(counts.values, [0.33, 0.67])

    def act_tier(c: int) -> str:
        if c <= q33:
            return "low"
        if c >= q67:
            return "high"
        return "mid"

    rows = [{"user": u, "activity_tier": act_tier(counts[u])} for u in users]
    return pd.DataFrame(rows)


def _relevance_ml1m(rating: int) -> str:
    if rating <= 2:
        return "Low"
    if rating == 3:
        return "Medium"
    return "High"


def _relevance_lfm_quantiles(weights: np.ndarray, q33: float, q67: float) -> np.ndarray:
    out = np.full(len(weights), "Medium", dtype=object)
    out[weights <= q33] = "Low"
    out[weights >= q67] = "High"
    return out


def build_annotations_ml1m(
    base: Path,
    inter_file: str,
    original_ratings: str,
    item_meta: str,
    core_k: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """MovieLens-1M: explicit ratings -> Low/Medium/High; item popularity + genre.
    Uses og_ids from 3_data_processing to align original ratings with remapped .inter."""
    data_path = base / "multimodal_ml1m_dbbook_lfm2k-main" / "4_mmrec" / "data" / "movielens_1m"
    inter_path = data_path / inter_file
    og_path = base / "multimodal_ml1m_dbbook_lfm2k-main" / "3_data_processing" / "ml-1m_grouplens" / "processed_data" / "movielens_1m_og_ids.inter"
    ratings_path = base / original_ratings
    movies_path = base / item_meta

    # Remapped .inter (userID, itemID = 0..n-1). Prefer 3_data_processing if row count matches og_ids.
    inter = pd.read_csv(inter_path, sep="\t", usecols=["userID", "itemID", "x_label"])
    inter = inter.rename(columns={"userID": "user", "itemID": "item"})
    valid_items = set(inter["item"].unique())
    valid_users = set(inter["user"].unique())

    # Original IDs: only align if og_ids has same row count (else 4_mmrec .inter was filtered)
    og = pd.read_csv(og_path, sep="\t", usecols=["userID", "itemID"])
    og = og.rename(columns={"userID": "user_orig", "itemID": "item_orig"})
    if len(og) == len(inter):
        inter["user_orig"] = og["user_orig"].values
        inter["item_orig"] = og["item_orig"].values
        ratings = pd.read_csv(ratings_path, sep="::", names=["user", "item", "rating", "timestamp"], engine="python")
        ratings = ratings[["user", "item", "rating"]].drop_duplicates()
        inter = inter.merge(ratings, left_on=["user_orig", "item_orig"], right_on=["user", "item"], how="left", suffixes=("", "_r"))
        rating_col = "rating" if "rating" in inter.columns else "rating_r"
        inter["relevance_class"] = inter[rating_col].fillna(4).astype(int).map(_relevance_ml1m)
    else:
        # Row count mismatch: use .inter rating column if present (binary) or default High
        inter_full = pd.read_csv(inter_path, sep="\t")
        if "rating" in inter_full.columns:
            inter = inter.merge(inter_full[["userID", "itemID", "rating"]].rename(columns={"userID": "user", "itemID": "item"}), on=["user", "item"], how="left")
            inter["relevance_class"] = inter["rating"].map(lambda x: "Low" if x == 0 else "High")
        else:
            inter["relevance_class"] = "High"
    relevance = inter[["user", "item", "relevance_class"]].drop_duplicates()

    # Item popularity (training only)
    train = inter[inter["x_label"] == 0][["user", "item"]]
    item_counts = train["item"].value_counts().reindex(valid_items, fill_value=0)
    q33 = np.quantile(item_counts.values, 0.33)
    q67 = np.quantile(item_counts.values, 0.67)
    def pop_tier(c):
        if c <= q33:
            return "tail"
        if c >= q67:
            return "head"
        return "mid"
    item_pop = {i: pop_tier(item_counts[i]) for i in valid_items}

    movies = pd.read_csv(movies_path, sep="::", names=["id", "name", "genres"], encoding="ISO-8859-1", engine="python")
    item_genre = movies.set_index("id")["genres"].to_dict()

    item_groups = pd.DataFrame(
        [
            {"item": i, "popularity_tier": item_pop.get(i, "mid"), "genres": item_genre.get(i, "")}
            for i in valid_items
        ]
    )

    # User activity tiers (training only)
    user_groups = _build_user_activity_groups(train[["user", "item"]])

    out = output_dir / "movielens_1m"
    os.makedirs(out, exist_ok=True)
    relevance.to_csv(out / "relevance.csv", index=False)
    item_groups.to_csv(out / "item_groups.csv", index=False)
    user_groups.to_csv(out / "user_groups.csv", index=False)
    inter[["user", "item", "x_label"]].to_csv(out / "inter.csv", index=False)
    return relevance, item_groups, inter[["user", "item", "x_label"]]


def build_annotations_lfm2k(
    base: Path,
    inter_file: str,
    original_ratings: str,
    core_k: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Last.fm-2K: weight quantiles -> Low/Medium/High; item popularity.
    Uses og_ids to align original user_artists weight with remapped .inter."""
    data_path = base / "multimodal_ml1m_dbbook_lfm2k-main" / "4_mmrec" / "data" / "lfm2k"
    inter_path = data_path / inter_file
    og_path = base / "multimodal_ml1m_dbbook_lfm2k-main" / "3_data_processing" / "hetrec2011-lastfm-2k" / "processed_data" / "lfm2k_og_ids.inter"
    ratings_path = base / original_ratings

    inter = pd.read_csv(inter_path, sep="\t", usecols=["userID", "itemID", "x_label"])
    inter = inter.rename(columns={"userID": "user", "itemID": "item"})
    valid_items = set(inter["item"].unique())

    og = pd.read_csv(og_path, sep="\t")
    if "artistID" in og.columns:
        og = og.rename(columns={"userID": "user_orig", "artistID": "item_orig"})
    else:
        og = og.rename(columns={"userID": "user_orig", "itemID": "item_orig"})
    if len(og) != len(inter):
        raise ValueError("lfm2k og_ids and .inter row count mismatch")
    inter["user_orig"] = og["user_orig"].values
    inter["item_orig"] = og["item_orig"].values

    ratings = pd.read_csv(ratings_path, sep="\t")
    if "artistID" in ratings.columns:
        ratings = ratings.rename(columns={"userID": "user_orig", "artistID": "item_orig", "weight": "weight"})
    else:
        ratings = ratings.rename(columns={"userID": "user_orig", "itemID": "item_orig"})
    ratings = ratings[["user_orig", "item_orig", "weight"]].drop_duplicates()
    inter = inter.merge(ratings, on=["user_orig", "item_orig"], how="left")
    inter["weight"] = inter["weight"].fillna(1).astype(float)
    w = inter["weight"].values
    q33, q67 = np.quantile(w, [0.33, 0.67])
    inter["relevance_class"] = _relevance_lfm_quantiles(w, q33, q67)
    relevance = inter[["user", "item", "relevance_class"]].drop_duplicates()

    train = inter[inter["x_label"] == 0][["user", "item"]]
    item_counts = train["item"].value_counts().reindex(valid_items, fill_value=0)
    q33c = np.quantile(item_counts.values, 0.33)
    q67c = np.quantile(item_counts.values, 0.67)
    def pop_tier(c):
        if c <= q33c:
            return "tail"
        if c >= q67c:
            return "head"
        return "mid"
    item_pop = {i: pop_tier(item_counts[i]) for i in valid_items}
    item_groups = pd.DataFrame([{"item": i, "popularity_tier": item_pop[i], "genres": ""} for i in valid_items])

    # User activity tiers (training only)
    user_groups = _build_user_activity_groups(train)

    out = output_dir / "lfm2k"
    os.makedirs(out, exist_ok=True)
    relevance.to_csv(out / "relevance.csv", index=False)
    item_groups.to_csv(out / "item_groups.csv", index=False)
    user_groups.to_csv(out / "user_groups.csv", index=False)
    inter[["user", "item", "x_label"]].to_csv(out / "inter.csv", index=False)
    return relevance, item_groups, inter[["user", "item", "x_label"]]


def build_annotations_dbbook(
    base: Path,
    inter_file: str,
    core_k: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DBbook: binary relevance (all High); item popularity only."""
    data_path = base / "multimodal_ml1m_dbbook_lfm2k-main" / "4_mmrec" / "data" / "dbbook"
    inter_path = data_path / inter_file

    inter = pd.read_csv(inter_path, sep="\t", usecols=["userID", "itemID", "x_label"])
    inter = inter.rename(columns={"userID": "user", "itemID": "item"})
    valid_items = set(inter["item"].unique())
    valid_users = set(inter["user"].unique())

    train = inter[inter["x_label"] == 0]
    relevance = train[["user", "item"]].copy()
    relevance["relevance_class"] = "High"
    # add val/test positives
    for xl in [1, 2]:
        r = inter[inter["x_label"] == xl][["user", "item"]]
        r = r.copy()
        r["relevance_class"] = "High"
        relevance = pd.concat([relevance, r], ignore_index=True).drop_duplicates(subset=["user", "item"])

    item_counts = train["item"].value_counts().reindex(valid_items, fill_value=0)
    q33 = np.quantile(item_counts.values, 0.33)
    q67 = np.quantile(item_counts.values, 0.67)
    def pop_tier(c):
        if c <= q33:
            return "tail"
        if c >= q67:
            return "head"
        return "mid"
    item_pop = {i: pop_tier(item_counts[i]) for i in valid_items}
    item_groups = pd.DataFrame([{"item": i, "popularity_tier": item_pop[i], "genres": ""} for i in valid_items])

    # User activity tiers (training only)
    user_groups = _build_user_activity_groups(train[["user", "item"]])

    out = output_dir / "dbbook"
    os.makedirs(out, exist_ok=True)
    relevance.to_csv(out / "relevance.csv", index=False)
    item_groups.to_csv(out / "item_groups.csv", index=False)
    user_groups.to_csv(out / "user_groups.csv", index=False)
    inter.to_csv(out / "inter.csv", index=False)
    return relevance, item_groups, inter


def run_annotate(base: Optional[Path] = None, config: Optional[Dict] = None):
    base = base or DEFAULT_BASE
    if config is None:
        import yaml
        with open(Path(__file__).parent / "config.yaml") as f:
            config = yaml.safe_load(f)
    out = Path(base) / config["data_annotations_dir"]
    core_k = config.get("core_k", 5)
    ds = config.get("datasets", {})

    if "movielens_1m" in ds:
        build_annotations_ml1m(
            base,
            ds["movielens_1m"]["inter_file"],
            ds["movielens_1m"]["original_ratings"],
            ds["movielens_1m"]["item_meta"],
            core_k,
            out,
        )
    if "lfm2k" in ds:
        build_annotations_lfm2k(
            base,
            ds["lfm2k"]["inter_file"],
            ds["lfm2k"]["original_ratings"],
            core_k,
            out,
        )
    if "dbbook" in ds:
        build_annotations_dbbook(
            base,
            ds["dbbook"]["inter_file"],
            core_k,
            out,
        )
    return out


if __name__ == "__main__":
    run_annotate()
