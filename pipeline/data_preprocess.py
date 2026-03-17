"""
Data preprocessing for RecSys 2026 pipeline.
Outputs Elliot- and RecBole-compatible formats so we call those repos for evaluation
(no duplicate metric scripts). Same protocol for multimodal (repo) and normal (baseline) data.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _split_per_user(df: pd.DataFrame, user_col: str, ratios: List[float], seed: int = 42) -> Tuple[pd.DataFrame, ...]:
    """Split per user: first ratio train, second valid, third test. Returns (train, valid, test)."""
    np.random.seed(seed)
    item_col = "itemID" if "itemID" in df.columns else ("item" if "item" in df.columns else "ItemID")
    out: List[List[pd.DataFrame]] = [[] for _ in ratios]
    for _, u_df in df.groupby(user_col):
        u_df = u_df.sample(frac=1, random_state=seed)
        n = len(u_df)
        i0 = int(n * ratios[0])
        i1 = int(n * (ratios[0] + ratios[1]))
        out[0].append(u_df.iloc[:i0])
        out[1].append(u_df.iloc[i0:i1])
        out[2].append(u_df.iloc[i1:])
    train = pd.concat(out[0], ignore_index=True)
    valid = pd.concat(out[1], ignore_index=True)
    test = pd.concat(out[2], ignore_index=True)
    train_u = set(train[user_col])
    train_i = set(train[item_col])
    valid = valid[valid[user_col].isin(train_u) & valid[item_col].isin(train_i)]
    test = test[test[user_col].isin(train_u) & test[item_col].isin(train_i)]
    return train, valid, test


def _ensure_cols(df: pd.DataFrame, user_col: str = "userID", item_col: str = "itemID", rating_col: str = "rating", time_col: Optional[str] = "timestamp") -> pd.DataFrame:
    """Normalize column names to UserID, ItemID, Rating, [TimeStamp]."""
    m = {}
    for c in df.columns:
        if c in ("userID", "user_id", "user"): m[c] = "UserID"
        elif c in ("itemID", "item_id", "item", "artistID"): m[c] = "ItemID"
        elif c in ("rating", "Rating"): m[c] = "Rating"
        elif c in ("timestamp", "Timestamp", "timestamp:float"): m[c] = "Timestamp"
    df = df.rename(columns=m)
    if "Timestamp" not in df.columns:
        df["Timestamp"] = 0
    return df[["UserID", "ItemID", "Rating", "Timestamp"]]


def from_repo_inter(
    inter_path: Path,
    elliot_dir: Path,
    recbole_dir: Path,
    dataset_name: str,
) -> None:
    """
    Read repo .inter (userID, itemID, rating, [timestamp], x_label).
    Write Elliot: train.tsv, test.tsv, valid.tsv (no header).
    Write RecBole: dataset_name.inter (header + all rows).
    """
    df = pd.read_csv(inter_path, sep="\t")
    user_col = "userID" if "userID" in df.columns else "user"
    item_col = "itemID" if "itemID" in df.columns else "item"
    label_col = "x_label"
    train = df[df[label_col] == 0].drop(columns=[label_col])
    valid = df[df[label_col] == 1].drop(columns=[label_col])
    test = df[df[label_col] == 2].drop(columns=[label_col])
    for part, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        part = _ensure_cols(part)
        out_path = elliot_dir / dataset_name / f"{name}.tsv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out_path, sep="\t", index=False, header=False)
    full = _ensure_cols(df.drop(columns=[label_col]))
    recbole_path = recbole_dir / f"{dataset_name}.inter"
    recbole_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recbole_path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for _, row in full.iterrows():
            f.write(f"{int(row['UserID'])}\t{int(row['ItemID'])}\t{float(row['Rating'])}\t{int(row['Timestamp'])}\n")


def from_raw_ml1m(ratings_path: Path, elliot_dir: Path, recbole_dir: Path, dataset_name: str, core_k: int, ratios: List[float], seed: int = 42) -> None:
    """Load ML-1M ratings.dat, core-k, split, write Elliot + RecBole (normal baseline)."""
    df = pd.read_csv(ratings_path, sep="::", names=["UserID", "ItemID", "Rating", "Timestamp"], engine="python")
    df = _core_k_filter(df.rename(columns={"UserID": "user", "ItemID": "item"}), "user", "item", core_k)
    df = df.rename(columns={"user": "UserID", "item": "ItemID"}).copy()
    train, valid, test = _split_per_user(df, "UserID", ratios, seed)
    for part, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        part = part[["UserID", "ItemID", "Rating", "Timestamp"]]
        out_path = elliot_dir / dataset_name / f"{name}.tsv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out_path, sep="\t", index=False, header=False)
    full = pd.concat([train, valid, test], ignore_index=True)[["UserID", "ItemID", "Rating", "Timestamp"]]
    recbole_path = recbole_dir / f"{dataset_name}.inter"
    recbole_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recbole_path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for _, row in full.iterrows():
            f.write(f"{int(row['UserID'])}\t{int(row['ItemID'])}\t{float(row['Rating'])}\t{int(row['Timestamp'])}\n")


def from_raw_lfm2k(user_artists_path: Path, elliot_dir: Path, recbole_dir: Path, dataset_name: str, core_k: int, ratios: List[float], seed: int = 42) -> None:
    """Load Last.fm user_artists.dat (userID, artistID, weight), core-k, split. Use weight as rating for Elliot."""
    df = pd.read_csv(user_artists_path, sep="\t")
    if "artistID" in df.columns:
        df = df.rename(columns={"userID": "UserID", "artistID": "ItemID", "weight": "Rating"})
    else:
        df = df.rename(columns={"userID": "UserID", "itemID": "ItemID", "weight": "Rating"})
    df["Timestamp"] = 0
    df = _core_k_filter(df.rename(columns={"UserID": "user", "ItemID": "item"}), "user", "item", core_k)
    df = df.rename(columns={"user": "UserID", "item": "ItemID"})
    train, valid, test = _split_per_user(df, "UserID", ratios, seed)
    for part, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        part = part[["UserID", "ItemID", "Rating", "Timestamp"]]
        out_path = elliot_dir / dataset_name / f"{name}.tsv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out_path, sep="\t", index=False, header=False)
    full = pd.concat([train, valid, test], ignore_index=True)[["UserID", "ItemID", "Rating", "Timestamp"]]
    recbole_path = recbole_dir / f"{dataset_name}.inter"
    recbole_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recbole_path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for _, row in full.iterrows():
            f.write(f"{int(row['UserID'])}\t{int(row['ItemID'])}\t{float(row['Rating'])}\t{int(row['Timestamp'])}\n")


def run_preprocess(
    base: Optional[Path] = None,
    config: Optional[Dict] = None,
    source: str = "repo",
    datasets: Optional[List[str]] = None,
) -> None:
    """
    source: "repo" = use existing .inter from 4_mmrec; "normal" = use raw from 3_data_processing (baseline).
    datasets: e.g. ["movielens_1m", "lfm2k", "dbbook"] or None for all.
    """
    base = Path(base or DEFAULT_BASE)
    if config is None:
        import yaml
        with open(Path(__file__).parent / "config.yaml") as f:
            config = yaml.safe_load(f)
    elliot_dir = base / config["data_elliot_dir"]
    recbole_dir = base / config["data_recbole_dir"]
    core_k = config.get("core_k", 5)
    ratios = config.get("split_ratio", [0.8, 0.1, 0.1])
    ds = config.get("datasets", {})
    datasets = datasets or list(ds.keys())

    for name in datasets:
        if name not in ds:
            continue
        if source == "repo":
            data_path = base / config["data_root"] / name
            inter_file = ds[name]["inter_file"]
            inter_path = data_path / inter_file
            if not inter_path.exists():
                continue
            from_repo_inter(inter_path, elliot_dir, recbole_dir, name)
        elif source == "normal":
            path_key = "normal_ratings_path"
            fallback_key = "original_ratings"
            if name == "movielens_1m":
                path = ds[name].get(path_key) or ds[name].get(fallback_key)
                if path and (base / path).exists():
                    from_raw_ml1m(Path(base / path), elliot_dir, recbole_dir, name + "_normal", core_k, ratios)
            elif name == "lfm2k":
                path = ds[name].get(path_key) or ds[name].get(fallback_key)
                if path and (base / path).exists():
                    from_raw_lfm2k(Path(base / path), elliot_dir, recbole_dir, name + "_normal", core_k, ratios)
            # dbbook normal: set normal_ratings_path in config if available
    return


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["repo", "normal"], default="repo")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--base", type=Path, default=None)
    args = p.parse_args()
    run_preprocess(base=args.base, source=args.source, datasets=args.datasets)
    print("Preprocess done. Elliot: pipeline_output/data_elliot/ | RecBole: pipeline_output/data_recbole/")
