"""
Run MMRec (multimodal_ml1m_dbbook_lfm2k-main/4_mmrec) models, then evaluate with our multi-class fairness.

Uses the paper's data (4_mmrec/data from 3_data_processing): same user/item IDs as data_annotations.
Flow:
  1) Ensure 4_mmrec/data/<dataset> has .inter (and .npy for multimodal). Run 3_data_processing if needed.
  2) Run annotate so data_annotations/ exist from repo .inter (relevance, item_groups).
  3) For each (dataset, model): run MMRec main.py → find recommend_topk/*.csv → run evaluate (fairness).
Results go to pipeline_output/baselines/<dataset>_<model>/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml


# Map mmrec dataset name to 3_data_processing folder name
_MMREC_PROCESSING_FOLDERS = {
    "movielens_1m": "ml-1m_grouplens",
    "lfm2k": "hetrec2011-lastfm-2k",
    "dbbook": "dbbook",
}


def _sync_mmrec_data_from_processing(base: Path, config: dict, log) -> None:
    """Copy .inter and .npy from 3_data_processing to 4_mmrec/data so MMRec can run without manual copy."""
    import shutil
    repo = base / config.get("repo_dir", "multimodal_ml1m_dbbook_lfm2k-main")
    mmrec_data = repo / "4_mmrec" / "data"
    proc = repo / "3_data_processing"
    for ds, proc_folder in _MMREC_PROCESSING_FOLDERS.items():
        if ds not in config.get("mmrec_datasets", []):
            continue
        target_dir = mmrec_data / ds
        inter_file = config.get("datasets", {}).get(ds, {}).get("inter_file", f"{ds}.inter")
        if (target_dir / inter_file).exists() or (target_dir / f"{ds}.inter").exists():
            continue
        proc_processed = proc / proc_folder / "processed_data"
        proc_npy = proc / proc_folder / "multimodal_features" / "mmrec_npy"
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in (proc_processed / f"{ds}.inter", proc_processed / inter_file):
            if f.exists():
                shutil.copy2(f, target_dir / f.name)
                log(f"Synced {f.name} -> 4_mmrec/data/{ds}/")
        for npy in (proc_npy.glob("*.npy") if proc_npy.exists() else []):
            shutil.copy2(npy, target_dir / npy.name)
            log(f"Synced {npy.name} -> 4_mmrec/data/{ds}/")


def _ensure_mmrec_data(base: Path, config: dict, log) -> bool:
    """Check 4_mmrec/data has .inter for each mmrec_dataset; optionally sync from 3_data_processing."""
    repo = base / config.get("repo_dir", "multimodal_ml1m_dbbook_lfm2k-main")
    mmrec_data = repo / "4_mmrec" / "data"
    datasets = config.get("mmrec_datasets", ["movielens_1m", "lfm2k", "dbbook"])
    _sync_mmrec_data_from_processing(base, config, log)
    ok = True
    for ds in datasets:
        inter_file = config.get("datasets", {}).get(ds, {}).get("inter_file", f"{ds}.inter")
        inter_path = mmrec_data / ds / inter_file
        if not inter_path.exists():
            inter_path = mmrec_data / ds / f"{ds}.inter"
        if not inter_path.exists():
            log(f"MMRec data missing: {mmrec_data / ds} (need .inter from 3_data_processing). Run notebooks in 3_data_processing first.")
            ok = False
    return ok


# Failed combos that still lack evaluation results (run only these with --failed).
# As of the latest aggregate, only LightGCN has no baseline/evaluation outputs.
FAILED_COMBOS = [
    ("movielens_1m", "LightGCN"),
    ("lfm2k", "LightGCN"),
    ("dbbook", "LightGCN"),
]


def main():
    parser = argparse.ArgumentParser(description="Run MMRec models then fairness evaluation.")
    parser.add_argument("--baseline", action="store_true", help="Use ID-only baseline models (mmrec_baseline_models) instead of multimodal mmrec_models.")
    parser.add_argument("--all", action="store_true", help="Run all 33 combos (no BPR): both mmrec_models and mmrec_baseline_models.")
    parser.add_argument("--failed", action="store_true", help="Run only the previously failed (dataset, model) combos listed in FAILED_COMBOS.")
    parser.add_argument("--binary", action="store_true", help="Include binary fairness metrics (default). Use --no-binary to disable.")
    parser.add_argument("--no-binary", action="store_true", help="Disable binary fairness metrics (multi-class/multi-group only).")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip MMRec training; only run evaluation on existing top-K CSVs.")
    parser.add_argument("--binary-only", action="store_true", help="Run only binary fairness (no multi-class/multi-group). Implies evaluate-only; writes only to binary_fairness/.")
    parser.add_argument("--resume", action="store_true", help="With --evaluate-only: only evaluate top-K runs whose run_id is not already in runs_metrics.csv; append new rows. Ignored when training.")
    parser.add_argument("--fill-user-fairness", action="store_true", help="For ID-only models: run evaluation only for run_ids that are in baselines but not in user_fairness; write only to user_fairness/ (no baseline duplicates). Implies --evaluate-only --baseline.")
    args = parser.parse_args()
    include_binary = not args.no_binary
    if args.binary_only:
        args.evaluate_only = True
    if args.fill_user_fairness:
        args.evaluate_only = True
        args.baseline = True

    base = Path(__file__).resolve().parent.parent
    with open(PIPELINE_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    out_root = base / config.get("output_root", "pipeline_output")
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    from run_logging import create_run_logger
    logger = create_run_logger(base)
    log = logger.log

    log("=== MMRec baselines + multi-class fairness pipeline ===")
    log(f"Base: {base}")
    if args.evaluate_only:
        log("Mode: --evaluate-only (skip MMRec training; evaluate existing top-K CSVs only).")
    if args.resume and args.evaluate_only:
        log("Mode: --resume (skip run_ids already in runs_metrics.csv; append only new runs).")
    if args.binary_only:
        log("Mode: --binary-only (compute only binary fairness; write only to binary_fairness/).")
    if args.fill_user_fairness:
        log("Mode: --fill-user-fairness (evaluate only run_ids in baselines but not in user_fairness; write only to user_fairness/).")
    log(f"Binary fairness metrics: {'enabled' if include_binary else 'disabled (--no-binary)'}.")

    # 1) Annotate from repo data (so fairness uses same user/item IDs as MMRec)
    log("Step 1: annotate (from 4_mmrec/data / 3_data_processing)")
    try:
        from annotate import run_annotate
        run_annotate(base=base, config=config)
        log("Annotate done.")
    except Exception as e:
        log(f"Annotate failed: {e}")
        if "4_mmrec" in str(e) or "File" in str(e):
            log("Ensure 4_mmrec/data/<dataset> has .inter files (run 3_data_processing notebooks, then copy to 4_mmrec/data).")
        logger.close()
        sys.exit(1)

    # 2) Check MMRec data exists (skip if evaluate-only)
    if not args.evaluate_only and not _ensure_mmrec_data(base, config, log):
        log("Fix MMRec data and re-run.")
        logger.close()
        sys.exit(1)

    # 3) Ensure MMRec deps (lmdb, etc.) so subprocess can import — skip if evaluate-only
    mmrec_src = base / config["mmrec_src_dir"]
    if not args.evaluate_only:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "import lmdb"],
                cwd=mmrec_src,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                log("MMRec needs 'lmdb'. Installing with pip...")
                subprocess.run([sys.executable, "-m", "pip", "install", "lmdb"], capture_output=False, timeout=120)
                proc2 = subprocess.run([sys.executable, "-c", "import lmdb"], cwd=mmrec_src, capture_output=True, text=True, timeout=10)
                if proc2.returncode != 0:
                    log("After pip install lmdb, import still failed. Install MMRec deps: pip install -r multimodal_ml1m_dbbook_lfm2k-main/4_mmrec/requirements.txt")
                    logger.close()
                    sys.exit(1)
        except Exception as e:
            log(f"MMRec env check failed: {e}. Install: pip install lmdb (or 4_mmrec/requirements.txt)")
            logger.close()
            sys.exit(1)
    rec_topk_dir = base / config["mmrec_recommend_topk_dir"]
    results_fairness_dir = base / config.get("results_fairness_dir", "pipeline_output/baselines")

    if args.failed:
        combos_to_run = FAILED_COMBOS
        log(f"Using --failed: {len(combos_to_run)} combos.")
    elif args.all:
        models_mm = config.get("mmrec_models", ["BPR", "VBPR", "LATTICE", "MMGCN", "SLMRec", "FREEDOM", "LightGCN"])
        models_id = config.get("mmrec_baseline_models", ["VBPR_IDOnly", "FREEDOM_IDOnly", "LATTICE_IDOnly", "MMGCN_IDOnly", "SLMRec_IDOnly"])
        models = [m for m in dict.fromkeys(models_mm + models_id) if m != "BPR"]
        datasets = config.get("mmrec_datasets", ["movielens_1m", "lfm2k", "dbbook"])
        combos_to_run = [(ds, m) for ds in datasets for m in models]
        log(f"Using --all (no BPR): {len(combos_to_run)} combos.")
    elif args.baseline:
        models = config.get("mmrec_baseline_models", ["VBPR_IDOnly", "FREEDOM_IDOnly", "LATTICE_IDOnly", "MMGCN_IDOnly", "SLMRec_IDOnly"])
        datasets = config.get("mmrec_datasets", ["movielens_1m", "lfm2k", "dbbook"])
        combos_to_run = [(ds, m) for ds in datasets for m in models]
        log("Using ID-only baseline models (mmrec_baseline_models); no modality.")
    else:
        models = [m for m in config.get("mmrec_models", ["BPR", "VBPR", "LATTICE", "MMGCN", "SLMRec", "FREEDOM", "LightGCN"]) if m != "BPR"]
        datasets = config.get("mmrec_datasets", ["movielens_1m", "lfm2k", "dbbook"])
        combos_to_run = [(ds, m) for ds in datasets for m in models]
        log("Using multimodal models (mmrec_models, no BPR).")
    rec_topk_dir.mkdir(parents=True, exist_ok=True)

    failed_combos = []
    for dataset, model in combos_to_run:
            log(f"--- {model} @ {dataset} ---")
            pattern = f"{model}-{dataset}-*.csv"
            csvs_before = set(rec_topk_dir.glob(pattern))

            if args.evaluate_only:
                csvs_to_eval = sorted(csvs_before)
                if not csvs_to_eval:
                    log(f"  No top-K CSVs {pattern}; skip.")
                    failed_combos.append((dataset, model, "no_topk_csv"))
                    continue
                # Resume: skip run_ids already present in output
                if args.resume and not args.fill_user_fairness:
                    import pandas as pd
                    if args.binary_only:
                        runs_path = base / config.get("binary_fairness_dir", "pipeline_output/binary_fairness") / f"{dataset}_{model}" / "runs_metrics.csv"
                    else:
                        runs_path = results_fairness_dir / f"{dataset}_{model}" / "runs_metrics.csv"
                    existing_run_ids = set()
                    if runs_path.exists():
                        try:
                            pd.read_csv(runs_path, usecols=["run_id"], nrows=0)
                        except (ValueError, KeyError):
                            pass
                        else:
                            df = pd.read_csv(runs_path, usecols=["run_id"])
                            existing_run_ids = set(df["run_id"].astype(str).dropna())
                    prefix_resume = f"{model}-{dataset}-"
                    all_csvs = list(csvs_to_eval)
                    csvs_to_eval = [
                        p for p in all_csvs
                        if (p.stem[len(prefix_resume):] if p.stem.startswith(prefix_resume) else p.stem) not in existing_run_ids
                    ]
                    skipped = len(all_csvs) - len(csvs_to_eval)
                    if skipped:
                        log(f"  Resume: {skipped} run(s) already in {runs_path.name}; {len(csvs_to_eval)} to evaluate.")
                    if not csvs_to_eval:
                        log(f"  All runs already present; skip.")
                        continue
                # Fill user_fairness only: run only for run_ids in baselines but not in user_fairness
                if args.fill_user_fairness:
                    import pandas as pd
                    baseline_runs = results_fairness_dir / f"{dataset}_{model}" / "runs_metrics.csv"
                    user_fairness_dir = base / config.get("user_fairness_dir", "pipeline_output/user_fairness")
                    user_runs = user_fairness_dir / f"{dataset}_{model}" / "runs_metrics.csv"
                    baseline_run_ids = set()
                    if baseline_runs.exists():
                        try:
                            df = pd.read_csv(baseline_runs, usecols=["run_id"])
                            baseline_run_ids = set(df["run_id"].astype(str).dropna())
                        except (ValueError, KeyError):
                            pass
                    user_run_ids = set()
                    if user_runs.exists():
                        try:
                            df = pd.read_csv(user_runs, usecols=["run_id"])
                            user_run_ids = set(df["run_id"].astype(str).dropna())
                        except (ValueError, KeyError):
                            pass
                    to_fill = baseline_run_ids - user_run_ids
                    prefix_fill = f"{model}-{dataset}-"
                    all_csvs = list(csvs_to_eval)
                    csvs_to_eval = [
                        p for p in all_csvs
                        if (p.stem[len(prefix_fill):] if p.stem.startswith(prefix_fill) else p.stem) in to_fill
                    ]
                    if not csvs_to_eval:
                        log(f"  All baseline run_ids already in user_fairness; skip.")
                        continue
                    log(f"  Fill user_fairness: {len(csvs_to_eval)} run(s) to evaluate (already in baselines, missing in user_fairness).")
                log(f"  Evaluate-only: {len(csvs_to_eval)} existing CSV(s).")
            else:
                log(f"Step 2: MMRec main.py --model={model} --dataset={dataset}")
                mmrec_timeout_sec = 3600 * 12 if args.failed else 3600 * 8  # 12h for --failed (movielens_1m heavy)
                try:
                    proc = subprocess.run(
                        [sys.executable, "main.py", f"--model={model}", f"--dataset={dataset}"],
                        cwd=mmrec_src,
                        capture_output=False,
                        text=True,
                        timeout=mmrec_timeout_sec,
                    )
                    if proc.returncode != 0:
                        log(f"FAIL: MMRec exited {proc.returncode} for {model} @ {dataset}; skipping evaluate.")
                        failed_combos.append((dataset, model, "mmrec_exit_nonzero"))
                        continue
                except subprocess.TimeoutExpired:
                    log(f"FAIL: MMRec timed out for {model} @ {dataset}; skipping evaluate.")
                    failed_combos.append((dataset, model, "mmrec_timeout"))
                    continue
                except Exception as e:
                    log(f"FAIL: MMRec failed for {model} @ {dataset}: {e}")
                    failed_combos.append((dataset, model, str(e)))
                    continue

                csvs_after = set(rec_topk_dir.glob(pattern))
                csvs_new = sorted(csvs_after - csvs_before)
                if not csvs_new:
                    if csvs_after:
                        log(f"  No new CSV(s); {len(csvs_after)} existing. Skipping evaluate (already have runs_metrics).")
                        runs_path = results_fairness_dir / f"{dataset}_{model}" / "runs_metrics.csv"
                        if not runs_path.exists():
                            log(f"FAIL: No runs_metrics.csv at {runs_path}; run evaluate on existing top-K CSVs manually.")
                            failed_combos.append((dataset, model, "no_runs_metrics"))
                    else:
                        log(f"FAIL: No recommend_topk file {pattern} in {rec_topk_dir}; MMRec did not save any CSV.")
                        failed_combos.append((dataset, model, "no_topk_csv"))
                    continue
                csvs_to_eval = csvs_new
                log(f"  New top-K CSVs: {len(csvs_to_eval)} (total for combo: {len(csvs_after)}).")

            prefix = f"{model}-{dataset}-"
            log(f"Step 3: evaluate (fairness) for {len(csvs_to_eval)} run(s)")

            try:
                from evaluate import run_evaluate
                for rec_path in csvs_to_eval:
                    run_id = rec_path.stem[len(prefix):] if rec_path.stem.startswith(prefix) else rec_path.stem
                    log(f"  evaluate {rec_path.name} -> run_id={run_id}")
                    res = run_evaluate(
                        dataset,
                        topk_path=rec_path,
                        base=base,
                        model_name=model,
                        save_results=True,
                        run_id=run_id,
                        include_binary_attributes=include_binary,
                        binary_only=args.binary_only,
                        save_baseline_results=not args.fill_user_fairness,
                    )
                runs_path = results_fairness_dir / f"{dataset}_{model}" / "runs_metrics.csv"
                if args.binary_only:
                    runs_path = base / config.get("binary_fairness_dir", "pipeline_output/binary_fairness") / f"{dataset}_{model}" / "runs_metrics.csv"
                if args.fill_user_fairness:
                    runs_path = base / config.get("user_fairness_dir", "pipeline_output/user_fairness") / f"{dataset}_{model}" / "runs_metrics.csv"
                if not runs_path.exists():
                    log(f"FAIL: runs_metrics.csv not created at {runs_path}")
                    failed_combos.append((dataset, model, "runs_metrics_not_saved"))
                else:
                    n_lines = len(runs_path.read_text().strip().splitlines())
                    log(f"  Verified: {runs_path.name} exists ({n_lines} lines).")
                log(f"Evaluate done for {dataset}_{model}: {len(csvs_to_eval)} runs.")
            except Exception as e:
                log(f"FAIL: Evaluate failed for {dataset}_{model}: {e}")
                failed_combos.append((dataset, model, f"evaluate: {e}"))

    log("=== MMRec + fairness pipeline finished ===")
    if failed_combos:
        log(f"FAILED combos ({len(failed_combos)}):")
        for ds, mod, reason in failed_combos:
            log(f"  {ds} x {mod}: {reason}")
        logger.close()
        sys.exit(1)
    logger.close()


if __name__ == "__main__":
    main()
