from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .train import run_train
    from .utils import load_config
except ImportError:
    from train import run_train
    from utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "outputs/model_experiments.csv"

FINAL_SPLIT = "detailed_stratified"
FINAL_INPUT_STYLE = "typed_entity_marker_punct_question"
FINAL_ARCHITECTURE = "entity_start"

DEFAULT_CONFIGS = [
    ROOT / "configs/model_roberta_base.yaml",
    ROOT / "configs/model_roberta_large.yaml",
    ROOT / "configs/model_bert_base.yaml",
    ROOT / "configs/model_luke_large.yaml",
]

SOURCE_CSVS = [
    ROOT / "outputs/model_experiments.csv",
    ROOT / "outputs/architecture_experiments.csv",
    ROOT / "outputs/experiments.csv",
    ROOT / "outputs/preprocess_experiments.csv",
]


def read_result_sources() -> pd.DataFrame:
    frames = []
    source_paths = [*SOURCE_CSVS, *sorted((ROOT / "outputs/parallel").glob("*.csv"))]
    for path in source_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except pd.errors.ParserError:
                print(f"[WARN] skip malformed result csv: {path}")
                continue
            df["_source_csv"] = path.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def is_final_condition(df: pd.DataFrame) -> pd.Series:
    required = ["model", "split_strategy", "input_style", "architecture"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return (
        (df["split_strategy"] == FINAL_SPLIT)
        & (df["input_style"] == FINAL_INPUT_STYLE)
        & (df["architecture"] == FINAL_ARCHITECTURE)
    )


def normalize_model_row(row: pd.Series, config_path: Path, source: str) -> dict:
    config = load_config(config_path)
    model_name = config["model_name_or_path"]
    experiment_name = config.get("experiment_name", config_path.stem)

    normalized = row.to_dict()
    normalized.update(
        {
            "config": str(config_path),
            "experiment_group": "model",
            "experiment_name": experiment_name,
            "model": model_name,
            "input_style": FINAL_INPUT_STYLE,
            "split_strategy": FINAL_SPLIT,
            "architecture": FINAL_ARCHITECTURE,
            "output_dir": config.get("output_dir"),
            "note": f"reused from {source}",
        }
    )
    return normalized


def find_reusable_row(all_results: pd.DataFrame, config_path: Path) -> dict | None:
    if all_results.empty:
        return None

    config = load_config(config_path)
    model_name = config["model_name_or_path"]
    matched = all_results[is_final_condition(all_results) & (all_results["model"] == model_name)].copy()
    if matched.empty:
        return None

    metric_cols = ["val_micro_f1", "val_auprc", "test_micro_f1", "test_auprc"]
    for col in metric_cols:
        if col not in matched.columns:
            matched[col] = pd.NA
    matched = matched.dropna(subset=["val_micro_f1"], how="any")
    if matched.empty:
        return None

    matched["_rank_metric"] = pd.to_numeric(matched["val_micro_f1"], errors="coerce")
    best = matched.sort_values("_rank_metric", ascending=False).iloc[0]
    return normalize_model_row(best.drop(labels=["_rank_metric"], errors="ignore"), config_path, best.get("_source_csv", "existing csv"))


def merge_and_save(rows: list[dict], output_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if output_csv.exists():
        old_df = pd.read_csv(output_csv)
        df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        df = new_df

    if "experiment_name" in df.columns:
        df = df.drop_duplicates(subset=["experiment_name"], keep="last")

    ordered_cols = [
        "timestamp",
        "config",
        "experiment_group",
        "experiment_name",
        "model",
        "input_style",
        "split_strategy",
        "architecture",
        "max_length",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "per_device_eval_batch_size",
        "output_dir",
        "val_accuracy",
        "val_macro_f1",
        "val_micro_f1",
        "val_auprc",
        "test_accuracy",
        "test_macro_f1",
        "test_micro_f1",
        "test_auprc",
        "note",
        "elapsed_sec",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = pd.NA
    extra_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + extra_cols]
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def run_model_experiments(
    configs: list[Path],
    output_csv: Path = OUTPUT_CSV,
    collect_only: bool = False,
    train_base: bool = False,
) -> pd.DataFrame:
    all_results = read_result_sources()
    rows = []

    for config_path in configs:
        config = load_config(config_path)
        model_name = config["model_name_or_path"]
        experiment_name = config.get("experiment_name", config_path.stem)

        reusable = find_reusable_row(all_results, config_path)
        if reusable is not None:
            print(f"[REUSE] {experiment_name}: {model_name}")
            rows.append(reusable)
            continue

        if collect_only:
            print(f"[MISSING] {experiment_name}: no reusable result found.")
            continue

        print(f"[TRAIN] {experiment_name}: {model_name}")
        rows.append(run_train(str(config_path)))
        all_results = read_result_sources()

    if not rows:
        print("No new or reusable model experiment rows were found.")
        return pd.DataFrame()

    return merge_and_save(rows, output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=[str(path) for path in DEFAULT_CONFIGS])
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv", default=str(OUTPUT_CSV))
    parser.add_argument("--collect-only", action="store_true", help="Only collect reusable results from existing CSV files.")
    parser.add_argument("--train-base", action="store_true", help="Deprecated: base is trained automatically when no reusable row exists.")
    args = parser.parse_args()

    configs = [Path(path) for path in args.configs]
    print(run_model_experiments(configs, output_csv=Path(args.output_csv), collect_only=args.collect_only, train_base=args.train_base))


if __name__ == "__main__":
    main()
