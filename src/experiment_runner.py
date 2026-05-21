from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from .train import run_train
except ImportError:
    from train import run_train


ROOT = Path(__file__).resolve().parents[1]


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """실험 결과 CSV의 공통 metric 컬럼을 보장한다."""
    for col in ["val_micro_f1", "val_auprc", "test_micro_f1", "test_auprc"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def run_experiment_configs(configs: list[str], output_csv: str) -> pd.DataFrame:
    os.chdir(ROOT)
    rows = [run_train(config) for config in configs]
    new_df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        if "experiment_name" in df.columns:
            df = df.drop_duplicates(subset=["experiment_name"], keep="last")
    else:
        df = new_df
    df = normalize_metric_columns(df)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return new_df


def parse_args(default_configs: list[str], default_output: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=default_configs)
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv", default=default_output)
    return parser.parse_args()
