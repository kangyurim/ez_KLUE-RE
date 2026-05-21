from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .experiment_runner import parse_args, run_experiment_configs
except ImportError:
    from experiment_runner import parse_args, run_experiment_configs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "outputs/split_experiments.csv"

DEFAULT_CONFIGS = [
    str(ROOT / "configs/split_random.yaml"),
    str(ROOT / "configs/split_stratified.yaml"),
    str(ROOT / "configs/split_detailed_stratified.yaml"),
]


def get_done_configs(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()

    df = pd.read_csv(output_csv)
    required_metrics = ["val_micro_f1", "val_auprc"]
    if not all(col in df.columns for col in required_metrics):
        return set()
    df = df.dropna(subset=required_metrics, how="any")

    done = set()
    if "config" in df.columns:
        done.update(df["config"].dropna().astype(str))
    if "config_path" in df.columns:
        done.update(df["config_path"].dropna().astype(str))
    if "experiment_name" in df.columns:
        done.update(df["experiment_name"].dropna().astype(str))
    return done


def filter_pending_configs(configs: list[str], output_csv: Path) -> list[str]:
    done = get_done_configs(output_csv)
    pending = []

    for config in configs:
        config_path = str(Path(config))
        config_abs = str(Path(config).resolve())
        config_name = Path(config).stem

        if config_path in done or config_abs in done or config_name in done:
            print(f"[SKIP] already done: {config_name}")
        else:
            pending.append(config_path)

    return pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv", default=str(OUTPUT_CSV))
    parser.add_argument("--force", action="store_true", help="Run all experiments even if previous results exist.")
    cli_args = parser.parse_args()

    output_csv = Path(cli_args.output_csv)
    pending_configs = cli_args.configs if cli_args.force else filter_pending_configs(cli_args.configs, output_csv)

    if not pending_configs:
        print("수행할 새 split 실험이 없습니다. 모든 split 실험이 이미 완료되었습니다.")
        return

    print("실행할 config 목록:")
    for config in pending_configs:
        print("-", Path(config).name)

    print(run_experiment_configs(pending_configs, str(output_csv)))


if __name__ == "__main__":
    main()
