from __future__ import annotations

from pathlib import Path

try:
    from .experiment_runner import parse_args, run_experiment_configs
except ImportError:
    from experiment_runner import parse_args, run_experiment_configs


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = [
    str(ROOT / "configs/architecture_cls.yaml"),
    str(ROOT / "configs/architecture_entity_start.yaml"),
    str(ROOT / "configs/architecture_entity_start_end.yaml"),
]


def main():
    args = parse_args(DEFAULT_CONFIGS, str(ROOT / "outputs/architecture_experiments.csv"))
    print(run_experiment_configs(args.configs, args.output_csv))


if __name__ == "__main__":
    main()
