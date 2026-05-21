from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "performance_review"

RESULT_FILES = {
    "split": ROOT / "outputs" / "split_experiments.csv",
    "preprocess": ROOT / "outputs" / "preprocess_experiments.csv",
    "architecture": ROOT / "outputs" / "architecture_experiments.csv",
    "model": ROOT / "outputs" / "model_experiments.csv",
}

METRIC_COLS = [
    "val_accuracy",
    "val_macro_f1",
    "val_micro_f1",
    "val_auprc",
    "test_accuracy",
    "test_macro_f1",
    "test_micro_f1",
    "test_auprc",
]


def read_result_file(path: Path, group: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "experiment_group" not in df.columns:
        df["experiment_group"] = group
    return df


def collect_results() -> pd.DataFrame:
    frames = [read_result_file(path, group) for group, path in RESULT_FILES.items()]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()

    results = pd.concat(frames, ignore_index=True, sort=False)
    for col in METRIC_COLS:
        if col not in results.columns:
            results[col] = pd.NA
        results[col] = pd.to_numeric(results[col], errors="coerce")
    return results


def summarize_best_by_group(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    rows = []
    for group, group_df in results.groupby("experiment_group", dropna=False):
        ranked = group_df.dropna(subset=["val_micro_f1"]).sort_values("val_micro_f1", ascending=False)
        if ranked.empty:
            continue
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows).sort_values("val_micro_f1", ascending=False)


def summarize_generalization_gap(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    df = results.copy()
    df["micro_f1_gap"] = df["val_micro_f1"] - df["test_micro_f1"]
    df["auprc_gap"] = df["val_auprc"] - df["test_auprc"]
    keep_cols = [
        "experiment_group",
        "experiment_name",
        "model",
        "input_style",
        "split_strategy",
        "architecture",
        "val_micro_f1",
        "test_micro_f1",
        "micro_f1_gap",
        "val_auprc",
        "test_auprc",
        "auprc_gap",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep_cols].sort_values("micro_f1_gap", ascending=False, na_position="last")


def summarize_model_efficiency(results: pd.DataFrame) -> pd.DataFrame:
    model_df = results[results.get("experiment_group") == "model"].copy()
    if model_df.empty:
        return pd.DataFrame()

    model_df["elapsed_min"] = pd.to_numeric(model_df.get("elapsed_sec"), errors="coerce") / 60
    model_df["efficiency"] = model_df["test_micro_f1"] / model_df["elapsed_min"]
    keep_cols = [
        "experiment_name",
        "model",
        "val_micro_f1",
        "test_micro_f1",
        "test_auprc",
        "elapsed_min",
        "efficiency",
    ]
    return model_df[keep_cols].sort_values("test_micro_f1", ascending=False, na_position="last")


def load_optional_prediction_analysis() -> dict[str, pd.DataFrame]:
    """Read detailed prediction analysis files if they already exist."""
    candidates = sorted((ROOT / "outputs").glob("**/error_analysis_test"))
    if not candidates:
        candidates = sorted((ROOT / "outputs").glob("**/error_analysis_val"))

    if not candidates:
        return {}

    analysis_dir = candidates[-1]
    outputs = {}
    label_scores = analysis_dir / "label_scores.csv"
    predictions = list(analysis_dir.glob("*_predictions.csv"))
    wrong_predictions = analysis_dir / "wrong_predictions.csv"

    if label_scores.exists():
        outputs["relation_scores"] = pd.read_csv(label_scores)
    if predictions:
        pred_df = pd.read_csv(predictions[0])
        outputs["wrong_by_type_pair"] = summarize_type_pair_errors(pred_df)
    if wrong_predictions.exists():
        wrong_df = pd.read_csv(wrong_predictions)
        outputs["high_confidence_errors"] = wrong_df.sort_values("confidence", ascending=False).head(30)

    return outputs


def summarize_type_pair_errors(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    for col in ["subject_entity", "object_entity"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "subject_type" not in df.columns and "subject_entity" in df.columns:
        df["subject_type"] = df["subject_entity"].str.extract(r"'type': '([^']+)'|\"type\": \"([^\"]+)\"").bfill(axis=1).iloc[:, 0]
    if "object_type" not in df.columns and "object_entity" in df.columns:
        df["object_type"] = df["object_entity"].str.extract(r"'type': '([^']+)'|\"type\": \"([^\"]+)\"").bfill(axis=1).iloc[:, 0]

    if "subject_type" not in df.columns or "object_type" not in df.columns:
        return pd.DataFrame()

    df["type_pair"] = df["subject_type"].astype(str) + "-" + df["object_type"].astype(str)
    df["correct"] = df["label"] == df["pred_label"]
    return (
        df.groupby("type_pair")
        .agg(total=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
        .sort_values(["accuracy", "total"], ascending=[True, False])
    )


def build_improvement_plan(best_by_group: pd.DataFrame, gap_summary: pd.DataFrame, detailed_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    has_relation_scores = "relation_scores" in detailed_outputs
    has_type_pair = "wrong_by_type_pair" in detailed_outputs
    has_errors = "high_confidence_errors" in detailed_outputs

    rows = [
        {
            "review_item": "val-test gap",
            "finding": "Check experiments with large val_micro_f1 - test_micro_f1 gap.",
            "next_action": "Use more robust validation, k-fold, or ensemble if gap is large.",
            "priority": "high",
        },
        {
            "review_item": "model selection",
            "finding": "Choose final PLM by test_micro_f1, test_auprc, and elapsed time.",
            "next_action": "Use the best model as the base for loss and hyperparameter experiments.",
            "priority": "high",
        },
        {
            "review_item": "relation-level F1",
            "finding": "Detailed relation scores are available." if has_relation_scores else "Prediction-level output is not available yet.",
            "next_action": "Run error_analysis.py for the final model, then inspect low-F1 relations.",
            "priority": "medium",
        },
        {
            "review_item": "entity type pair",
            "finding": "Type-pair error summary is available." if has_type_pair else "Type-pair error summary needs prediction output.",
            "next_action": "If a type pair is weak, consider type-pair candidate filtering.",
            "priority": "medium",
        },
        {
            "review_item": "high-confidence errors",
            "finding": "High-confidence wrong examples are available." if has_errors else "High-confidence wrong examples need prediction output.",
            "next_action": "Consider focal loss, label smoothing, calibration, or hard-example augmentation.",
            "priority": "medium",
        },
    ]
    return pd.DataFrame(rows)


def run_performance_review(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results()
    detailed_outputs = load_optional_prediction_analysis()

    best_by_group = summarize_best_by_group(results)
    gap_summary = summarize_generalization_gap(results)
    model_efficiency = summarize_model_efficiency(results)
    improvement_plan = build_improvement_plan(best_by_group, gap_summary, detailed_outputs)

    results.to_csv(output_dir / "all_experiment_results.csv", index=False, encoding="utf-8-sig")
    best_by_group.to_csv(output_dir / "best_by_group.csv", index=False, encoding="utf-8-sig")
    gap_summary.to_csv(output_dir / "generalization_gap.csv", index=False, encoding="utf-8-sig")
    model_efficiency.to_csv(output_dir / "model_efficiency.csv", index=False, encoding="utf-8-sig")
    improvement_plan.to_csv(output_dir / "improvement_plan.csv", index=False, encoding="utf-8-sig")

    for name, df in detailed_outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run_performance_review(Path(args.output_dir))


if __name__ == "__main__":
    main()
