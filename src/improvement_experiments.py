from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from .error_analysis import run_error_analysis
    from .train import run_train
    from .utils import klue_re_auprc, klue_re_micro_f1, load_config, softmax_np
except ImportError:
    from error_analysis import run_error_analysis
    from train import run_train
    from utils import klue_re_auprc, klue_re_micro_f1, load_config, softmax_np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_roberta_large.yaml"
DEFAULT_WEIGHTED_CONFIG = ROOT / "configs" / "improvement_weighted_ce_roberta_large.yaml"
DEFAULT_MODEL_DIR = ROOT / "outputs" / "architecture_entity_start"
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "improvement_experiments.csv"
DETAIL_DIR = ROOT / "outputs" / "improvement_details"


def read_existing(output_csv: Path) -> pd.DataFrame:
    if output_csv.exists():
        return pd.read_csv(output_csv)
    return pd.DataFrame()


def has_result(output_csv: Path, experiment_name: str) -> bool:
    df = read_existing(output_csv)
    return "experiment_name" in df.columns and experiment_name in set(df["experiment_name"].astype(str))


def append_or_replace(output_csv: Path, row: dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if output_csv.exists():
        old_df = pd.read_csv(output_csv)
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["experiment_name"], keep="last")
    else:
        df = new_df
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def load_probs_and_labels(config_path: Path, model_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    analysis_dir = model_dir / f"error_analysis_{split}"
    probs_path = analysis_dir / f"{split}_probs.npy"
    labels_path = analysis_dir / f"{split}_labels.npy"

    if not probs_path.exists() or not labels_path.exists():
        run_error_analysis(str(config_path), split=split, model_dir=str(model_dir))

    if probs_path.exists() and labels_path.exists():
        return np.load(probs_path), np.load(labels_path)

    prob_csv = analysis_dir / f"{split}_probabilities.csv"
    pred_csv = analysis_dir / f"{split}_predictions.csv"
    if prob_csv.exists() and pred_csv.exists():
        probs = pd.read_csv(prob_csv).to_numpy(dtype=float)
        labels = pd.read_csv(pred_csv)["label"].to_numpy(dtype=int)
        np.save(probs_path, probs)
        np.save(labels_path, labels)
        return probs, labels

    raise FileNotFoundError(f"Prediction probabilities are missing: {analysis_dir}")


def metrics_from_preds(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> dict:
    num_labels = probs.shape[1]
    return {
        "accuracy": float(accuracy_score(labels, preds) * 100.0),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0) * 100.0),
        "micro_f1": klue_re_micro_f1(preds, labels, num_labels=num_labels),
        "auprc": klue_re_auprc(probs, labels, num_labels=num_labels),
    }


def threshold_predict(probs: np.ndarray, threshold: float, no_relation_id: int = 0) -> np.ndarray:
    relation_probs = np.delete(probs, no_relation_id, axis=1)
    relation_ids = [idx for idx in range(probs.shape[1]) if idx != no_relation_id]
    best_relation_pos = np.argmax(relation_probs, axis=1)
    best_relation_score = relation_probs[np.arange(len(probs)), best_relation_pos]
    best_relation_id = np.asarray(relation_ids)[best_relation_pos]
    return np.where(best_relation_score >= threshold, best_relation_id, no_relation_id)


def run_threshold_experiment(config_path: Path, model_dir: Path, output_csv: Path, force: bool = False) -> dict:
    experiment_name = "improvement_threshold_tuning"
    detail_path = DETAIL_DIR / "threshold_grid.csv"
    if not force and has_result(output_csv, experiment_name) and detail_path.exists():
        print(f"[SKIP] {experiment_name} already exists in {output_csv}")
        return read_existing(output_csv).query("experiment_name == @experiment_name").iloc[-1].to_dict()

    config = load_config(config_path)
    val_probs, val_labels = load_probs_and_labels(config_path, model_dir, "val")
    test_probs, test_labels = load_probs_and_labels(config_path, model_dir, "test")

    candidates = np.round(np.arange(0.50, 0.901, 0.05), 2)
    best_threshold = 0.5
    best_val_metrics = None
    detail_rows = []
    for threshold in candidates:
        val_preds = threshold_predict(val_probs, threshold)
        val_metrics = metrics_from_preds(val_preds, val_labels, val_probs)
        test_preds_for_threshold = threshold_predict(test_probs, threshold)
        test_metrics_for_threshold = metrics_from_preds(test_preds_for_threshold, test_labels, test_probs)
        detail_rows.append(
            {
                "threshold": float(threshold),
                "val_micro_f1": val_metrics["micro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "test_micro_f1": test_metrics_for_threshold["micro_f1"],
                "test_macro_f1": test_metrics_for_threshold["macro_f1"],
                "test_accuracy": test_metrics_for_threshold["accuracy"],
                "test_auprc": test_metrics_for_threshold["auprc"],
            }
        )
        if best_val_metrics is None or val_metrics["micro_f1"] > best_val_metrics["micro_f1"]:
            best_threshold = float(threshold)
            best_val_metrics = val_metrics

    test_preds = threshold_predict(test_probs, best_threshold)
    test_metrics = metrics_from_preds(test_preds, test_labels, test_probs)

    row = build_row(
        config=config,
        experiment_name=experiment_name,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        note=f"Best relation threshold on val: {best_threshold}",
    )
    row["threshold"] = best_threshold
    append_or_replace(output_csv, row)

    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    pred_dir = ROOT / "outputs" / "improvement_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "threshold_test_preds.npy", test_preds)
    return row


def calibrate_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return softmax_np(logits / temperature)


def nll_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    selected = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def run_calibration_experiment(config_path: Path, model_dir: Path, output_csv: Path, force: bool = False) -> dict:
    experiment_name = "improvement_temperature_calibration"
    detail_path = DETAIL_DIR / "calibration_grid.csv"
    if not force and has_result(output_csv, experiment_name) and detail_path.exists():
        print(f"[SKIP] {experiment_name} already exists in {output_csv}")
        return read_existing(output_csv).query("experiment_name == @experiment_name").iloc[-1].to_dict()

    config = load_config(config_path)
    val_probs, val_labels = load_probs_and_labels(config_path, model_dir, "val")
    test_probs, test_labels = load_probs_and_labels(config_path, model_dir, "test")

    candidates = np.round(np.arange(1.0, 3.01, 0.2), 2)
    detail_rows = []
    scores = []
    for temp in candidates:
        temperature = float(temp)
        calibrated_val_probs_for_temp = calibrate_probs(val_probs, temperature)
        calibrated_test_probs_for_temp = calibrate_probs(test_probs, temperature)
        val_nll = nll_loss(calibrated_val_probs_for_temp, val_labels)
        val_preds_for_temp = np.argmax(calibrated_val_probs_for_temp, axis=1)
        test_preds_for_temp = np.argmax(calibrated_test_probs_for_temp, axis=1)
        val_metrics_for_temp = metrics_from_preds(val_preds_for_temp, val_labels, calibrated_val_probs_for_temp)
        test_metrics_for_temp = metrics_from_preds(test_preds_for_temp, test_labels, calibrated_test_probs_for_temp)
        scores.append((temperature, val_nll))
        detail_rows.append(
            {
                "temperature": temperature,
                "val_nll": val_nll,
                "val_micro_f1": val_metrics_for_temp["micro_f1"],
                "val_macro_f1": val_metrics_for_temp["macro_f1"],
                "val_accuracy": val_metrics_for_temp["accuracy"],
                "test_micro_f1": test_metrics_for_temp["micro_f1"],
                "test_macro_f1": test_metrics_for_temp["macro_f1"],
                "test_accuracy": test_metrics_for_temp["accuracy"],
                "test_auprc": test_metrics_for_temp["auprc"],
            }
        )
    best_temperature = min(scores, key=lambda item: item[1])[0]

    calibrated_val_probs = calibrate_probs(val_probs, best_temperature)
    calibrated_test_probs = calibrate_probs(test_probs, best_temperature)
    val_preds = np.argmax(calibrated_val_probs, axis=1)
    test_preds = np.argmax(calibrated_test_probs, axis=1)
    val_metrics = metrics_from_preds(val_preds, val_labels, calibrated_val_probs)
    test_metrics = metrics_from_preds(test_preds, test_labels, calibrated_test_probs)

    row = build_row(
        config=config,
        experiment_name=experiment_name,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        note=f"Best temperature on val NLL: {best_temperature}",
    )
    row["temperature"] = best_temperature
    append_or_replace(output_csv, row)

    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    pred_dir = ROOT / "outputs" / "improvement_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "calibrated_test_probs.npy", calibrated_test_probs)
    return row


def run_weighted_ce_experiment(config_path: Path, output_csv: Path, force: bool = False) -> dict:
    experiment_name = "improvement_weighted_ce"
    if not force and has_result(output_csv, experiment_name):
        print(f"[SKIP] {experiment_name} already exists in {output_csv}")
        return read_existing(output_csv).query("experiment_name == @experiment_name").iloc[-1].to_dict()

    row = run_train(str(config_path))
    append_or_replace(output_csv, row)
    return row


def build_row(config: dict, experiment_name: str, val_metrics: dict, test_metrics: dict, note: str) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_group": "improvement",
        "experiment_name": experiment_name,
        "model": config["model_name_or_path"],
        "input_style": config.get("input_style"),
        "split_strategy": config.get("split_strategy"),
        "architecture": config.get("architecture"),
        "max_length": config.get("max_length"),
        "learning_rate": config.get("learning_rate"),
        "val_accuracy": val_metrics.get("accuracy"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "val_micro_f1": val_metrics.get("micro_f1"),
        "val_auprc": val_metrics.get("auprc"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_micro_f1": test_metrics.get("micro_f1"),
        "test_auprc": test_metrics.get("auprc"),
        "note": note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="postprocess", choices=["all", "postprocess", "threshold", "calibration", "weighted_ce"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weighted-config", default=str(DEFAULT_WEIGHTED_CONFIG))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    weighted_config_path = Path(args.weighted_config)
    model_dir = Path(args.model_dir)
    output_csv = Path(args.output_csv)

    rows = []
    if args.mode in {"all", "weighted_ce"}:
        rows.append(run_weighted_ce_experiment(weighted_config_path, output_csv, force=args.force))
    if args.mode in {"all", "postprocess", "threshold"}:
        rows.append(run_threshold_experiment(config_path, model_dir, output_csv, force=args.force))
    if args.mode in {"all", "postprocess", "calibration"}:
        rows.append(run_calibration_experiment(config_path, model_dir, output_csv, force=args.force))

    if rows:
        print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
