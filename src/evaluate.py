from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

try:
    from .data_loader import load_as_pandas
    from .dataset_split import make_train_val_test
    from .preprocess import preprocess_dataframe
    from .train import tokenize_dataset
    from .utils import compute_metrics, klue_re_auprc as _klue_re_auprc, klue_re_micro_f1 as _klue_re_micro_f1
    from .utils import load_config, save_json, softmax_np
except ImportError:
    from data_loader import load_as_pandas
    from dataset_split import make_train_val_test
    from preprocess import preprocess_dataframe
    from train import tokenize_dataset
    from utils import compute_metrics, klue_re_auprc as _klue_re_auprc, klue_re_micro_f1 as _klue_re_micro_f1
    from utils import load_config, save_json, softmax_np


def klue_re_micro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """KLUE-RE official micro-F1, excluding no_relation, on a 0~100 scale."""
    return _klue_re_micro_f1(preds, labels)


def klue_re_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    """KLUE-RE official AUPRC over all classes, on a 0~100 scale."""
    return _klue_re_auprc(probs, labels)


def run_evaluate(config_path: str, split: str = "val", model_dir: str | None = None) -> dict:
    config = load_config(config_path)
    model_dir = model_dir or config["output_dir"]
    hf_train_df, hf_test_df, label_names, _, _ = load_as_pandas()
    _, val_df, test_df = make_train_val_test(
        hf_train_df,
        hf_test_df,
        val_ratio=config.get("train_val_split_ratio", 0.1),
        seed=config.get("seed", 42),
        strategy=config.get("split_strategy", "label_stratified"),
    )
    target_df = val_df if split == "val" else test_df
    target_df = preprocess_dataframe(target_df, config.get("input_style", "basic_marker"))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dataset = tokenize_dataset(target_df, tokenizer, config.get("max_length", 256))
    trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    output = trainer.predict(dataset)
    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=-1)

    labels = target_df["label"].to_numpy()
    base_metrics = compute_metrics((logits, labels))
    metrics = {f"{split}_{key}": float(value) for key, value in base_metrics.items()}

    save_dir = Path(model_dir) / f"evaluation_{split}"
    save_json(metrics, save_dir / "evaluation.json")
    report = classification_report(target_df["label"], preds, target_names=label_names, zero_division=0)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--model_dir", default=None)
    args = parser.parse_args()
    print(run_evaluate(args.config, args.split, args.model_dir))


if __name__ == "__main__":
    main()
