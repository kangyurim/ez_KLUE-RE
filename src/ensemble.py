from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

try:
    from .data_loader import load_as_pandas
    from .dataset_split import make_train_val_test
    from .preprocess import preprocess_dataframe
    from .train import tokenize_dataset
    from .utils import compute_metrics_from_probs, load_config, save_json, softmax_np
except ImportError:
    from data_loader import load_as_pandas
    from dataset_split import make_train_val_test
    from preprocess import preprocess_dataframe
    from train import tokenize_dataset
    from utils import compute_metrics_from_probs, load_config, save_json, softmax_np


def predict_probs(model_dir: str, df: pd.DataFrame, max_length: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dataset = tokenize_dataset(df, tokenizer, max_length)
    trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer))
    logits = trainer.predict(dataset).predictions
    return softmax_np(logits)


def run_ensemble(config_path: str, model_dirs: list[str], split: str = "val", output_dir: str = "outputs/ensemble"):
    config = load_config(config_path)
    hf_train_df, hf_test_df, label_names, _, _ = load_as_pandas()
    _, val_df, test_df = make_train_val_test(
        hf_train_df,
        hf_test_df,
        val_ratio=config.get("train_val_split_ratio", 0.1),
        seed=config.get("seed", 42),
        strategy=config.get("split_strategy", "label_stratified"),
    )
    raw_df = val_df if split == "val" else test_df
    df = preprocess_dataframe(raw_df, config.get("input_style", "basic_marker"))
    probs = np.mean([predict_probs(model_dir, df, config.get("max_length", 256)) for model_dir in model_dirs], axis=0)
    preds = np.argmax(probs, axis=1)
    metrics = compute_metrics_from_probs(probs, raw_df["label"].to_numpy())
    if split == "val":
        metrics.update({"val_micro_f1": metrics["micro_f1"], "val_auprc": metrics["auprc"]})
    else:
        metrics.update({"test_micro_f1": metrics["micro_f1"], "test_auprc": metrics["auprc"]})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out / f"ensemble_{split}_metrics.json")
    result = raw_df.copy()
    result["pred_label"] = preds
    result["pred_label_name"] = result["pred_label"].map(lambda x: label_names[int(x)])
    result.to_csv(out / f"ensemble_{split}_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", default="outputs/ensemble")
    args = parser.parse_args()
    print(run_ensemble(args.config, args.model_dirs, args.split, args.output_dir))


if __name__ == "__main__":
    main()
