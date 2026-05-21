from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover
    load_safetensors = None

try:
    from .data_loader import load_as_pandas
    from .dataset_split import make_train_val_test
    from .models import build_model
    from .preprocess import add_special_tokens, preprocess_dataframe
    from .tokenization import ReDataCollator, tokenize_dataset
    from .utils import get_entity_types, load_config, softmax_np
except ImportError:
    from data_loader import load_as_pandas
    from dataset_split import make_train_val_test
    from models import build_model
    from preprocess import add_special_tokens, preprocess_dataframe
    from tokenization import ReDataCollator, tokenize_dataset
    from utils import get_entity_types, load_config, softmax_np


CUSTOM_ARCHITECTURES = {"entity_start", "entity_start_end"}


def entity_to_dict(entity) -> dict:
    if isinstance(entity, dict):
        return entity
    if hasattr(entity, "as_py"):
        return entity.as_py()
    if isinstance(entity, str):
        try:
            parsed = ast.literal_eval(entity)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return {}


def find_checkpoint_file(model_dir: Path) -> Path:
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    for path in candidates:
        if path.exists():
            return path

    checkpoint_files = sorted(model_dir.glob("checkpoint-*/*.safetensors")) + sorted(model_dir.glob("checkpoint-*/pytorch_model.bin"))
    if checkpoint_files:
        return checkpoint_files[-1]

    raise FileNotFoundError(
        f"No model checkpoint found in {model_dir}. "
        "Expected model.safetensors, pytorch_model.bin, or checkpoint-* files."
    )


def load_state_dict(checkpoint_path: Path) -> dict:
    if checkpoint_path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("safetensors is required to load model.safetensors.")
        return load_safetensors(str(checkpoint_path))
    return torch.load(checkpoint_path, map_location="cpu")


def load_analysis_model(config: dict, model_dir: Path, tokenizer, label_names, label2id, id2label):
    architecture = config.get("architecture", "cls")
    if architecture in CUSTOM_ARCHITECTURES:
        model = build_model(config, tokenizer, label_names, id2label, label2id)
        checkpoint_path = find_checkpoint_file(model_dir)
        state_dict = load_state_dict(checkpoint_path)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] missing keys while loading custom model: {len(missing)}")
        if unexpected:
            print(f"[WARN] unexpected keys while loading custom model: {len(unexpected)}")
        return model

    return AutoModelForSequenceClassification.from_pretrained(model_dir)


def add_entity_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    subjects = result["subject_entity"].map(entity_to_dict)
    objects = result["object_entity"].map(entity_to_dict)
    result["subject"] = subjects.map(lambda x: x.get("word"))
    result["object"] = objects.map(lambda x: x.get("word"))
    result["subject_type"] = subjects.map(lambda x: str(x.get("type", "")).upper())
    result["object_type"] = objects.map(lambda x: str(x.get("type", "")).upper())
    result["type_pair"] = result["subject_type"] + "-" + result["object_type"]
    return result


def save_confusion_matrix(labels, preds, label_names, save_dir: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        save_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        annot=True,
        fmt="d",
        annot_kws={"size": 6},
        linewidths=0.2,
        linecolor="white",
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def save_type_pair_scores(result_df: pd.DataFrame, save_dir: Path) -> None:
    type_pair_scores = (
        result_df.groupby("type_pair")
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["accuracy", "total"], ascending=[True, False])
    )
    type_pair_scores.to_csv(save_dir / "type_pair_scores.csv", index=False, encoding="utf-8-sig")

    label_type_pair_scores = (
        result_df.groupby(["type_pair", "gold_label_name"])
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values(["accuracy", "total"], ascending=[True, False])
    )
    label_type_pair_scores.to_csv(save_dir / "label_type_pair_scores.csv", index=False, encoding="utf-8-sig")


def run_error_analysis(config_path: str, split: str = "test", model_dir: str | None = None) -> Path:
    config = load_config(config_path)
    model_dir_path = Path(model_dir or config["output_dir"])
    if not model_dir_path.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir_path}")

    hf_train_df, hf_test_df, label_names, label2id, id2label = load_as_pandas()
    train_df, val_df, test_df = make_train_val_test(
        hf_train_df,
        hf_test_df,
        val_ratio=config.get("train_val_split_ratio", 0.1),
        seed=config.get("seed", 42),
        strategy=config.get("split_strategy", "label_stratified"),
    )

    raw_df = val_df if split == "val" else test_df
    raw_df = add_entity_columns(raw_df)
    df = preprocess_dataframe(raw_df, config.get("input_style", "basic_marker"))

    tokenizer_source = model_dir_path if (model_dir_path / "tokenizer_config.json").exists() else config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    add_special_tokens(tokenizer, config.get("input_style", "basic_marker"), get_entity_types(train_df, val_df, test_df))

    model = load_analysis_model(config, model_dir_path, tokenizer, label_names, label2id, id2label)
    architecture = config.get("architecture", "cls")
    dataset = tokenize_dataset(df, tokenizer, config.get("max_length", 256), architecture)

    # predict 단계에서는 loss 계산이 필요 없으므로 labels를 제거
    # y_true는 result_df의 label 컬럼을 사용
    if "labels" in dataset.column_names:
        dataset_for_predict = dataset.remove_columns(["labels"])
    else:
        dataset_for_predict = dataset

    data_collator = ReDataCollator(tokenizer) if architecture in CUSTOM_ARCHITECTURES else DataCollatorWithPadding(tokenizer)
    trainer = Trainer(model=model, data_collator=data_collator)

    output = trainer.predict(dataset_for_predict)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=-1)

    result_df = raw_df.copy()
    result_df["input_text"] = df["input_text"]
    result_df["pred_label"] = preds
    result_df["gold_label_name"] = result_df["label"].map(lambda x: label_names[int(x)])
    result_df["pred_label_name"] = result_df["pred_label"].map(lambda x: label_names[int(x)])
    result_df["confidence"] = probs.max(axis=1)
    result_df["correct"] = result_df["label"] == result_df["pred_label"]

    save_dir = model_dir_path / f"error_analysis_{split}"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{split}_probs.npy", probs)
    np.save(save_dir / f"{split}_labels.npy", result_df["label"].to_numpy())
    pd.DataFrame(probs, columns=label_names).to_csv(
        save_dir / f"{split}_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result_df.to_csv(save_dir / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
    result_df[~result_df["correct"]].sort_values("confidence", ascending=False).to_csv(
        save_dir / "wrong_predictions.csv", index=False, encoding="utf-8-sig"
    )

    report = classification_report(
        result_df["label"],
        result_df["pred_label"],
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label_name"})
    label_id_map = {label_name: idx for idx, label_name in enumerate(label_names)}
    report_df.insert(0, "label_id", report_df["label_name"].map(label_id_map))
    report_df.to_csv(save_dir / "label_scores.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrix(result_df["label"], result_df["pred_label"], label_names, save_dir)
    save_type_pair_scores(result_df, save_dir)

    print(save_dir)
    return save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--model_dir", default=None)
    args = parser.parse_args()
    run_error_analysis(args.config, args.split, args.model_dir)


if __name__ == "__main__":
    main()
