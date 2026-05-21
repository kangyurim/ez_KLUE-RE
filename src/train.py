from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer, TrainingArguments

try:
    from .custom_trainer import LossTrainer
    from .data_loader import load_as_pandas
    from .dataset_split import make_train_val_test
    from .models import build_model
    from .preprocess import add_special_tokens, preprocess_dataframe
    from .tokenization import ReDataCollator, tokenize_dataset
    from .utils import append_experiment_result, compute_metrics, get_entity_types, load_config, print_device_info, set_seed
except ImportError:
    from custom_trainer import LossTrainer
    from data_loader import load_as_pandas
    from dataset_split import make_train_val_test
    from models import build_model
    from preprocess import add_special_tokens, preprocess_dataframe
    from tokenization import ReDataCollator, tokenize_dataset
    from utils import append_experiment_result, compute_metrics, get_entity_types, load_config, print_device_info, set_seed


def build_training_args(config, cuda_available: bool) -> TrainingArguments:
    """config 파일의 학습 설정을 Hugging Face TrainingArguments로 변환한다."""
    params = {
        "output_dir": config["output_dir"],
        "num_train_epochs": config.get("num_train_epochs", 3),
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", config.get("train_batch_size", 16)),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", config.get("eval_batch_size", 32)),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "weight_decay": config.get("weight_decay", 0.01),
        "logging_steps": config.get("logging_steps", 100),
        "save_total_limit": config.get("save_total_limit", 2),
        "warmup_ratio": config.get("warmup_ratio", 0.0),
        "load_best_model_at_end": config.get("load_best_model_at_end", True),
        "metric_for_best_model": config.get("metric_for_best_model", "micro_f1"),
        "greater_is_better": config.get("greater_is_better", True),
        "fp16": bool(config.get("fp16", True) and cuda_available),
        "report_to": config.get("report_to", "none"),
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        params["eval_strategy"] = config.get("eval_strategy", "epoch")
    else:
        params["evaluation_strategy"] = config.get("eval_strategy", "epoch")
    params["save_strategy"] = config.get("save_strategy", "epoch")

    return TrainingArguments(**params)


def build_class_weights(config, train_df, label_names):
    """class_weighting 옵션이 켜진 경우 label별 weight tensor를 생성한다."""
    if not config.get("class_weighting", False):
        return None

    weights = compute_class_weight(
        "balanced",
        classes=np.arange(len(label_names)),
        y=train_df["label"].values,
    )
    return torch.tensor(weights, dtype=torch.float)


def run_train(config_path: str) -> dict:
    """config 하나에 대한 전체 학습 파이프라인을 실행한다."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    cuda_available = print_device_info()

    hf_train_df, hf_test_df, label_names, label2id, id2label = load_as_pandas()
    train_df, val_df, test_df = make_train_val_test(
        hf_train_df,
        hf_test_df,
        val_ratio=config.get("train_val_split_ratio", 0.1),
        seed=config.get("seed", 42),
        strategy=config.get("split_strategy", "label_stratified"),
    )

    input_style = config.get("input_style", "entity_marker")
    train_df = preprocess_dataframe(train_df, input_style)
    val_df = preprocess_dataframe(val_df, input_style)
    test_df = preprocess_dataframe(test_df, input_style)

    architecture = config.get("architecture", "cls")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    add_special_tokens(tokenizer, input_style, get_entity_types(train_df, val_df, test_df))

    model = build_model(config, tokenizer, label_names, id2label, label2id)

    max_length = config.get("max_length", 256)
    train_dataset = tokenize_dataset(train_df, tokenizer, max_length, architecture)
    val_dataset = tokenize_dataset(val_df, tokenizer, max_length, architecture)
    data_collator = ReDataCollator(tokenizer)

    training_args = build_training_args(config, cuda_available)
    trainer_kwargs = {}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = LossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=build_class_weights(config, train_df, label_names),
        focal_gamma=config.get("focal_gamma"),
        **trainer_kwargs,
    )

    start = time.time()
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    val_metrics = trainer.evaluate(val_dataset, metric_key_prefix="val")
    test_metrics = {}
    if config.get("evaluate_test", False):
        test_dataset = tokenize_dataset(test_df, tokenizer, max_length, architecture)
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    row = build_result_row(
        config_path=config_path,
        config=config,
        input_style=input_style,
        architecture=architecture,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        elapsed_sec=round(time.time() - start, 2),
    )

    root = Path(__file__).resolve().parents[1]
    append_experiment_result(root / "outputs" / "experiments.csv", row)
    return row


def build_result_row(config_path, config, input_style, architecture, val_metrics, test_metrics, elapsed_sec):
    """공통 실험 결과 CSV에 저장할 row를 생성한다."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(config_path),
        "experiment_group": config.get("experiment_group", "single"),
        "experiment_name": config.get("experiment_name", Path(config_path).stem),
        "model": config["model_name_or_path"],
        "input_style": input_style,
        "split_strategy": config.get("split_strategy", "label_stratified"),
        "architecture": architecture,
        "max_length": config.get("max_length", 256),
        "learning_rate": config.get("learning_rate", 2e-5),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", config.get("train_batch_size", 16)),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", config.get("eval_batch_size", 32)),
        "output_dir": config["output_dir"],
        "val_accuracy": val_metrics.get("val_accuracy"),
        "val_macro_f1": val_metrics.get("val_macro_f1"),
        "val_micro_f1": val_metrics.get("val_micro_f1"),
        "val_auprc": val_metrics.get("val_auprc"),
        "test_accuracy": test_metrics.get("test_accuracy"),
        "test_macro_f1": test_metrics.get("test_macro_f1"),
        "test_micro_f1": test_metrics.get("test_micro_f1"),
        "test_auprc": test_metrics.get("test_auprc"),
        "note": config.get("note", ""),
        "elapsed_sec": elapsed_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_train(args.config))


if __name__ == "__main__":
    main()
