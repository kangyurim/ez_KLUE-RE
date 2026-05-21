from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """logit을 확률로 변환한다."""
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def klue_re_micro_f1(
    preds: np.ndarray,
    labels: np.ndarray,
    num_labels: int | None = None,
    no_relation_id: int = 0,
) -> float:
    """KLUE-RE 공식 micro-F1.

    공식 기준에 맞춰 no_relation class를 제외하고 0~100 스케일로 반환한다.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if num_labels is None:
        num_labels = int(max(preds.max(), labels.max())) + 1
    label_indices = [idx for idx in range(num_labels) if idx != no_relation_id]
    return float(f1_score(labels, preds, average="micro", labels=label_indices, zero_division=0) * 100.0)


def klue_re_auprc(probs: np.ndarray, labels: np.ndarray, num_labels: int | None = None) -> float:
    """KLUE-RE 공식 AUPRC.

    no_relation을 포함한 전체 class에 대해 class별 PR AUC를 계산한 뒤 평균한다.
    결과는 0~100 스케일로 반환한다.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if num_labels is None:
        num_labels = probs.shape[1]

    one_hot_labels = np.eye(num_labels)[labels]
    scores = np.zeros((num_labels,))
    for class_idx in range(num_labels):
        targets_c = one_hot_labels.take([class_idx], axis=1).ravel()
        preds_c = probs.take([class_idx], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        scores[class_idx] = auc(recall, precision)
    return float(np.average(scores) * 100.0)


def compute_metrics(eval_pred) -> Dict[str, float]:
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    probs = softmax_np(logits)
    return compute_metrics_from_probs(probs, labels)


def compute_metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """확률값으로부터 공통 평가 지표를 계산한다."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = np.argmax(probs, axis=-1)
    num_labels = probs.shape[1]
    return {
        "accuracy": float(accuracy_score(labels, preds) * 100.0),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0) * 100.0),
        "micro_f1": klue_re_micro_f1(preds, labels, num_labels=num_labels),
        "auprc": klue_re_auprc(probs, labels, num_labels=num_labels),
    }


def append_experiment_result(path: str | os.PathLike, row: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_device_info() -> bool:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return cuda_available


def get_entity_types(*dfs) -> list[str]:
    types = set()
    for df in dfs:
        for col in ("subject_entity", "object_entity"):
            for entity in df[col]:
                types.add(str(entity["type"]).upper())
    return sorted(types)
