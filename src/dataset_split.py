from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_STRATEGIES = {"random", "label_stratified", "detailed_stratified"}


def _entity_type(entity) -> str:
    return str(entity["type"]).upper()


def make_detailed_stratify_key(df: pd.DataFrame) -> pd.Series:
    """Create label-subject_type-object_type stratification keys."""
    subject_type = df["subject_entity"].map(_entity_type)
    object_type = df["object_entity"].map(_entity_type)
    return df["label"].astype(str) + "-" + subject_type + "-" + object_type


def _safe_detailed_key(df: pd.DataFrame) -> pd.Series:
    """Merge very rare detailed groups so sklearn stratified split can run."""
    detail_key = make_detailed_stratify_key(df)
    counts = detail_key.value_counts()
    rare_mask = detail_key.map(counts) < 2
    safe_key = detail_key.copy()
    safe_key.loc[rare_mask] = df.loc[rare_mask, "label"].astype(str) + "-RARE"
    if safe_key.value_counts().min() < 2:
        return df["label"]
    return safe_key


def split_train_val(
    train_df: pd.DataFrame,
    strategy: str = "label_stratified",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the original train split into train/val by strategy."""
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(SPLIT_STRATEGIES)}")

    stratify = None
    if strategy == "label_stratified":
        stratify = train_df["label"]
    elif strategy == "detailed_stratified":
        stratify = _safe_detailed_key(train_df)

    train_part, val_part = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def stratified_train_val_split(
    train_df: pd.DataFrame,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper for label stratified split."""
    return split_train_val(train_df, "label_stratified", val_ratio, seed)


def make_train_val_test(
    hf_train_df: pd.DataFrame,
    hf_test_df: pd.DataFrame,
    val_ratio: float = 0.1,
    seed: int = 42,
    strategy: str = "label_stratified",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the original KLUE held-out split as test and split HF train into train/val."""
    train_df, val_df = split_train_val(hf_train_df, strategy, val_ratio, seed)
    test_df = hf_test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def label_distribution(df: pd.DataFrame, label_names: list[str] | None = None) -> pd.DataFrame:
    dist = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    dist["ratio"] = dist["count"] / len(df)
    if label_names is not None:
        dist["label_name"] = dist["label"].map(lambda x: label_names[int(x)])
    return dist.reset_index(drop=True)


def compare_label_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    train_dist = label_distribution(train_df, label_names).rename(
        columns={"count": "train_count", "ratio": "train_ratio"}
    )
    val_dist = label_distribution(val_df, label_names).rename(
        columns={"count": "val_count", "ratio": "val_ratio"}
    )
    merge_cols = ["label"] + (["label_name"] if label_names is not None else [])
    result = train_dist.merge(val_dist, on=merge_cols, how="outer").fillna(0)
    result["ratio_gap"] = (result["train_ratio"] - result["val_ratio"]).abs()
    return result.sort_values("label").reset_index(drop=True)
