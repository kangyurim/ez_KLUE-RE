from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "klue_re"
HF_TEST_SPLIT = "validation"


def _get_label_info(dataset: DatasetDict) -> Tuple[list[str], Dict[str, int], Dict[int, str]]:
    label_feature = dataset["train"].features["label"]
    label_names = list(label_feature.names)
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_names, label2id, id2label


def save_klue_re_raw(dataset: DatasetDict, raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR) -> Path:
    """Save the original KLUE RE DatasetDict under data/raw/klue_re."""
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.parent.mkdir(parents=True, exist_ok=True)
    if not raw_data_dir.exists():
        dataset.save_to_disk(str(raw_data_dir))
    return raw_data_dir


def load_saved_klue_re(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Tuple[DatasetDict, list[str], Dict[str, int], Dict[int, str]]:
    """Load the KLUE RE dataset saved by save_klue_re_raw()."""
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Saved KLUE RE data was not found: {raw_data_dir}\n"
            "Run full_pipeline.ipynb or load_klue_re() first to save the raw dataset."
        )
    dataset = load_from_disk(str(raw_data_dir))
    label_names, label2id, id2label = _get_label_info(dataset)
    return dataset, label_names, label2id, id2label


def load_klue_re(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    save_raw: bool = True,
    prefer_saved: bool = True,
) -> Tuple[DatasetDict, list[str], Dict[str, int], Dict[int, str]]:
    """
    Load KLUE RE with Hugging Face load_dataset("klue", "re").

    By default, the first call downloads the original dataset and saves it to
    data/raw/klue_re. Later calls reuse the saved copy when it exists.
    """
    raw_data_dir = Path(raw_data_dir)
    if prefer_saved and raw_data_dir.exists():
        return load_saved_klue_re(raw_data_dir)

    dataset = load_dataset("klue", "re")
    if save_raw:
        save_klue_re_raw(dataset, raw_data_dir)
    label_names, label2id, id2label = _get_label_info(dataset)
    return dataset, label_names, label2id, id2label


def get_train_test_splits() -> Tuple[Dataset, Dataset]:
    """Return train and test splits."""
    dataset, _, _, _ = load_klue_re()
    return dataset["train"], dataset[HF_TEST_SPLIT]


def to_pandas(dataset: Dataset) -> pd.DataFrame:
    """Convert a Hugging Face Dataset split to pandas."""
    return dataset.to_pandas()


def load_as_pandas() -> Tuple[pd.DataFrame, pd.DataFrame, list[str], Dict[str, int], Dict[int, str]]:
    """Load KLUE RE and return train and test DataFrames."""
    dataset, label_names, label2id, id2label = load_klue_re()
    return (
        to_pandas(dataset["train"]),
        to_pandas(dataset[HF_TEST_SPLIT]),
        label_names,
        label2id,
        id2label,
    )
