"""
CIC-IDS 2017 ships as 8 daily CSV files (the MachineLearningCVE export) with no
train/test split, so the first call builds one and saves it; later calls just
reload it. All of that lives in utils.py -- this module only describes what is
specific to CIC-IDS 2017: how to clean one of its daily files.
"""

from __future__ import annotations

import os
import pandas as pd

import libs.preprocess.utils as utils
from libs.preprocess.utils import DatasetConfig


# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
CATEGORICAL_COLS = ["Destination Port", "Protocol"]  # excluded from the FGSM attack

# TCP flags are discrete counters.
FLAG_COLS = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
]
# For avoiding FGSM to attack FLAG_COLS
CATEGORICAL_COLS+=FLAG_COLS

BENIGN_LABEL = "BENIGN"

def get_categorical_cols():
    return CATEGORICAL_COLS


def _normalize_label(value) -> str:
    """Normalise non-ASCII bytes to '-'"""
    text = str(value).strip()
    text = "".join(ch if ord(ch) < 128 else "-" for ch in text)
    return " ".join(text.split())


def _clean_file(path: str, verbose: bool = False) -> pd.DataFrame:
    """Read one daily capture and return it with attack_cat + label attached."""
    # latin1: some files hold bytes that are not valid UTF-8
    df = utils.read_raw(path, encoding="latin1")

    # column names come padded with spaces, and "Fwd Header Length" appears
    # twice (pandas renames the second one "Fwd Header Length.1")
    df = utils.normalize_columns(df)

    label_col = utils.find_label_col(df)
    df["attack_cat"] = df[label_col].map(_normalize_label)
    df["label"] = (df["attack_cat"].str.upper() != BENIGN_LABEL).astype("int64")

    return df.drop(columns=[label_col])


CONFIG = DatasetConfig(
    name="CIC-IDS 2017",
    kaggle_handle="chethuhn/network-intrusion-dataset",
    dataset_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "cicids2017"),
    train_basename="CIC-IDS2017_training-set",
    test_basename="CIC-IDS2017_testing-set",
    clean_file=_clean_file,
    categorical_cols=CATEGORICAL_COLS,
)


def get_train_val_test_set(download_dataset: bool = False,verbose: bool = False, **kwargs):
    return utils.get_train_val_test_set(CONFIG, download_dataset, verbose, **kwargs)


def load_cicids(download_dataset: bool = False, verbose: bool = False, **kwargs):
    return utils.load_tensors(CONFIG, download_dataset, verbose, **kwargs)


if __name__ == "__main__":
    get_train_val_test_set(download_dataset=True, verbose=True)
