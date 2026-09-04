"""
CSE-CIC-IDS2018 ships as 10 daily captures (14/02/2018 to 02/03/2018) with no
train/test split, so the first call builds one and saves it; later calls just
reload it. All of that lives in utils.py -- this module only describes what is
specific to CSE-CIC-IDS2018: how to clean one of its daily files.

Very big dataset, roughly 16M flows.
"""

from __future__ import annotations

import os
import pandas as pd

import libs.preprocess.utils as utils
from libs.preprocess.utils import DatasetConfig


# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
CATEGORICAL_COLS = ["Protocol"]  # excluded from the FGSM attack

# TCP flags are discrete counters
FLAG_COLS = [
    "Fwd PSH Flags", "Fwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
]
# For avoiding FGSM to attack FLAG_COLS
CATEGORICAL_COLS+=FLAG_COLS

# For having balanced dataset with ~ 200k rows
MAX_PER_CLASS  = 14 * 1000
MAJORITY_RATIO = 0.11

BENIGN_LABEL = "BENIGN"   # compared uppercased


def get_categorical_cols():
    return CATEGORICAL_COLS


def _clean_file(path: str, verbose: bool = False) -> pd.DataFrame:
    """Read one daily capture and return it with attack_cat + label attached."""
    df = utils.normalize_columns(utils.read_raw(path))
    label_col = utils.find_label_col(df)

    df["attack_cat"] = df[label_col].map(lambda v: " ".join(str(v).strip().split()))
    df["label"] = (df["attack_cat"].str.upper() != BENIGN_LABEL).astype("int64")
    # a raw column already named 'label' was overwritten just above: dropping
    # it would throw the binary label away
    if label_col != "label":
        df = df.drop(columns=[label_col])

    return df


CONFIG = DatasetConfig(
    name="CSE-CIC-IDS2018",
    kaggle_handle="dhoogla/csecicids2018",
    dataset_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "csecicids2018"),
    train_basename="CSE-CIC-IDS2018_training-set",
    test_basename="CSE-CIC-IDS2018_testing-set",
    clean_file=_clean_file,
    categorical_cols=CATEGORICAL_COLS,
)


def get_train_val_test_set(download_dataset: bool = False, verbose: bool = False, **kwargs):
    """X_train, y_train, X_val, y_val, X_test, y_test. See utils for the options."""
    return utils.get_train_val_test_set(CONFIG, download_dataset, verbose,
        max_per_class=MAX_PER_CLASS, majority_ratio=MAJORITY_RATIO, **kwargs)


def load_csecicids2018(download_dataset: bool = False, verbose: bool = False, **kwargs):
    return utils.load_tensors(CONFIG, download_dataset, verbose,
        max_per_class=MAX_PER_CLASS, majority_ratio=MAJORITY_RATIO, **kwargs)


if __name__ == "__main__":
    get_train_val_test_set(download_dataset=True, verbose=True)
