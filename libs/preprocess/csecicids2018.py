"""
CSE-CIC-IDS2018 ships as 10 daily captures (14/02/2018 to 02/03/2018) with no
train/test split, so the first call builds one and saves it; later calls just
reload it. All of that lives in utils.py -- this module only describes what is
specific to CSE-CIC-IDS2018: how to clean one of its daily files.

Three things make these raw files nastier than the 2017 ones:

  * The schema is not uniform. Most files have 80 columns, but the
    Thursday-01-03-2018 capture also carries Flow ID / Src IP / Src Port /
    Dst IP. Those are identifiers and get dropped here; utils then aligns the
    captures on the columns they all share.
  * Several files (02-16, 02-28, 03-01) repeat the header row in the middle of
    the data, which makes pandas type every column as object.
  * It is big, roughly 16M flows.
"""

from __future__ import annotations

import os
import pandas as pd

import libs.preprocess.utils as utils
from libs.preprocess.utils import DatasetConfig


# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
CATEGORICAL_COLS = ["Dst Port", "Protocol"]  # excluded from the FGSM attack

# TCP flags are discrete counters
FLAG_COLS = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
]
# For avoiding FGSM to attack FLAG_COLS
CATEGORICAL_COLS+=FLAG_COLS

# Only one capture carries the first four fields,
# while the Timestamp could has a negative effect.
IDENTIFIER_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]

BENIGN_LABEL = "BENIGN"   # compared uppercased


def get_categorical_cols():
    return CATEGORICAL_COLS


def _clean_file(path: str, verbose: bool = False) -> pd.DataFrame:
    """Read one daily capture and return it with attack_cat + label attached."""
    df = utils.normalize_columns(utils.read_raw(path))
    label_col = utils.find_label_col(df)

    # Repeated header rows: whole lines where every cell holds its column name.
    # They are what forces pandas to type the entire file as object.
    header_rows = df[label_col].astype(str).str.strip().str.lower() == "label"
    if header_rows.any():
        df = df[~header_rows]
        if verbose:
            print(f"    repeated header rows removed: {int(header_rows.sum())}")

    # Identifiers must go before anything else: Timestamp in particular makes
    # every row unique and would defeat duplicate removal further down.
    df = utils.drop_if_present(df, IDENTIFIER_COLS, "identifier columns", verbose)

    df["attack_cat"] = df[label_col].map(lambda v: " ".join(str(v).strip().split()))
    df["label"] = (df["attack_cat"].str.upper() != BENIGN_LABEL).astype("int64")
    df = df.drop(columns=[label_col])

    # Everything except the labels is numeric
    feature_cols = [c for c in df.columns if c not in ("attack_cat", "label")]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

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
    return utils.get_train_val_test_set(CONFIG, download_dataset, verbose, **kwargs)


def load_csecicids2018(download_dataset: bool = False, verbose: bool = False, **kwargs):
    return utils.load_tensors(CONFIG, download_dataset, verbose, **kwargs)


if __name__ == "__main__":
    get_train_val_test_set(download_dataset=True, verbose=True)
