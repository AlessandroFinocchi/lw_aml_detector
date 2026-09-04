"""
CTU-13 ships as multi separate captures ("scenarios") of bidirectional
NetFlow records, with no train/test split, so the first call builds one and
saves it; later calls just reload it.

This dataset is heavily imbalanced (botnet flows are ~1-2% of the total, the 
rest mostly background). `drop_background=True` is therefore the default: it
keeps only Normal + Botnet, which is the usual setup since background
traffic has no reliable ground truth.
"""

from __future__ import annotations

import os
import pandas as pd
from functools import partial

import libs.preprocess.utils as utils
from libs.preprocess.utils import DatasetConfig


# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
CATEGORICAL_COLS = ["proto", "dir", "state"]  # excluded from the FGSM attack

# Columns to drop: they identify the host or the capture moment rather than
# describing the traffic. Timestamps also dropped before deduplication, or
# every row looks unique. 'Family' is the capture file name.
IDENTIFIER_COLS = ["StartTime", "SrcAddr", "DstAddr", "LastTime", "Timestamp",
                   "Family"]

# For having balanced dataset with ~ 200k rows
MAX_PER_CLASS  = 100 * 1000
MAJORITY_RATIO = 1

BOTNET_CLASS     = "Botnet"
NORMAL_CLASS     = "Normal"
BACKGROUND_CLASS = "Background"


def get_categorical_cols():
    return CATEGORICAL_COLS


def _attack_category(label) -> str:
    """
    Raw labels look like 'flow=From-Botnet-V42-TCP-Attempt' or
    'flow=Background-Established-cmpgw-CVUT', but always contain the keys
    botnet, normal or background. Some redistributions replace them
    with plain 0/1 flags or similar, which is handled here as well.
    """
    text = str(label).strip().lower()

    if text in ("1", "1.0", "true", "botnet", "malicious", "attack"):
        return BOTNET_CLASS
    if text in ("0", "0.0", "false", "benign", "normal"):
        return NORMAL_CLASS

    if "botnet" in text:
        return BOTNET_CLASS
    if "normal" in text:
        return NORMAL_CLASS
    if "background" in text:
        return BACKGROUND_CLASS
    return "Unknown"



def _clean_file(path: str, verbose: bool = False,
                drop_background: bool = False) -> pd.DataFrame:
    """
    Read one scenario and return it with attack_cat + label attached.
    Background traffic has no reliable ground truth, so drop it by def.
    """
    df = utils.normalize_columns(utils.read_raw(path))

    label_col = utils.find_label_col(df)
    df["attack_cat"] = df[label_col].map(_attack_category)
    df["label"] = (df["attack_cat"] == BOTNET_CLASS).astype("int64")

    # avoid that another column contains the labels
    if label_col != "label":
        df = df.drop(columns=[label_col])

    if drop_background:
        n_before = len(df)
        df = df[df["attack_cat"] != BACKGROUND_CLASS].reset_index(drop=True)
        if verbose:
            print(f"    background flows dropped: {n_before - len(df)}")

    df = utils.drop_if_present(df, IDENTIFIER_COLS, "identifier columns", verbose)

    # type of service: a missing value is meaningful, encode it as -1 rather
    # than losing the whole row later
    for col in ("sTos", "dTos"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    return df


CONFIG = DatasetConfig(
    name="CTU-13",
    kaggle_handle="dhoogla/ctu13",
    dataset_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "ctu13"),
    train_basename="CTU13_training-set",
    test_basename="CTU13_testing-set",
    clean_file=_clean_file,
    categorical_cols=CATEGORICAL_COLS,
    raw_extensions=(".parquet", ".csv", ".binetflow"),
)


def get_train_val_test_set(download_dataset: bool = False, verbose: bool = False, *, 
                           drop_background: bool = True, **kwargs):
    return utils.get_train_val_test_set(
        CONFIG, download_dataset=download_dataset, verbose=verbose,
        max_per_class=MAX_PER_CLASS, majority_ratio=MAJORITY_RATIO,
        clean_file=partial(_clean_file, drop_background=drop_background), **kwargs
    )


def load_ctu13(download_dataset: bool = False, verbose: bool = False, *, 
               drop_background: bool = True, **kwargs):
    return utils.load_tensors(
        CONFIG, download_dataset=download_dataset, verbose=verbose,
        max_per_class=MAX_PER_CLASS, majority_ratio=MAJORITY_RATIO,
        clean_file=partial(_clean_file, drop_background=drop_background), **kwargs
    )


if __name__ == "__main__":
    get_train_val_test_set(download_dataset=True, verbose=True)
