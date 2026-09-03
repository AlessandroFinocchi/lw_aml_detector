"""
CTU-13 ships as multi separate captures ("scenarios") of bidirectional
NetFlow records, with no train/test split, so the first call builds one and
saves it; later calls just reload it.

Two things worth knowing about this dataset:
  * Source/destination IPs are dropped. The botnet hosts keep fixed addresses
    inside a scenario, so keeping them lets a classifier memorise the answer
    instead of learning traffic behaviour.
  * It is heavily imbalanced (botnet flows are ~1-2% of the total, the rest
    mostly background). Use `drop_background=True` to keep only Normal +
    Botnet, which is the usual setup since background traffic has no reliable
    ground truth.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from functools import partial

import libs.preprocess.utils as utils
from libs.preprocess.utils import DatasetConfig


# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
CATEGORICAL_COLS = ["Proto", "Dir", "State"]  # excluded from the FGSM attack

# Ports are numeric but discrete
PORT_COLS = ["Sport", "Dport"]

# For avoiding FGSM to attack PORT_COLS
CATEGORICAL_COLS+=PORT_COLS

# Columns that identify the host or the moment of the capture rather than
# describing the traffic. Timestamps must also go before deduplication, or
# every row looks unique.
IDENTIFIER_COLS = ["StartTime", "SrcAddr", "DstAddr", "LastTime", "Timestamp"]

# For having balanced dataset with ~ 200k rows
MAX_PER_CLASS  = 50 * 1000
MAJORITY_RATIO = 0.4

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


def _to_port(series: pd.Series) -> pd.Series:
    """
    Parse a port column. ICMP flows store ports as hex strings such as '0x0303',
    so a plain to_numeric() would silently turn them into NaN.
    """
    text = series.astype(str).str.strip().str.lower()
    port = pd.to_numeric(text, errors="coerce")

    hex_mask = port.isna() & text.str.startswith("0x")
    if hex_mask.any():
        def parse_hex(value):
            try:
                return float(int(value, 16))
            except ValueError:
                return np.nan
        port.loc[hex_mask] = text[hex_mask].map(parse_hex)

    return port.astype("float32")


def _clean_file(path: str, verbose: bool = False,
                drop_background: bool = False) -> pd.DataFrame:
    """Read one scenario and return it with attack_cat + label attached."""
    df = utils.normalize_columns(utils.read_raw(path))

    label_col = utils.find_label_col(df)
    df["attack_cat"] = df[label_col].map(_attack_category)
    df["label"] = (df["attack_cat"] == BOTNET_CLASS).astype("int64")
    df = df.drop(columns=[label_col])

    if drop_background:
        n_before = len(df)
        df = df[df["attack_cat"] != BACKGROUND_CLASS].reset_index(drop=True)
        if verbose:
            print(f"    background flows dropped: {n_before - len(df)}")

    df = utils.drop_if_present(df, IDENTIFIER_COLS, "identifier columns", verbose)

    # ports may be decimal or hexadecimal
    for col in PORT_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = _to_port(df[col])

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
                           drop_background: bool = False, **kwargs):
    return utils.get_train_val_test_set(
        CONFIG, download_dataset=download_dataset, verbose=verbose,
        max_per_class=MAX_PER_CLASS, majority_ratio=MAJORITY_RATIO,
        clean_file=partial(_clean_file, drop_background=drop_background), **kwargs
    )


def load_ctu13(download_dataset: bool = False, verbose: bool = False, *, 
               drop_background: bool = False, **kwargs):
    return utils.load_tensors(
        CONFIG, download_dataset=download_dataset, verbose=verbose,
        max_per_class=MAX_PER_CLASS, majority_ratio=MAJORITY_RATIO,
        clean_file=partial(_clean_file, drop_background=drop_background), **kwargs
    )


if __name__ == "__main__":
    get_train_val_test_set(download_dataset=True, verbose=True)
