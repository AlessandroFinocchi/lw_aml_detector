"""
csecicids2018.py
================
Loader for the CSE-CIC-IDS2018 dataset (Kaggle: "dhoogla/csecicids2018",
https://www.kaggle.com/datasets/dhoogla/csecicids2018), exposing the same
interface used for UNSW-NB15, CIC-IDS 2017 and CTU-13.

Like those two, CSE-CIC-IDS2018 does NOT ship a train/test split: it is
distributed as 10 daily captures (14/02/2018 to 02/03/2018), each one a
CICFlowMeter-v3 export.

On the first run this module:

    1. (optionally) downloads the raw daily files into <dataset_path>/raw/;
    2. merges them, cleans them and builds a stratified train/test split;
    3. writes both sets to disk (parquet, with a csv fallback).

Every later call simply reloads those two files: the expensive preprocessing is
redone ONLY when they are missing (or when force_preprocess=True).

Three things make the raw 2018 files nastier than the 2017 ones:

  * The schema is not uniform. Most files have 80 columns, but the
    Thursday-01-03-2018 capture also carries Flow ID / Src IP / Src Port /
    Dst IP. Those are identifiers and get dropped, after which every file is
    aligned on the common set of columns.
  * Several files (02-16, 02-28, 03-01) repeat the header row in the middle of
    the data, which makes pandas type every column as object. Those rows are
    filtered out and the features are coerced back to numeric.
  * It is big, roughly 16M flows. Peak memory during preprocessing is a few GB;
    use `subsample` to work on a fraction of each daily file.
"""

from __future__ import annotations

import os
import glob
import shutil

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:                       # only needed to download the dataset
    import kagglehub
except ImportError:        # the module stays usable offline
    kagglehub = None


DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "csecicids2018")
KAGGLE_HANDLE = "dhoogla/csecicids2018"

RAW_SUBDIR = "raw"                                  # <dataset_path>/raw/*
TRAIN_BASENAME = "CSE-CIC-IDS2018_training-set"     # extension added when saving
TEST_BASENAME = "CSE-CIC-IDS2018_testing-set"

RAW_EXTENSIONS = (".parquet", ".csv")

# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
# There are no text features here: port and protocol are the only truly
# categorical ones (they are numeric, but perturbing them makes no sense).
CATEGORICAL_COLS = ["Dst Port", "Protocol"]  # excluded from the FGSM attack

# TCP flags are discrete counters. Add them to the mask if you do not want FGSM
# to touch them, via load_csecicids2018(non_attackable=CATEGORICAL_COLS + FLAG_COLS).
FLAG_COLS = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
]

# Columns that identify the host or the moment of the capture instead of
# describing the traffic. Only the 01-03-2018 file carries the first four.
IDENTIFIER_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]

BENIGN_LABEL = "BENIGN"   # compared uppercased: the files spell it "Benign"


# ===========================================================================
# Helpers
# ===========================================================================
def get_categorical_cols():
    return CATEGORICAL_COLS

def _find_split(basename: str) -> str | None:
    """Return the path of an already saved split (parquet or csv), else None."""
    for ext in (".parquet", ".csv"):
        path = os.path.join(DATASET_PATH, basename + ext)
        if os.path.exists(path):
            return path
    return None


def _read_split(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _write_split(df: pd.DataFrame, basename: str) -> str:
    """Save as parquet (compact and fast); fall back to csv without pyarrow."""
    parquet_path = os.path.join(DATASET_PATH, basename + ".parquet")
    csv_path = os.path.join(DATASET_PATH, basename + ".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        written, stale = parquet_path, csv_path
    except Exception:
        df.to_csv(csv_path, index=False)
        written, stale = csv_path, parquet_path

    # never leave an outdated copy of the same split in the other format
    if os.path.exists(stale):
        os.remove(stale)
    return written


def _read_raw(path: str) -> pd.DataFrame:
    """Read one raw daily file. Kaggle ships parquet, the AWS bucket ships csv."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, skipinitialspace=True)


# ===========================================================================
# Helpers: cleaning
# ===========================================================================
def _find_label_col(df: pd.DataFrame) -> str:
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(f"No 'Label' column found. Available columns: {list(df.columns)}")
    return label_col


def _normalize_label(value) -> str:
    """Trim and collapse whitespace, e.g. 'Brute Force -Web' stays readable."""
    return " ".join(str(value).strip().split())


def _capture_from_filename(path: str) -> str:
    """Keep track of which daily capture every flow comes from."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("_TrafficForML_CICFlowMeter", "")


def _safe_stratify(col: pd.Series, verbose: bool = False):
    """train_test_split fails on classes with < 2 samples: skip stratification."""
    counts = col.value_counts()
    if (counts < 2).any():
        if verbose:
            rare = list(counts[counts < 2].index)
            print(f"[warn] classes with fewer than 2 samples {rare}: stratification disabled")
        return None
    return col


def _clean_daily_file(path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Read one daily capture and bring it back to a sane numeric schema.

    Handles the two structural defects of the raw files: the repeated header
    rows and the four extra identifier columns of the 01-03-2018 capture.
    """
    df = _read_raw(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    label_col = _find_label_col(df)

    # Repeated header rows: whole lines where every cell holds its column name.
    # They are what forces pandas to type the entire file as object.
    header_rows = df[label_col].astype(str).str.strip().str.lower() == "label"
    if header_rows.any():
        df = df[~header_rows]
        if verbose:
            print(f"    repeated header rows removed: {int(header_rows.sum())}")

    # Identifiers must go before anything else: Timestamp in particular makes
    # every row unique and would defeat duplicate removal further down.
    id_cols = [c for c in IDENTIFIER_COLS if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
        if verbose:
            print(f"    identifier columns dropped: {id_cols}")

    # Everything except the label is numeric; anything that is not becomes NaN
    # and is dropped later. This also undoes the object typing above.
    df[label_col] = df[label_col].map(_normalize_label)
    feature_cols = [c for c in df.columns if c != label_col]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].astype("float32")

    return df


# ===========================================================================
# 1. Download the raw daily files
# ===========================================================================
def _download_raw(raw_dir: str) -> str:
    if kagglehub is None:
        raise ImportError("kagglehub is not installed: `pip install kagglehub`")

    os.makedirs(raw_dir, exist_ok=True)

    try:
        path = kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=raw_dir)
    except TypeError:
        # older/newer kagglehub without 'output_dir': download to cache, then copy
        path = kagglehub.dataset_download(KAGGLE_HANDLE)

    # normalise the layout: every raw file ends up in <dataset_path>/raw/
    if os.path.abspath(path) != os.path.abspath(raw_dir):
        for ext in RAW_EXTENSIONS:
            for src in glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True):
                dst = os.path.join(raw_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    print("Path to dataset files:", path)
    return raw_dir


# ===========================================================================
# 2. Preprocessing: from the 10 daily captures to a training and a test set
#    (runs once, the result is written to disk and reused afterwards)
# ===========================================================================
def build_train_test_set(
    download_dataset: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
    drop_constant_cols: bool = True,
    subsample: float | int | None = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """
    Merge the raw daily captures, clean them, split into train/test and save both.

    `subsample` is applied per daily file, not on the merged frame: with ~16M
    flows this keeps peak memory bounded. A float is the fraction kept from each
    day, an int is the maximum number of rows kept from each day; either way the
    sampling is stratified on that day's labels.
    """
    os.makedirs(DATASET_PATH, exist_ok=True)
    raw_dir = os.path.join(DATASET_PATH, RAW_SUBDIR)

    if download_dataset:
        _download_raw(raw_dir)

    files = []
    for ext in RAW_EXTENSIONS:
        files += glob.glob(os.path.join(raw_dir, "**", f"*{ext}"), recursive=True)
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No raw capture file found in '{raw_dir}'. "
            f"Run with download_dataset=True, or manually place the "
            f"'{KAGGLE_HANDLE}' files in that directory."
        )

    # ---- 2.1 read, clean and (optionally) subsample every daily capture ----
    frames = []
    for f in files:
        if verbose:
            print(f"  {os.path.basename(f)}")
        part = _clean_daily_file(f, verbose=verbose)

        # deduplicate here, while the frame is still small: most duplicates are
        # within a single day and this keeps the merged frame much lighter
        n_before = len(part)
        part = part.drop_duplicates(ignore_index=True)
        if verbose and n_before != len(part):
            print(f"    duplicate rows dropped: {n_before - len(part)}")

        if subsample is not None and len(part) > 1:
            label_col = _find_label_col(part)
            train_size = subsample if isinstance(subsample, float) else int(subsample)
            if not isinstance(train_size, float):
                train_size = min(train_size, len(part) - 1)
            part, _ = train_test_split(
                part, train_size=train_size,
                stratify=_safe_stratify(part[label_col]),
                random_state=random_state,
            )
            part = part.reset_index(drop=True)

        part["capture"] = _capture_from_filename(f)
        if verbose:
            print(f"    kept: {part.shape}")
        frames.append(part)

    # ---- 2.2 align the schemas before merging -----------------------------
    # After dropping the identifiers every file should match, but an unexpected
    # extra column would silently become an all-NaN column after concat.
    common = set(frames[0].columns)
    for part in frames[1:]:
        common &= set(part.columns)
    ordered_common = [c for c in frames[0].columns if c in common]

    all_cols = set().union(*(set(p.columns) for p in frames))
    extra_cols = sorted(all_cols - common)
    if extra_cols:
        print(f"[warn] columns missing from at least one capture, dropped: {extra_cols}")
    frames = [part[ordered_common] for part in frames]

    df = pd.concat(frames, ignore_index=True)
    frames.clear()
    if verbose:
        print(f"\nMerged dataset: {df.shape}")

    # ---- 2.3 labels: 'attack_cat' (multi-class) + binary 'label' ----------
    label_col = _find_label_col(df)
    df["attack_cat"] = df[label_col]
    df["label"] = (df["attack_cat"].str.upper() != BENIGN_LABEL).astype("int64")
    if label_col not in ("attack_cat", "label"):
        df = df.drop(columns=[label_col])

    # ---- 2.4 inf -> NaN, drop leftover NaN rows, drop duplicates ----------
    meta_cols = ["attack_cat", "label", "capture"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df = df.dropna(subset=feature_cols)
    n1 = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["attack_cat"], ignore_index=True)
    n2 = len(df)
    if verbose:
        print(f"Rows dropped (inf/NaN):        {n0 - n1}")
        print(f"Cross-capture duplicates:      {n1 - n2}")

    # ---- 2.5 constant columns carry no information ------------------------
    if drop_constant_cols:
        const_cols = [c for c in feature_cols
                      if c in df.columns and df[c].nunique(dropna=False) <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            if verbose:
                print(f"Constant columns dropped ({len(const_cols)}): {const_cols}")

    # low cardinality strings: category dtype keeps memory under control
    for col in ("attack_cat", "capture"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")

    if verbose:
        dist = df["attack_cat"].value_counts()
        print("\nClass distribution before splitting:")
        print(pd.concat([dist, (100 * dist / len(df)).round(3)],
                        axis=1, keys=["flows", "%"]).to_string())

    # ---- 2.6 stratified train/test split on the attack category -----------
    tr, te = train_test_split(
        df,
        test_size=test_size,
        stratify=_safe_stratify(df["attack_cat"], verbose),
        random_state=random_state,
    )
    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)

    # ---- 2.7 save both sets for later reuse -------------------------------
    train_path = _write_split(tr, TRAIN_BASENAME)
    test_path = _write_split(te, TEST_BASENAME)
    print(f"Training set saved to: {train_path} {tr.shape}")
    print(f"Test set saved to:     {test_path} {te.shape}")

    return train_path, test_path


# ===========================================================================
# 3. Loading (same structure as the UNSW-NB15 module)
# ===========================================================================
def get_train_val_test_set(
    download_dataset: bool = False,
    verbose: bool = False,
    *,
    force_preprocess: bool = False,
    val_size: float = 0.3,
    test_size: float = 0.3,
    random_state: int = 42,
    subsample: float | int | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # 1. Build the training/test sets only if they do not exist yet
    train_path = _find_split(TRAIN_BASENAME)
    test_path = _find_split(TEST_BASENAME)

    if force_preprocess or train_path is None or test_path is None:
        if verbose:
            print("Training/test sets not found: running the preprocessing...\n")
        train_path, test_path = build_train_test_set(
            download_dataset=download_dataset,
            test_size=test_size,
            random_state=random_state,
            subsample=subsample,
            verbose=verbose,
        )
    elif verbose:
        print(f"Reusing the existing splits:\n  {train_path}\n  {test_path}")

    # 2. Load the dataset
    tr = _read_split(train_path)
    te = _read_split(test_path)

    # 3. Handle categorical features with ordinal encoding
    #    (here: attack_cat and the capture tag, everything else is numeric)
    #    Non numeric covers str/object/category, depending on the pandas version
    #    and on the format the split was reloaded from.
    categorical_cols = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]

    oe = OrdinalEncoder(
        handle_unknown='use_encoded_value',  # for allowing unknown values
        unknown_value=-1,                    # for unknown values
        encoded_missing_value=-1             # for missing values
    )

    tr[categorical_cols] = oe.fit_transform(tr[categorical_cols].astype(str))
    te[categorical_cols] = oe.transform(te[categorical_cols].astype(str))

    # the encoder returns float64, which would upcast the whole frame during
    # scaling: stay in float32, it is what the tensors use anyway
    tr[categorical_cols] = tr[categorical_cols].astype("float32")
    te[categorical_cols] = te[categorical_cols].astype("float32")

    # keep the code -> class name mapping, only used for the verbose report
    attack_names = (
        list(oe.categories_[categorical_cols.index("attack_cat")])
        if "attack_cat" in categorical_cols else []
    )

    # 4. Handle missing values replacing them with the column name (as string)
    for col in tr.columns:
        if tr[col].isnull().any():
            tr[col] = tr[col].fillna(col)
    for col in te.columns:
        if te[col].isnull().any():
            te[col] = te[col].fillna(col)

    # 5. Standardize numerical features
    exclude_columns = ['attack_cat', 'label', 'capture']
    scaler = StandardScaler()

    tr_features = tr.drop(columns=exclude_columns, errors='ignore')
    numeric_cols = tr_features.select_dtypes(include=['number']).columns

    # fit the scaler only on the training data, transform both sets
    tr[numeric_cols] = scaler.fit_transform(tr_features[numeric_cols])
    te[numeric_cols] = scaler.transform(te[numeric_cols])

    # 6. Stratify the train/val split (70/30) based on attack_cat
    strat_col = tr['attack_cat']

    X = tr.drop(columns=['label'])
    y = tr['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size,
        stratify=_safe_stratify(strat_col, verbose), random_state=random_state
    )

    X_test = te.drop(columns=['label'])
    y_test = te['label']

    if verbose:
        print("\nTraining set size:", X_train.shape)
        print("Validation set size:", X_val.shape)
        print("Test set size:", X_test.shape)

        attack_counts = pd.concat([
                X_train['attack_cat'].value_counts(),
                X_val['attack_cat'].value_counts(),
                X_test['attack_cat'].value_counts()
            ],
            axis=1
        )
        attack_counts.columns = ['Train', 'Validation', 'Test']
        attack_counts = attack_counts.fillna(0).astype(int)
        if attack_names:
            attack_counts.index = [
                attack_names[int(i)] if 0 <= int(i) < len(attack_names) else i
                for i in attack_counts.index
            ]
        print("\nStratification on attack categories:\n")
        print(attack_counts)

        pos = int(y_train.sum())
        print(f"\nAttack flows in the training set: {pos} "
              f"({100 * pos / max(len(y_train), 1):.3f}%)")

    # 7. Drop id, capture and attack_cat columns
    def clean(df):
        drop_cols = [
            "id",          # it is just a sequential number
            "capture",     # daily capture tag, not a traffic feature
            "attack_cat"   # it is the multi-class version of label
        ]
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_val, X_test = clean(X_train), clean(X_val), clean(X_test)
    print("Dropped 'id', 'capture' and 'attack_cat' columns")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ===========================================================================
# Data: CSE-CIC-IDS2018 through the functions of this module
# ===========================================================================
def load_csecicids2018(
    download_dataset: bool = False,
    verbose: bool = False,
    non_attackable: list[str] | None = None,
    **kwargs,
):
    """Load the three sets and build the attack mask."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_train_val_test_set(
        download_dataset, verbose, **kwargs
    )

    feature_names = list(X_tr.columns)
    non_attackable = CATEGORICAL_COLS if non_attackable is None else non_attackable

    # 1.0 = continuous attackable feature, 0.0 = untouchable categorical one
    attack_mask = torch.tensor(
        [0.0 if c in non_attackable else 1.0 for c in feature_names]
    )

    to_x = lambda df: torch.from_numpy(df.to_numpy(dtype="float32"))
    to_y = lambda s: torch.from_numpy(s.to_numpy()).long()
    return (to_x(X_tr), to_y(y_tr), to_x(X_val), to_y(y_val),
            to_x(X_te), to_y(y_te), feature_names, attack_mask)


if __name__ == "__main__":
    get_train_val_test_set("dataset/cse-cic-ids2018/", download_dataset=True, verbose=True)
