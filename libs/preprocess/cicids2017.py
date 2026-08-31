from __future__ import annotations

import os
import glob
import shutil

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    import kagglehub
except ImportError:        # module is usable offline
    kagglehub = None


DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "cicids2017")
KAGGLE_HANDLE = "chethuhn/network-intrusion-dataset"

RAW_SUBDIR = "raw"                             # <dataset_path>/raw/*.csv
TRAIN_BASENAME = "CIC-IDS2017_training-set"    # no ext
TEST_BASENAME = "CIC-IDS2017_testing-set"

CATEGORICAL_COLS = ["Destination Port", "Protocol"]  # excluded from FGSM attack

# Flag TCP: discretee/binary counts
FLAG_COLS = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
]

# To exclude FLAG_COLS too
#CATEGORICAL_COLS += FLAG_COLS

BENIGN_LABEL = "BENIGN"


# ===========================================================================
# Helpers
# ===========================================================================
def get_categorical_cols():
    return CATEGORICAL_COLS

def _find_split(basename: str) -> str | None:
    """Returns already stored split path (parquet or csv), otherwise None."""
    for ext in (".parquet", ".csv"):
        path = os.path.join(DATASET_PATH, basename + ext)
        if os.path.exists(path):
            return path
    return None


def _read_split(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _write_split(df: pd.DataFrame, basename: str) -> str:
    """Stores parquet files if possibile, otherwise uses csv"""
    parquet_path = os.path.join(DATASET_PATH, basename + ".parquet")
    csv_path = os.path.join(DATASET_PATH, basename + ".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        written, stale = parquet_path, csv_path
    except Exception:
        df.to_csv(csv_path, index=False)
        written, stale = csv_path, parquet_path

    # only 1 existing version (no duplicate)
    if os.path.exists(stale):
        os.remove(stale)
    return written


# ===========================================================================
# Cleaning
# ===========================================================================
def _normalize_label(value) -> str:
    """
    Normilizing non-ASCII label characters with a simple '-' 
    """
    s = str(value).strip()
    s = "".join(ch if ord(ch) < 128 else "-" for ch in s)
    return " ".join(s.split())


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float32 instead of float64/int64 for sparing RAM (the dataset has got ~2.8M rows)."""
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].astype("float32")
    return df


def _safe_stratify(col: pd.Series, verbose: bool = False):
    """don't stratify classes with less than 2 samples"""
    counts = col.value_counts()
    if (counts < 2).any():
        if verbose:
            rare = list(counts[counts < 2].index)
            print(f"[warn] classes with less than 2 samples {rare}: no stratification")
        return None
    return col


# ===========================================================================
# 1. Download raw CSV
# ===========================================================================
def _download_raw(raw_dir: str) -> str:
    if kagglehub is None:
        raise ImportError("kagglehub not installed: `pip install kagglehub`")

    os.makedirs(raw_dir, exist_ok=True)
    path = kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=raw_dir)

    # moves csv files in /raw directory
    if os.path.abspath(path) != os.path.abspath(raw_dir):
        for src in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
            dst = os.path.join(raw_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print("Path to dataset files:", path)
    return raw_dir


# ===========================================================================
# 2. Preprocessing
# ===========================================================================
def build_train_test_set(
    download_dataset: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
    drop_constant_cols: bool = True,
    subsample: float | int | None = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Merges raw csvs, cleans them all, splits in train and test and stores them, returning their paths """
    os.makedirs(DATASET_PATH, exist_ok=True)
    raw_dir = os.path.join(DATASET_PATH, RAW_SUBDIR)

    if download_dataset:
        _download_raw(raw_dir)

    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"CSV not found in '{raw_dir}'. "
            f"Execute with download_dataset=True."
        )

    # ---- 2.1 read and merge daily CSVs ------------------------------------
    frames = []
    for f in files:
        # encoding latin1: some files contain UTF-8 not valid bytes
        part = pd.read_csv(f, encoding="latin1", low_memory=False, skipinitialspace=True)

        # original column names contain blank spaces
        part.columns = [str(c).strip() for c in part.columns]

        # some columns are duplicated, pandas renames them with "....1" -----
        dup = [c for c in part.columns if c.endswith(".1") and c[:-2] in part.columns]
        part = part.drop(columns=dup)
        part = part.loc[:, ~part.columns.duplicated()]

        if verbose:
            print(f"  {os.path.basename(f):<55} {part.shape}")
        frames.append(_downcast(part))

    df = pd.concat(frames, ignore_index=True)
    frames.clear()
    if verbose:
        print(f"\Merged Dataset: {df.shape}")

    # ---- 2.2 label: 'attack_cat' (multi-class) + 'label' (binary) ---------
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(f"'Label' column not found. Columns are: {list(df.columns)}")

    df["attack_cat"] = df[label_col].map(_normalize_label)
    df["label"] = (df["attack_cat"].str.upper() != BENIGN_LABEL).astype("int64")
    if label_col not in ("attack_cat", "label"):
        df = df.drop(columns=[label_col])

    # ---- 2.3 cleaning: inf -> NaN, drop NaN, drop duplicates --------------
    feature_cols = [c for c in df.columns if c not in ("attack_cat", "label")]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df = df.dropna(subset=feature_cols)
    n1 = len(df)
    df = df.drop_duplicates(ignore_index=True)
    n2 = len(df)
    if verbose:
        print(f"Righe con inf/NaN rimosse: {n0 - n1}")
        print(f"Righe duplicate rimosse:   {n1 - n2}")

    # ---- 2.4 null variance columns ----------------------------------------
    if drop_constant_cols:
        const_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            if verbose:
                print(f"Colonne costanti rimosse ({len(const_cols)}): {const_cols}")

    # ---- 2.5 option subsample (dataset has go ~2.8M raws) -----------------
    if subsample is not None:
        strat = _safe_stratify(df["attack_cat"], verbose)
        kwargs = {"random_state": random_state, "stratify": strat}
        if isinstance(subsample, float):
            df, _ = train_test_split(df, train_size=subsample, **kwargs)
        else:
            df, _ = train_test_split(df, train_size=int(subsample), **kwargs)
        df = df.reset_index(drop=True)
        if verbose:
            print(f"Subsampling to shape: {df.shape}")

    # ---- 2.6 stratified train/test split on attack category ---------------
    tr, te = train_test_split(
        df,
        test_size=test_size,
        stratify=_safe_stratify(df["attack_cat"], verbose),
        random_state=random_state,
    )
    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)

    # ---- 2.7 store --------------------------------------------------------
    train_path = _write_split(tr, TRAIN_BASENAME)
    test_path = _write_split(te, TEST_BASENAME)
    print(f"Training set stored in: {train_path} {tr.shape}")
    print(f"Test set stored in:     {test_path} {te.shape}")

    return train_path, test_path


# ===========================================================================
# 3. Loading
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
    # 1. Creates training/testing set only if they don't exist
    train_path = _find_split(TRAIN_BASENAME)
    test_path = _find_split(TEST_BASENAME)

    if force_preprocess or train_path is None or test_path is None:
        if verbose:
            print("Training/test set not found: executing preprocessing...\n")
        train_path, test_path = build_train_test_set(
            download_dataset=download_dataset,
            test_size=test_size,
            random_state=random_state,
            subsample=subsample,
            verbose=verbose,
        )
    elif verbose:
        print(f"Reusing already existing splits:\n  {train_path}\n  {test_path}")

    # 2. Load the dataset
    tr = _read_split(train_path)
    te = _read_split(test_path)

    # 3. Handle Categorical Features with Ordinal Encoding
    categorical_cols = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]

    categorical_cols = [c for c in tr.columns if tr[c].dtype == object or str(tr[c].dtype) == "category"]

    oe = OrdinalEncoder(
        handle_unknown='use_encoded_value',  # for allowing unknown values
        unknown_value=-1,                    # for unknown values
        encoded_missing_value=-1             # for missing values
    )

    tr[categorical_cols] = oe.fit_transform(tr[categorical_cols].astype(str))
    te[categorical_cols] = oe.transform(te[categorical_cols].astype(str))

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
    exclude_columns = ['attack_cat', 'label']
    scaler = StandardScaler()

    tr_features = tr.drop(columns=exclude_columns, errors='ignore')
    numeric_cols = tr_features.select_dtypes(include=['number']).columns

    # fit scaler only on training data, transform both training and testing data
    tr[numeric_cols] = scaler.fit_transform(tr_features[numeric_cols])
    te[numeric_cols] = scaler.transform(te[numeric_cols])

    # 6. Stratify the train/val split (70/30) based on the attack_cat
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
        print("\nStatification on attack categories:\n")
        print(attack_counts)

    # 7. Drop id and attack_cat columns
    def clean(df):
        drop_cols = [
            "id",          # It's just a sequential number
            "attack_cat"   # It's the multi-class version of label
        ]
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_val, X_test = clean(X_train), clean(X_val), clean(X_test)
    print("Dropped 'id' and 'attack_cat' columns")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_cicids(
    download_dataset: bool = False,
    verbose: bool = False,
    non_attackable: list[str] | None = None,
    **kwargs,
):
    """Loads the sets and builds the attack mask."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_train_val_test_set(
        download_dataset=download_dataset, verbose=verbose, **kwargs
    )

    feature_names = list(X_tr.columns)
    non_attackable = CATEGORICAL_COLS if non_attackable is None else non_attackable

    # 1.0 = real features (attackable), 0.0 = categoric features (not attackable)
    attack_mask = torch.tensor(
        [0.0 if c in non_attackable else 1.0 for c in feature_names]
    )

    to_x = lambda df: torch.from_numpy(df.to_numpy(dtype="float32"))
    to_y = lambda s: torch.from_numpy(s.to_numpy()).long()
    return (to_x(X_tr), to_y(y_tr), to_x(X_val), to_y(y_val),
            to_x(X_te), to_y(y_te), feature_names, attack_mask)


if __name__ == "__main__":
    get_train_val_test_set("dataset/cic-ids2017/", download_dataset=True, verbose=True)
