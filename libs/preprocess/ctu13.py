"""
Columns: StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State,
sTos, dTos, TotPkts, TotBytes, SrcBytes, Label.

Two CTU-13 specific points worth knowing:
  * Source/destination IP addresses are dropped. The botnet hosts have fixed
    addresses inside each scenario.
  * The dataset is extremely imbalanced (botnet flows are roughly 1-2% of the
    total, the rest being mostly background traffic). Use `drop_background=True`
    to keep only Normal + Botnet flows, which is the usual setup in the
    literature since background traffic has no reliable ground truth.
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

try:
    import kagglehub
except ImportError:        # the module stays usable offline
    kagglehub = None

try:
    import pyarrow as _pa
except ImportError:
    _pa = None

if _pa is not None and not getattr(_pa, "_lwad_dedup_patched", False):
    # torch/kagglehub can trigger pandas' pyarrow extension-type registration
    # (pandas.core.arrays.arrow.extension_types) more than once in the same
    # process; the second attempt raises ArrowKeyError even though the type
    # is already correctly registered. Make re-registration a no-op instead.
    _orig_register_extension_type = _pa.register_extension_type

    def _dedup_register_extension_type(ext_type):
        try:
            _orig_register_extension_type(ext_type)
        except _pa.lib.ArrowKeyError:
            pass

    _pa.register_extension_type = _dedup_register_extension_type
    _pa._lwad_dedup_patched = True


DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "ctu13")
KAGGLE_HANDLE = "dhoogla/ctu13"

RAW_SUBDIR = "raw"                         # <dataset_path>/raw/*
TRAIN_BASENAME = "CTU13_training-set"      # extension added when saving
TEST_BASENAME = "CTU13_testing-set"

RAW_EXTENSIONS = (".parquet", ".csv", ".binetflow")

# 1.0 = continuous, attackable feature / 0.0 = categorical, untouchable one.
# These are the CTU-13 counterpart of proto/service/state in UNSW-NB15.
CATEGORICAL_COLS = ["Proto", "Dir", "State"]  # excluded from the FGSM attack

# Ports are numeric but discrete: add them to the mask if you do not want FGSM
# to perturb them, via load_ctu13(non_attackable=CATEGORICAL_COLS + PORT_COLS).
PORT_COLS = ["Sport", "Dport"]

# Columns that must never reach the model: they identify the host or the moment
# of the capture rather than describing the traffic.
IDENTIFIER_COLS = ["StartTime", "SrcAddr", "DstAddr", "LastTime", "Timestamp"]

BOTNET_CLASS = "Botnet"
NORMAL_CLASS = "Normal"
BACKGROUND_CLASS = "Background"


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
    """Read one raw scenario file. Kaggle ships parquet, the original CTU-13 csv."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    # .binetflow files are plain comma separated text
    return pd.read_csv(path, low_memory=False, skipinitialspace=True)


# ===========================================================================
# Helpers: cleaning
# ===========================================================================
def _attack_category(label) -> str:
    """
    Map a CTU-13 label to a coarse class.

    Raw labels look like 'flow=From-Botnet-V42-TCP-Attempt' or
    'flow=Background-Established-cmpgw-CVUT'. Some redistributions replace them
    with a plain 0/1 flag, which is handled here as well.
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


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float32 instead of float64/int64: halves memory on a 20M row dataset."""
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].astype("float32")
    return df


def _safe_stratify(col: pd.Series, verbose: bool = False):
    """train_test_split fails on classes with < 2 samples: skip stratification."""
    counts = col.value_counts()
    if (counts < 2).any():
        if verbose:
            rare = list(counts[counts < 2].index)
            print(f"[warn] classes with fewer than 2 samples {rare}: stratification disabled")
        return None
    return col


def _scenario_from_filename(path: str) -> str:
    """Keep track of which capture every flow comes from (useful for analysis)."""
    return os.path.splitext(os.path.basename(path))[0]


# ===========================================================================
# 1. Download the raw scenario files
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
# 2. Preprocessing: from the 13 scenarios to a training set and a test set
#    (runs once, the result is written to disk and reused afterwards)
# ===========================================================================
def build_train_test_set(
    download_dataset: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
    drop_background: bool = False,
    drop_constant_cols: bool = True,
    subsample: float | int | None = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Merge the raw scenarios, clean them, split into train/test and save both."""
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
            f"No raw scenario file found in '{raw_dir}'. "
            f"Run with download_dataset=True, or manually place the "
            f"'{KAGGLE_HANDLE}' files in that directory."
        )

    # ---- 2.1 read and merge every scenario --------------------------------
    frames = []
    for f in files:
        part = _read_raw(f)
        part.columns = [str(c).strip() for c in part.columns]
        part = part.loc[:, ~part.columns.duplicated()]
        part["scenario"] = _scenario_from_filename(f)

        if verbose:
            print(f"  {os.path.basename(f):<45} {part.shape}")
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    frames.clear()
    if verbose:
        print(f"\nMerged dataset: {df.shape}")

    # ---- 2.2 labels: 'attack_cat' (Botnet/Normal/Background) + binary 'label' --
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(
            f"No 'Label' column found. Available columns: {list(df.columns)}"
        )

    df["attack_cat"] = df[label_col].map(_attack_category)
    df["label"] = (df["attack_cat"] == BOTNET_CLASS).astype("int64")
    if label_col not in ("attack_cat", "label"):
        df = df.drop(columns=[label_col])

    if drop_background:
        # background traffic has no trustworthy ground truth: many papers drop it
        n_before = len(df)
        df = df[df["attack_cat"] != BACKGROUND_CLASS].reset_index(drop=True)
        if verbose:
            print(f"Background flows dropped: {n_before - len(df)}")

    # ---- 2.3 drop host/time identifiers -----------------------------------
    # The botnet hosts keep the same IP inside a scenario, so SrcAddr/DstAddr
    # would leak the label straight into the features.
    id_cols = [c for c in IDENTIFIER_COLS if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
        if verbose:
            print(f"Identifier columns dropped: {id_cols}")

    # ---- 2.4 ports: decimal or hexadecimal ('0x0303' for ICMP) ------------
    for col in PORT_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = _to_port(df[col])

    # ---- 2.5 type of service: NaN is meaningful, encode it as -1 ----------
    for col in ("sTos", "dTos"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    # ---- 2.6 inf -> NaN, drop leftover NaN rows, drop duplicates ----------
    meta_cols = ["attack_cat", "label", "scenario"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    numeric_features = df[feature_cols].select_dtypes(include=["number"]).columns

    df[numeric_features] = df[numeric_features].replace([np.inf, -np.inf], np.nan)
    df = _downcast(df)

    n0 = len(df)
    df = df.dropna(subset=feature_cols)
    n1 = len(df)
    df = df.drop_duplicates(ignore_index=True)
    n2 = len(df)
    if verbose:
        print(f"Rows dropped (inf/NaN):   {n0 - n1}")
        print(f"Duplicate rows dropped:   {n1 - n2}")

    # ---- 2.7 constant columns carry no information ------------------------
    if drop_constant_cols:
        const_cols = [c for c in feature_cols
                      if c in df.columns and df[c].nunique(dropna=False) <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            if verbose:
                print(f"Constant columns dropped ({len(const_cols)}): {const_cols}")

    # low cardinality strings: category dtype keeps memory under control
    for col in CATEGORICAL_COLS + ["attack_cat", "scenario"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")

    # ---- 2.8 optional subsampling (the full dataset has ~20M flows) -------
    if subsample is not None:
        strat = _safe_stratify(df["attack_cat"], verbose)
        train_size = subsample if isinstance(subsample, float) else int(subsample)
        df, _ = train_test_split(
            df, train_size=train_size, stratify=strat, random_state=random_state
        )
        df = df.reset_index(drop=True)
        if verbose:
            print(f"Subsampled to: {df.shape}")

    if verbose:
        dist = df["attack_cat"].value_counts()
        print("\nClass distribution before splitting:")
        print((pd.concat([dist, (100 * dist / len(df)).round(3)], axis=1,
                         keys=["flows", "%"])).to_string())

    # ---- 2.9 stratified train/test split on the coarse class --------------
    tr, te = train_test_split(
        df,
        test_size=test_size,
        stratify=_safe_stratify(df["attack_cat"], verbose),
        random_state=random_state,
    )
    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)

    # ---- 2.10 save both sets for later reuse ------------------------------
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
    drop_background: bool = False,
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
            drop_background=drop_background,
            subsample=subsample,
            verbose=verbose,
        )
    elif verbose:
        print(f"Reusing the existing splits:\n  {train_path}\n  {test_path}")

    # 2. Load the dataset
    tr = _read_split(train_path)
    te = _read_split(test_path)

    # 3. Handle categorical features with ordinal encoding
    #    (Proto, Dir, State, plus attack_cat and the scenario tag)
    # anything non numeric: string/object/category, depending on the pandas
    # version and on whether the split was reloaded from parquet or csv
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
    exclude_columns = ['attack_cat', 'label', 'scenario']
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
        print(f"\nBotnet flows in the training set: {pos} "
              f"({100 * pos / max(len(y_train), 1):.3f}%) -- consider class "
              f"weights or resampling, CTU-13 is heavily imbalanced")

    # 7. Drop id, scenario and attack_cat columns
    def clean(df):
        drop_cols = [
            "id",          # it is just a sequential number
            "scenario",    # capture tag, not a traffic feature
            "attack_cat"   # it is the multi-class version of label
        ]
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_val, X_test = clean(X_train), clean(X_val), clean(X_test)
    print("Dropped 'id', 'scenario' and 'attack_cat' columns")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ===========================================================================
# Data: CTU-13 through the functions of this module
# ===========================================================================
def load_ctu13(
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
    get_train_val_test_set("dataset/ctu13/", download_dataset=True, verbose=True)
