"""
Shared utilities behind the CIC-IDS 2017, CTU-13 and CSE-CIC-IDS2018 loaders.

Those three datasets are distributed as a bunch of raw capture files instead of
a ready made train/test split, so they all need the same pipeline:

    download -> clean each file -> merge -> split train/test-> save -> 
    reload -> split again into train/val -> tensors

Everything in that chain is identical from one dataset to the next except one
step: how a single raw file is turned into a clean numeric table. Each dataset
module therefore provides just that step, as a `clean_file` callback, plus a
`DatasetConfig` describing where its data lives.

--------------------------------------------------------------------------
The `clean_file(path, verbose)` contract
--------------------------------------------------------------------------
Read one raw capture file and return a DataFrame that has:

  * one column per feature, all numeric;
  * `attack_cat`, the multi-class label, as a string;
  * `label`, the binary label, 0 for benign traffic and 1 for an attack;
  * no identifier column left (IP addresses, flow ids, timestamps), because
    those either leak the answer or make duplicate removal useless.

After that downcasting, deduplication, subsampling, schema alignment, 
merging, splitting and saving.

Sizing knobs, both optional:
  * `undersample`    keeps a stratified fraction of each daily file during the
                     preprocessing, bounding peak memory but preserving the
                     original class ratios.
  * `max_per_class` / `majority_ratio` undersample the over-represented classes
                     at load time, shrinking the data AND rebalancing it. They
                     can be changed between runs without redoing preprocessing.
"""

from __future__ import annotations

import os
import glob
import shutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:                       # only needed to download a dataset
    import kagglehub
except ImportError:        # the modules stay usable offline
    kagglehub = None


RAW_SUBDIR = "raw"

# Bookkeeping columns: never features, never scaled, dropped before the tensors.
# 'scenario' and 'capture' are the names earlier versions used for 'source';
# they are listed so splits saved back then still load.
META_COLS = ("attack_cat", "label", "source")
NON_FEATURE_COLS = ("id", "source", "scenario", "capture", "attack_cat")


@dataclass
class DatasetConfig:
    """Everything the shared pipeline needs to know about one dataset."""
    name: str                                  # human readable, used in prints
    kaggle_handle: str                         # e.g. "dhoogla/ctu13"
    dataset_path: str
    train_basename: str                        # file name, extension added later
    test_basename: str
    clean_file: Callable[..., pd.DataFrame]    # see the contract above
    categorical_cols: list[str] = field(default_factory=list)  # untouchable by FGSM
    raw_extensions: tuple[str, ...] = (".parquet", ".csv")


# ===========================================================================
# Splits methods
# ===========================================================================
def find_split(dataset_path: str, basename: str) -> str | None:
    """Return the path of an already saved split (parquet or csv), else None."""
    for ext in (".parquet", ".csv"):
        path = os.path.join(dataset_path, basename + ext)
        if os.path.exists(path):
            return path
    return None


def read_split(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def write_split(df: pd.DataFrame, dataset_path: str, basename: str) -> str:
    """Save as parquet, fall back to csv wheather without pyarrow."""
    parquet_path = os.path.join(dataset_path, basename + ".parquet")
    csv_path = os.path.join(dataset_path, basename + ".csv")
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


# ===========================================================================
# Raw files: download, list, read
# ===========================================================================
def download_raw(kaggle_handle: str, raw_dir: str,
                 extensions: tuple[str, ...] = (".parquet", ".csv")) -> str:
    if kagglehub is None:
        raise ImportError("kagglehub is not installed: `pip install kagglehub`")

    os.makedirs(raw_dir, exist_ok=True)

    path = kagglehub.dataset_download(kaggle_handle, output_dir=raw_dir)

    # normalise the layout: every raw file ends up in <dataset_path>/raw/
    if os.path.abspath(path) != os.path.abspath(raw_dir):
        for ext in extensions:
            for src in glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True):
                dst = os.path.join(raw_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    print("Path to dataset files:", path)
    return raw_dir


def list_raw_files(raw_dir: str, extensions: tuple[str, ...]) -> list[str]:
    files: list[str] = []
    for ext in extensions:
        files += glob.glob(os.path.join(raw_dir, "**", f"*{ext}"), recursive=True)
    return sorted(set(files))


def read_raw(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Read one raw capture in parquet or csv format."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, skipinitialspace=True, encoding=encoding)


# ===========================================================================
# Helpers
# ===========================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim the whitespace these exports put around column names and drop repeated
    columns, including the ones pandas renames to 'X.1' when a header appears
    twice in the same file.
    """
    df.columns = [str(c).strip() for c in df.columns]
    renamed = [c for c in df.columns if c.endswith(".1") and c[:-2] in df.columns]
    if renamed:
        df = df.drop(columns=renamed)
    return df.loc[:, ~df.columns.duplicated()]


def find_label_col(df: pd.DataFrame) -> str:
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(f"No 'Label' column found. Available columns: {list(df.columns)}")
    return label_col


def drop_if_present(df: pd.DataFrame, cols, what: str, verbose: bool = False) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if present:
        df = df.drop(columns=present)
        if verbose:
            print(f"    {what} dropped: {present}")
    return df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float32 instead of float64/int64: halves memory on the bigger datasets."""
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].astype("float32")
    return df


def source_from_filename(path: str) -> str:
    """Tag every row with the capture it came from, for per-capture analysis."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("_TrafficForML_CICFlowMeter", "")


# ===========================================================================
# Sampling
# ===========================================================================
def safe_stratify(col: pd.Series, verbose: bool = False):
    """train_test_split fails on classes with less than 2 samples: no stratification."""
    counts = col.value_counts()
    if (counts < 2).any():
        if verbose:
            rare = list(counts[counts < 2].index)
            print(f"[warn] classes with fewer than 2 samples {rare}: stratification disabled")
        return None
    return col


def undersample(
    df: pd.DataFrame,
    max_per_class: int | None = None,
    majority_ratio: float | None = None,
    label_col: str = "attack_cat",
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Shrink a set by dropping rows from the over-represented classes.

    max_per_class   every class keeps at most this many rows. Classes that are
                    already smaller are left untouched, so the rare attacks
                    survive instead of being sampled away.
    majority_ratio  the biggest class keeps at most ratio * (all other rows)

    The two can be combined. Passing neither returns the frame unchanged.
    Sampling is without replacement and the result is shuffled, so the rows do
    not come out grouped by class.
    """
    if max_per_class is None and majority_ratio is None:
        return df

    counts = df[label_col].value_counts()
    counts = counts[counts > 0]          # categorical dtypes report empty classes
    caps = counts.to_dict()

    if majority_ratio is not None:
        majority = counts.index[0]       # value_counts is sorted descending
        others = int(counts.sum() - counts.iloc[0])
        caps[majority] = min(caps[majority], max(int(majority_ratio * others), 1))

    if max_per_class is not None:
        caps = {cls: min(n, int(max_per_class)) for cls, n in caps.items()}

    rng = np.random.default_rng(random_state)
    kept = []
    for cls, cap in caps.items():
        idx = df.index[df[label_col] == cls].to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        kept.append(idx)

    out = df.loc[np.concatenate(kept)]
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if verbose:
        print(f"\nUndersampling: {len(df)} -> {len(out)} rows "
              f"({100 * len(out) / max(len(df), 1):.2f}% kept)")
        print(out[label_col].value_counts().to_string())

    return out


def _subsample_file(df: pd.DataFrame, subsample, random_state: int) -> pd.DataFrame:
    """Keep a stratified slice of one capture, to bound peak memory on merge."""
    if subsample is None or len(df) < 2:
        return df

    train_size = subsample if isinstance(subsample, float) else min(int(subsample), len(df) - 1)
    part, _ = train_test_split(
        df, train_size=train_size,
        stratify=safe_stratify(df["attack_cat"]), 
        random_state=random_state,
    )
    return part.reset_index(drop=True)


# ===========================================================================
# Preprocessing: storing train and test set on persistance layer
# ===========================================================================
def _merge_aligned(frames: list[pd.DataFrame], verbose: bool = False) -> pd.DataFrame:
    """
    Concatenate the captures on the columns they all share.

    A column present in only some files would silently become all-NaN after the
    concat and then wipe out the whole dataset at the dropna below, so mismatches
    are resolved explicitly and reported.
    """
    common = set(frames[0].columns)
    for part in frames[1:]:
        common &= set(part.columns)
    ordered = [c for c in frames[0].columns if c in common]

    extra = sorted(set().union(*(set(p.columns) for p in frames)) - common)
    if extra:
        print(f"[warn] columns missing from at least one capture, dropped: {extra}")

    df = pd.concat([part[ordered] for part in frames], ignore_index=True)
    frames.clear()
    if verbose:
        print(f"\nMerged dataset: {df.shape}")
    return df


def _clean_merged(df: pd.DataFrame, drop_constant_cols: bool, verbose: bool) -> pd.DataFrame:
    """inf -> NaN, drop what is left broken, drop duplicates and dead columns."""
    feature_cols = [c for c in df.columns if c not in META_COLS]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df = df.dropna(subset=feature_cols)
    n1 = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["attack_cat"], ignore_index=True)
    n2 = len(df)
    if verbose:
        print(f"Rows dropped (inf/NaN):   {n0 - n1}")
        print(f"Cross-capture duplicates: {n1 - n2}")

    if drop_constant_cols:
        const = [c for c in feature_cols
                 if c in df.columns and df[c].nunique(dropna=False) <= 1]
        if const:
            df = df.drop(columns=const)
            if verbose:
                print(f"Constant columns dropped ({len(const)}): {const}")

    # low cardinality strings: category dtype keeps memory under control
    for col in ("attack_cat", "source"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")

    if verbose:
        dist = df["attack_cat"].value_counts()
        print("\nClass distribution before splitting:")
        print(pd.concat([dist, (100 * dist / len(df)).round(3)],
                        axis=1, keys=["flows", "%"]).to_string())
    return df


def build_train_test_set(
    config: DatasetConfig,
    dataset_path: str,
    download_dataset: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
    drop_constant_cols: bool = True,
    subsample: float | int | None = None,
    verbose: bool = False,
    clean_file: Callable[..., pd.DataFrame] | None = None,
) -> tuple[str, str]:
    """
    Merge the raw captures, clean them, split into train/test and save both.

    `subsample` is applied per capture file rather than on the merged frame, so
    peak memory stays bounded. A float is the fraction kept from each file, an
    int the maximum number of rows kept from each file; either way the sampling
    is stratified on that file's own labels.
    """
    clean_file = clean_file or config.clean_file
    os.makedirs(dataset_path, exist_ok=True)
    raw_dir = os.path.join(dataset_path, RAW_SUBDIR)

    if download_dataset:
        download_raw(config.kaggle_handle, raw_dir, config.raw_extensions)

    files = list_raw_files(raw_dir, config.raw_extensions)
    if not files:
        raise FileNotFoundError(
            f"No raw capture file found in '{raw_dir}'. Run with "
            f"download_dataset=True, or manually place the "
            f"'{config.kaggle_handle}' files in that directory."
        )

    frames = []
    for path in files:
        if verbose:
            print(f"  {os.path.basename(path)}")

        part = clean_file(path, verbose)
        missing = {"attack_cat", "label"} - set(part.columns)
        if missing:
            raise KeyError(f"{os.path.basename(path)}: clean_file did not produce {missing}")

        part = downcast(part)

        # deduplicate here, while the frame is still small: most duplicates are
        # internal to one capture and this keeps the merged frame much lighter
        n_before = len(part)
        part = part.drop_duplicates(ignore_index=True)
        if verbose and n_before != len(part):
            print(f"    duplicate rows dropped: {n_before - len(part)}")

        part = _subsample_file(part, subsample, random_state)
        part["source"] = source_from_filename(path)
        if verbose:
            print(f"    kept: {part.shape}")
        frames.append(part)

    df = _clean_merged(_merge_aligned(frames, verbose), drop_constant_cols, verbose)

    tr, te = train_test_split(
        df, test_size=test_size,
        stratify=safe_stratify(df["attack_cat"], verbose), 
        random_state=random_state,
    )
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

    train_path = write_split(tr, dataset_path, config.train_basename)
    test_path = write_split(te, dataset_path, config.test_basename)
    print(f"Training set saved to: {train_path} {tr.shape}")
    print(f"Test set saved to:     {test_path} {te.shape}")

    return train_path, test_path


# ===========================================================================
# Loading: the saved splits -> X_train, y_train, X_val, y_val, X_test, y_test
# ===========================================================================
def get_train_val_test_set(
    config: DatasetConfig,
    download_dataset: bool = False,
    verbose: bool = False,
    *,
    force_preprocess: bool = False,
    val_size: float = 0.3,
    test_size: float = 0.3,
    random_state: int = 42,
    subsample: float | int | None = None,
    max_per_class: int | None = None,
    majority_ratio: float | None = None,
    test_max_rows: int | None = None,
    clean_file: Callable[..., pd.DataFrame] | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # 1. Build the training/test sets only if they do not exist yet
    dataset_path = config.dataset_path
    train_path = find_split(dataset_path, config.train_basename)
    test_path = find_split(dataset_path, config.test_basename)

    if force_preprocess or train_path is None or test_path is None:
        if verbose:
            print(f"{config.name}: training/test sets not found, running the preprocessing...\n")
        train_path, test_path = build_train_test_set(
            config, dataset_path,
            download_dataset=download_dataset, test_size=test_size,
            random_state=random_state, subsample=subsample, verbose=verbose,
            clean_file=clean_file,
        )
    elif verbose:
        print(f"Reusing the existing splits:\n  {train_path}\n  {test_path}")

    # 2. Load the dataset
    tr = read_split(train_path)
    te = read_split(test_path)

    # 2b. Optional undersampling. This is where the size of a single experiment
    #     is decided, so it can be changed without redoing the preprocessing.
    #     Only the training set is rebalanced: the test set is merely capped in
    #     size, so the reported metrics keep reflecting the real class ratios.
    tr = undersample(tr, max_per_class=max_per_class, majority_ratio=majority_ratio,
                     random_state=random_state, verbose=verbose)

    if test_max_rows is not None and len(te) > test_max_rows:
        te = te.sample(n=int(test_max_rows), random_state=random_state).reset_index(drop=True)
        if verbose:
            print(f"\nTest set capped at {len(te)} rows (class ratios preserved)")

    # 3. Handle categorical features with ordinal encoding.
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
    scaler = StandardScaler()

    tr_features = tr.drop(columns=list(NON_FEATURE_COLS) + ["label"], errors='ignore')
    numeric_cols = tr_features.select_dtypes(include=['number']).columns

    # fit the scaler only on the training data, transform both sets
    tr[numeric_cols] = scaler.fit_transform(tr_features[numeric_cols])
    te[numeric_cols] = scaler.transform(te[numeric_cols])

    # 6. Stratify the train/val split based on attack_cat
    strat_col = tr['attack_cat']

    X = tr.drop(columns=['label'])
    y = tr['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size,
        stratify=safe_stratify(strat_col, verbose), random_state=random_state,
    )

    X_test = te.drop(columns=['label'])
    y_test = te['label']

    if verbose:
        _report(X_train, X_val, X_test, y_train, attack_names)

    # 7. Drop the bookkeeping columns, keeping only the features
    clean = lambda df: df.drop(
        columns=[c for c in NON_FEATURE_COLS if c in df.columns]
    )
    X_train, X_val, X_test = clean(X_train), clean(X_val), clean(X_test)
    print(f"Dropped the bookkeeping columns {NON_FEATURE_COLS}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def _report(X_train, X_val, X_test, y_train, attack_names) -> None:
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


# ===========================================================================
# Tensors and the FGSM attack mask
# ===========================================================================
def load_tensors(
    config: DatasetConfig,
    download_dataset: bool = False,
    verbose: bool = False,
    non_attackable: list[str] | None = None,
    **kwargs,
):
    """Load the three sets as tensors and build the attack mask."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_train_val_test_set(
        config, download_dataset=download_dataset, verbose=verbose, **kwargs
    )

    feature_names = list(X_tr.columns)
    non_attackable = config.categorical_cols if non_attackable is None else non_attackable

    # 1.0 = continuous attackable feature, 0.0 = untouchable categorical one
    attack_mask = torch.tensor(
        [0.0 if c in non_attackable else 1.0 for c in feature_names]
    )

    to_x = lambda df: torch.from_numpy(df.to_numpy(dtype="float32"))
    to_y = lambda s: torch.from_numpy(s.to_numpy()).long()
    return (to_x(X_tr), to_y(y_tr), to_x(X_val), to_y(y_val),
            to_x(X_te), to_y(y_te), feature_names, attack_mask)
