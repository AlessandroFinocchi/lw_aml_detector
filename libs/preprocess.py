import kagglehub
import torch
import os

import pandas as pd

from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["proto", "service", "state"]  # escluse dall'attacco FGSM


def get_train_val_test_set(dataset_path, download_dataset=False, verbose=False) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # 1. Download latest version
    if download_dataset:
        path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15", output_dir="../dataset/unsw-nb15")
        
        # Training Set and Testing Set are inverted, so revert them
        os.rename(dataset_path + "UNSW_NB15_training-set.csv",  dataset_path + "temp.csv")
        os.rename(dataset_path + "UNSW_NB15_testing-set.csv",   dataset_path + "UNSW_NB15_training-set.csv")
        os.rename(dataset_path + "temp.csv",                    dataset_path + "UNSW_NB15_testing-set.csv")

        print("Path to dataset files:", path)


    # 2. Load the dataset
    tr = pd.read_csv(dataset_path + "UNSW_NB15_training-set.csv")
    te = pd.read_csv(dataset_path + "UNSW_NB15_testing-set.csv")


    # 3. Handle Categorical Features with Ordinal Encoding
    categorical_cols = ['proto', 'service', 'state', 'attack_cat']  # based on dataset

    oe = OrdinalEncoder(
        handle_unknown='use_encoded_value', # for allowing unknown values
        unknown_value=-1,                   # for unknown values
        encoded_missing_value=-1            # for missing values
    )
    
    tr[categorical_cols] = oe.fit_transform(tr[categorical_cols].astype(str))
    te[categorical_cols] = oe.transform(te[categorical_cols].astype(str))
    

    # 4. Handle missing values replacing them with the column name (as string)
    for col in tr.columns:
        if tr[col].isnull().any():
            tr[col].fillna(col, inplace=True)
    for col in te.columns:
        if te[col].isnull().any():
            te[col].fillna(col, inplace=True)
    

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
    val_size = 0.3

    X = tr.drop(columns=['label'])
    y = tr['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=strat_col, random_state=42
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


# ===========================================================================
# Dati: UNSW-NB15 tramite il tuo preprocess.py
# ===========================================================================
def load_unsw(dataset_path: str, download_dataset=False, verbose=False):
    """Carica i set da preprocess.py e costruisce la maschera d'attacco."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_train_val_test_set(
        dataset_path, download_dataset=False, verbose=False
    )

    feature_names = list(X_tr.columns)

    # 1.0 = feature continua attaccabile, 0.0 = categorica intoccabile
    attack_mask = torch.tensor(
        [0.0 if c in CATEGORICAL_COLS else 1.0 for c in feature_names]
    )

    to_x = lambda df: torch.from_numpy(df.to_numpy(dtype="float32"))
    to_y = lambda s: torch.from_numpy(s.to_numpy()).long()
    return (to_x(X_tr), to_y(y_tr), to_x(X_val), to_y(y_val),
            to_x(X_te), to_y(y_te), feature_names, attack_mask)

if __name__ == "__main__":
    get_train_val_test_set("dataset/unsw-nb15/")