import kagglehub
import os

import pandas as pd

from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = "../dataset/unsw-nb15/"

def get_train_val_test_set(download_dataset=False, verbose=False):
    # 1. Download latest version
    if download_dataset:
        path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15", output_dir="../dataset/unsw-nb15")
        
        # Training Set and Testing Set are inverted, so revert them
        os.rename(DATASET_PATH + "UNSW_NB15_training-set.csv",  DATASET_PATH + "temp.csv")
        os.rename(DATASET_PATH + "UNSW_NB15_testing-set.csv",   DATASET_PATH + "UNSW_NB15_training-set.csv")
        os.rename(DATASET_PATH + "temp.csv",                    DATASET_PATH + "UNSW_NB15_testing-set.csv")

        print("Path to dataset files:", path)


    # 2. Load the dataset
    tr = pd.read_csv(DATASET_PATH + "UNSW_NB15_training-set.csv")
    te = pd.read_csv(DATASET_PATH + "UNSW_NB15_testing-set.csv")


    # 3. Handle Categorical Features with Label Encoding
    categorical_cols = ['proto', 'service', 'state', 'attack_cat']  # based on dataset
    for col in categorical_cols:
        if col in tr.columns:
            le = LabelEncoder()
            tr[col] = le.fit_transform(tr[col].astype(str))
            te[col] = le.fit_transform(te[col].astype(str))
    

    # 4. Handle missing values replacing them with the column name (as string)
    for col in tr.columns:
        if tr[col].isnull().any():
            tr[col].fillna(col, inplace=True)
    for col in te.columns:
        if te[col].isnull().any():
            te[col].fillna(col, inplace=True)
    

    # 5. Standardize numerical features
    exclude_columns = ['attack_cat', 'label']
    tr_features = tr.drop(columns=exclude_columns, errors='ignore')

    numeric_cols = tr_features.select_dtypes(include=['number']).columns
    scaler = StandardScaler()

    tr_features_scaled = tr_features.copy()

    # fit scaler only on training data
    # transform both training and testing data
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
    

    return X_train, y_train, X_val, y_val, X_test, y_test