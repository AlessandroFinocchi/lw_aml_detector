from enum import Enum

from libs.preprocess import unsw_bw15, cicids2017, ctu13, csecicids2018


class KaggleDataset(Enum):
    UNSW_BW15 = "unsw-nb15"
    CICIDS2017 = "cicids2017"
    CTU13 = "ctu13"
    CSECICIDS2018 = "cseecicids2017"

def load_dataset(kd: KaggleDataset, download_dataset=False, verbose=False):
    match kd:
        case KaggleDataset.UNSW_BW15: 
            return unsw_bw15.load_unsw(download_dataset, verbose)
        case KaggleDataset.CICIDS2017:
            return cicids2017.load_cicids(download_dataset, verbose)
        case KaggleDataset.CTU13:
            return ctu13.load_ctu13(download_dataset, verbose)
        case KaggleDataset.CSECICIDS2018:
            return csecicids2018.load_csecicids2018(download_dataset, verbose)
        case _:
            raise ValueError("Not a valid KaggleDataset instance")

def get_categorical_cols(kd: KaggleDataset):
    match kd:
        case KaggleDataset.UNSW_BW15: 
            return unsw_bw15.get_categorical_cols()
        case KaggleDataset.CICIDS2017:
            return cicids2017.get_categorical_cols()
        case KaggleDataset.CTU13:
            return ctu13.get_categorical_cols()
        case KaggleDataset.CSECICIDS2018:
            return csecicids2018.get_categorical_cols()
        case _:
            raise ValueError("Not a valid KaggleDataset instance")

def get_train_val_test_set(kd: KaggleDataset, download_dataset=False, verbose=False, **kwargs): 
    match kd:
        case KaggleDataset.UNSW_BW15: 
            return unsw_bw15.get_train_val_test_set(download_dataset, verbose)
        case KaggleDataset.CICIDS2017:
            return cicids2017.get_train_val_test_set(download_dataset, verbose)
        case KaggleDataset.CTU13:
            return ctu13.get_train_val_test_set(download_dataset, verbose)
        case KaggleDataset.CSECICIDS2018:
            return csecicids2018.get_train_val_test_set(download_dataset, verbose)
        case _:
            raise ValueError("Not a valid KaggleDataset instance")
            
