import argparse
from preprocess import get_data
import warnings
import pandas as pd
from test import test

warnings.filterwarnings("ignore")

def main(cfg_proj):
    X_train, y_train, train_uids = get_data(cfg_proj) 
    X_test, test_uids = get_data(cfg_proj, train = 0)
    test(X_train, X_test, y_train, test_uids, cfg_proj)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  

    parser.add_argument("--acoustic", nargs='+', type=str, default = ["Wav2Vec", "MFCC"], required=False) # ["Wav2Vec", "MFCC"]
    parser.add_argument("--model", type=str, default = "SVC", required=False) # MLP, SVC, GBC, RF, LR, NB
    parser.add_argument("--remove_noise", action = "store_true") # False, True
    parser.add_argument("--grid_search", action = "store_true") 
    cfg_proj = parser.parse_args()
    main(cfg_proj)