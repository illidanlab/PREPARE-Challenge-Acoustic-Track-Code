import argparse
from preprocess import get_data
import warnings
import pandas as pd
from test import test

warnings.filterwarnings("ignore")

def main(cfg_proj):
    X_train, y_train, train_uids = get_data(cfg_proj) 
    X_test, test_uids = get_data(cfg_proj, train = 0)
    return test(X_train, X_test, y_train, test_uids, cfg_proj)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  

    parser.add_argument("--acoustic", nargs='+', type=str, default = ["Wav2Vec", "MFCC"], required=False) # ["Wav2Vec", "Hubert", "MFCC", "eGeMAPS", "BERT"]
    parser.add_argument("--model", type=str, default = "SVC", required=False) # MLP, SVC, GBC, MLP, RF, LR
    parser.add_argument("--remove_alexa", type = bool, default=True) # False, True
    parser.add_argument("--remove_zh", type = bool, default=True) # False, True
    parser.add_argument("--remove_gl", type = bool, default=True) # False, True
    parser.add_argument("--remove_noise", type = bool, default=True) # False, True
    parser.add_argument("--outlier_detection", type = str, default="svm") # svm

    cfg_proj = parser.parse_args()
    main(cfg_proj)