import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import argparse
from preprocess import get_data
import warnings
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

# Add the parent directory to the system path
sys.path.append(parent_dir)


def get_group(test_uids, cfg_proj):
    if cfg_proj.group == "age":
        meta_df = pd.read_csv("data/metadata.csv")
        ages = []
        for uid in test_uids:
            ages.append(int(meta_df[meta_df["uid"] == uid]["age"]))

        group_index = []
        groupname2id = {"49-70":0, "71-80":1, "81-99":2}
        groupid2name = {0:"49-70", 1:"71-80", 2:"81-99"}

        for age in ages:
            for name in groupname2id:
                lower = int(name[:2])
                upper = int(name[-2:])
                if lower <= age and age <= upper:
                    group_index.append(groupname2id[name])
    
    if cfg_proj.group == "gender":
        meta_df = pd.read_csv("data/metadata.csv")
        genders = []
    
        for uid in test_uids:
            genders.append(str(meta_df[meta_df["uid"] == uid]["gender"].values[0]))
        
        group_index = []
        groupname2id = {"male":0, "female":1}
        groupid2name = {0:"male", 1:"female"}

        for gender in genders:
            for name in groupname2id:
                if gender == name:
                    group_index.append(groupname2id[name])

    if cfg_proj.group == "language":
        transcript_df = pd.read_csv("data/transcripts.csv")
        lans = []
        for uid in test_uids:
            lans.append(str(transcript_df[transcript_df["uid"] == uid]["language"].values[0]))

        group_index = []
        groupname2id = {"en":0, "es":1, "zh":2, "gl":3}
        groupid2name = {0:"en", 1:"es", 2:"zh", 3:"gl"}

        for lan in lans:
            for name in groupname2id:
                if lan == name:
                    group_index.append(groupname2id[name])

    if cfg_proj.group == "cognitive":
        cognitives = np.array(pd.read_csv("data/acoustic_test_labels.csv").iloc[:, 1:])

        group_index = []

        groupname2id = {"diagnosis_control":0, "diagnosis_mci":1 ,"diagnosis_adrd":2}
        groupid2name = {0: "diagnosis_control", 1: "diagnosis_mci", 2: "diagnosis_adrd"}

        for cognitive in cognitives:
            for i in range(3):
                if cognitive[i] == 1.0:
                    group_index.append(i)

    if cfg_proj.group == "length":
        with open("features/length.pkl", "rb") as f:
            uid2length = pickle.load(f)

        lengths = []
        for uid in test_uids:
            lengths.append(uid2length[uid])

        group_index = []
        groupname2id = {"10-25":0, "26-30":1}
        groupid2name = {0:"10-25", 1:"26-30"}

        for length in lengths:
            for name in groupname2id:
                lower = int(name[:2])
                upper = int(name[-2:])
                if lower <= length and length <= upper:
                    group_index.append(groupname2id[name])

    return group_index, groupid2name

def main(cfg_proj):
    X_train, y_train, train_uids = get_data(cfg_proj) 
    X_test, test_uids = get_data(cfg_proj, train = 0)
    
    group_index, groupid2name = get_group(test_uids, cfg_proj)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    svm = OneClassSVM(gamma='auto', nu=0.1)
    svm.fit(X_train)
    outliers = svm.predict(X_train)

    X_train = X_train[outliers == 1]
    y_train = np.array(y_train)[outliers == 1]
    
    if cfg_proj.model == "SVC":
        best_parameters = {'C': 1, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
        classifier = SVC(probability=True, max_iter=1000000, random_state = 42, **best_parameters)
    elif cfg_proj.model == "RF":
        best_parameters = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}
        classifier = RandomForestClassifier(random_state=42, **best_parameters)
    classifier.fit(X_train, y_train)    

    pred = classifier.predict(X_test)

    cognitives = np.array(pd.read_csv("data/acoustic_test_labels.csv").iloc[:, 1:])
    y_true = []

    for cognitive in cognitives:
        for i in range(3):
            if cognitive[i] == 1.0:
                y_true.append(i)

    y_true = np.array(y_true)

    print(cfg_proj.group)
    for group in np.unique(group_index):
        print("Group name:", groupid2name[group], " Number of samples:", np.sum(group_index==group), " Accuracy:", f"{accuracy_score(y_true[group_index == group], pred[group_index == group]):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  

    parser.add_argument("--acoustic", nargs='+', type=str, default = ["Wav2Vec", "MFCC"], required=False) # ["Wav2Vec", "MFCC"]
    parser.add_argument("--group", nargs='+', type=str, default = "length", required=False) # age, gender, language, length, cognitive
    parser.add_argument("--model", type=str, default = "RF", required=False) #  SVC, RF
    parser.add_argument("--remove_noise", action = "store_true") # False, True
    parser.add_argument("--grid_search", action = "store_true") 
    cfg_proj = parser.parse_args()
    main(cfg_proj)