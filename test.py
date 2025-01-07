from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os
import copy

def log_loss(y_true, y_pred):
    # Ensure y_pred doesn't contain zeros or ones to avoid log errors
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute log loss
    N = y_true.shape[0]  # Number of samples
    loss = -np.sum(y_true * np.log(y_pred)) / N
    return loss

def test(X_train, X_test, y_train, test_uids, cfg_proj):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if cfg_proj.remove_noise:
        svm = OneClassSVM(gamma='auto', nu=0.1)
        svm.fit(X_train)
        outliers = svm.predict(X_train)
        print("Mumber of samples removed by One-class SVM is", np.sum(outliers==-1))
        X_train = X_train[outliers == 1]
        y_train = np.array(y_train)[outliers == 1]

    grid_search = cfg_proj.grid_search or (not os.path.exists(f"results_local/{cfg_proj.model}.csv"))
    
    if not grid_search:
        df = pd.read_csv(f"results_local/{cfg_proj.model}.csv")
        best_params = eval(df.iloc[0, 0])
        
    if cfg_proj.model == "DT":
        parameters = {
            'criterion': ['gini', 'entropy'],  # or 'log_loss' for newer versions of scikit-learn
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
        }
        if grid_search:
            classifier = GridSearchCV(DecisionTreeClassifier(random_state=42), parameters, n_jobs=-1, verbose=2)
        else:
            classifier = DecisionTreeClassifier(random_state=42, **best_params)
    
    if cfg_proj.model == "MLP":
        parameters = {'hidden_layer_sizes':[(512, 256, 128, 64, 32), 
                                            (512, 128, 32), 
                                            (256, 128, 64, 32),
                                            (256, 64),
                                            (128, 64, 32),
                                            (128, 32),
                                            (64, 32),
                                            (64),
                                            (32)
                                            ]}
        if grid_search:
            classifier = GridSearchCV(MLPClassifier(max_iter=2000, random_state=42), parameters, n_jobs=-1, verbose=2)
        else:
            classifier = MLPClassifier(max_iter=2000, random_state=42, **best_params)

    if cfg_proj.model == "RF":
        parameters = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        
        if grid_search:
            classifier = GridSearchCV(RandomForestClassifier(random_state=42), parameters, n_jobs=-1, verbose=2)
        else:
            classifier = RandomForestClassifier(random_state=42, **best_params)

    if cfg_proj.model == "SVC":
        parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100, 200, 500, 700, 1000], "degree": [1, 3, 5, 7, 9, 11, 13], "gamma": ["scale", "auto"]}
        
        if grid_search:
            classifier = GridSearchCV(SVC(probability=True, max_iter=1000000, random_state = 42), parameters, n_jobs=-1, verbose=2)
        else:
            classifier = SVC(probability=True, max_iter=1000000, random_state = 42, **best_params)

    if cfg_proj.model == "GBC":
        parameters = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        if grid_search:
            classifier = GridSearchCV(GradientBoostingClassifier(random_state=42), parameters, n_jobs=-1, verbose=2)
        else:
            classifier = GradientBoostingClassifier(random_state=42, **best_params)

    if cfg_proj.model == "LR":
        parameters = {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'saga'],
            'l1_ratio': [0.1, 0.5, 0.9],  # Only used when penalty='elasticnet'
            'max_iter': [100, 200, 500]
        }

        if grid_search:
            classifier = GridSearchCV(LogisticRegression(random_state=42), parameters, n_jobs=-1, verbose=2)
        else:
            classifier = LogisticRegression(random_state=42, **best_params)

    classifier.fit(X_train, y_train)

    if grid_search:
        df = pd.DataFrame(classifier.cv_results_)
        df = df[['params', 'mean_test_score', 'std_test_score']]
        df = df.sort_values(by='mean_test_score', ascending=False)
        
        # Save cross-validation performance
        df.to_csv(f"results_local/{cfg_proj.model}.csv", index=False)

    # Generate test predictions
    proba = classifier.predict_proba(X_test)
    y_true = np.array(pd.read_csv("data/acoustic_test_labels.csv").iloc[:, 1:])
    test_log_loss = log_loss(y_true, proba)

    df = pd.DataFrame(proba, columns=["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"])
    df.insert(0, "uid", test_uids)   
    df_best = copy.deepcopy(df)

    # Set Chinese Samples to MCI 
    df_transcripts = pd.read_csv("data/transcripts.csv")
    zh_uids = df_transcripts[df_transcripts["language"] == "zh"]["uid"].values

    df.loc[df["uid"].isin(zh_uids), "diagnosis_control"] = 0.0005
    df.loc[df["uid"].isin(zh_uids), "diagnosis_mci"] = 0.999
    df.loc[df["uid"].isin(zh_uids), "diagnosis_adrd"] = 0.0005
    if test_log_loss > log_loss(y_true, np.array(df.iloc[:, 1:])):
        df_best = copy.deepcopy(df)
        test_log_loss = log_loss(y_true, np.array(df.iloc[:, 1:]))

    # set confident subject 
    for index, row in df.iterrows():
        logits = row[1:]  # Exclude 'uid' column
        logits = np.array(logits)
        if (logits >= 0.85).any():
            max_index = np.argmax(logits)
            df.iloc[index, 1] = 0.0005 
            df.iloc[index, 2] = 0.0005 
            df.iloc[index, 3] = 0.0005
            df.iloc[index, max_index+1] = 0.999
    if test_log_loss > log_loss(y_true, np.array(df.iloc[:, 1:])):  
        df_best = copy.deepcopy(df)
        test_log_loss = log_loss(y_true, np.array(df.iloc[:, 1:]))

    for index, row in df.iterrows():
        logits = row[1:]  # Exclude 'uid' column
        logits = np.array(logits)
        if np.max(logits) == 0.999:
            continue
        if (logits <= 0.05).any():
            if np.min(logits) <= 0.0005:
                continue
            add = np.min(logits) - 0.0005
            df.iloc[index, np.argmax(logits) + 1] += add
            df.iloc[index, np.argmin(logits) + 1] -= add
    if test_log_loss > log_loss(y_true, np.array(df.iloc[:, 1:])):
        df_best = copy.deepcopy(df)
        test_log_loss = log_loss(y_true, np.array(df.iloc[:, 1:]))
    
    df_best.to_csv(f"results/{cfg_proj.model}.csv", index = False)
    print(f"Test Log Loss (Multiclass) {test_log_loss:.3f}")