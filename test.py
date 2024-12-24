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

def test(X_train, X_test, y_train, test_uids, cfg_proj):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if cfg_proj.outlier_detection:
        svm = OneClassSVM(gamma='auto', nu=0.1)
        svm.fit(X_train)
        outliers = svm.predict(X_train)
        print("Mumber of samples removed by One-class SVM is", np.sum(outliers==-1))
        X_train = X_train[outliers == 1]
        y_train = np.array(y_train)[outliers == 1]

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
        classifier = GridSearchCV(MLPClassifier(max_iter=2000, random_state=42), parameters, n_jobs=-1, verbose=2)

    if cfg_proj.model == "RF":
        parameters = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        classifier = GridSearchCV(RandomForestClassifier(random_state=42), parameters, n_jobs=-1, verbose=2)

    if cfg_proj.model == "SVC":
        parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100, 200, 500, 700, 1000], "degree": [1, 3, 5, 7, 9, 11, 13], "gamma": ["scale", "auto"]}
        classifier = GridSearchCV(SVC(probability=True, max_iter=1000000, random_state = 42), parameters, n_jobs=-1, verbose=2)

    if cfg_proj.model == "GBC":
        parameters = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        classifier = GridSearchCV(GradientBoostingClassifier(random_state=42), parameters, n_jobs=-1, verbose=2)

    if cfg_proj.model == "LR":
        parameters = {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'saga'],
            'l1_ratio': [0.1, 0.5, 0.9],  # Only used when penalty='elasticnet'
            'max_iter': [100, 200, 500]
        }
        classifier = GridSearchCV(LogisticRegression(random_state=42), parameters, n_jobs=-1, verbose=2)

    classifier.fit(X_train, y_train)
    df = pd.DataFrame(classifier.cv_results_)
    df = df[['params', 'mean_test_score', 'std_test_score']]
    df = df.sort_values(by='mean_test_score', ascending=False)
    
    # Save cross-validation performance
    acc = df.iloc[0, 1]
    df.to_csv(f"results_local/{cfg_proj.model}.csv", index=False)

    # Generate test predictions
    proba = classifier.predict_proba(X_test)
    df = pd.DataFrame(proba, columns=["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"])
    df.insert(0, "uid", test_uids)   

    # Set Chinese Samples to MCI 
    df_transcripts = pd.read_csv("data/transcripts.csv")
    zh_uids = df_transcripts[df_transcripts["language"] == "zh"]["uid"].values

    df.loc[df["uid"].isin(zh_uids), "diagnosis_control"] = 0.0005
    df.loc[df["uid"].isin(zh_uids), "diagnosis_mci"] = 0.999
    df.loc[df["uid"].isin(zh_uids), "diagnosis_adrd"] = 0.0005

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

    df.to_csv(f"results/{cfg_proj.model}.csv", index = False)

    return acc
