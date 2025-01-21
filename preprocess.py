import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from feature_extractor import extract

def get_data(cfg_proj, train = 1):
    df = pd.read_csv("data/train_labels.csv")
    df_transcript = pd.read_csv("data/transcripts.csv")
    alexa_uids = ['aaop', 'amsn', 'aooz', 'avdc', 'bclu', 'bqjw', 'cheh', 'ctiq', 'cvxa', 'dcfp', 'dsqt', 'eafy', 'ecvs', 'ewqw', 'fdzt', 'fhsx', 'fkvj', 'fqdl', 'frix', 'gxyh', 'hpme', 'iefo', 'iqhu', 'iqzp', 'jcra', 'jmoo', 'jnia', 'jnmb', 'joxx', 'jrgq', 'jupu', 'jygb', 'kkkl', 'klzk', 'kqay', 'ldml', 'lmwe', 'lunv', 'lwvt', 'mryi', 'mvlw', 'mxbw', 'ncpc', 'nhrv', 'oees', 'oghy', 'ojvu', 'olub', 'onlv', 'pjex', 'ptcr', 'pxqc', 'pzdb', 'pzoi', 'qdyv', 'qgxn', 'qkpp', 'qneb', 'qwos', 'rlxh', 'scyo', 'suzi', 'szfw', 'tbrc', 'toft', 'tomj', 'ttjk', 'tudo', 'uipr', 'uwrh', 'vpbc', 'vpxq', 'vrsk', 'vwhr', 'vyoe', 'vyoy', 'wrih', 'wzww', 'xarx', 'yiqh', 'ylsp', 'ystq', 'zcyg', 'zdhx', 'zouf']
    short_uids = ['anek', 'bajg', 'cubn', 'dcdp', 'dsdk', 'iopn', 'jryz', 'kwqg', 'omct', 'ozof', 'pyki', 'veex', 'vjbx', 'vtqu', 'zljk', 'zriz', 'zrsl', 'zyzb']
    uids = list(df["uid"].values)
    dic2features = {}

    for acoustic_type in cfg_proj.acoustic:
        if not os.path.exists(f"features/{acoustic_type}.pkl"):
            extract(acoustic_type)
        with open(f"features/{acoustic_type}.pkl", "rb") as f:
            dic2feature = pickle.load(f)
        for uid in dic2feature:
            if uid not in dic2features:
                dic2features[uid] = np.array([])
            dic2features[uid] = np.concatenate((dic2features[uid], dic2feature[uid]), axis = 0)

    X, y = [], []
    test_uids = []
    train_uids = []

    for uid in dic2features:
        if train and uid in uids:
            if cfg_proj.remove_noise == "all" and ((uid in alexa_uids) \
                                          or (df_transcript[df_transcript["uid"] == uid]["language"].values[0] == "zh") \
                                            or (df_transcript[df_transcript["uid"] == uid]["language"].values[0] == "gl") \
                                                or (uid in short_uids)):
                continue

            if cfg_proj.remove_noise == "alexa" and (uid in alexa_uids): 
                continue

            if cfg_proj.remove_noise == "language" and ((df_transcript[df_transcript["uid"] == uid]["language"].values[0] == "zh") \
                                            or (df_transcript[df_transcript["uid"] == uid]["language"].values[0] == "gl")):
                continue
        
            if cfg_proj.remove_noise == "short" and (uid in short_uids):
                continue
            
            cognitive_status = list(df[df["uid"] == uid].iloc[:, 1:].values[0])
            y.append(cognitive_status.index(1.0))
            X.append(dic2features[uid])
            train_uids.append(uid)
        if (not train) and (uid not in uids):
            X.append(dic2features[uid])
            test_uids.append(uid)

    if train:
        return X, y, train_uids

    return X, test_uids