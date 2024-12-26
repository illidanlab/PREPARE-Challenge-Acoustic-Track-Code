import pandas as pd
import numpy as np
df_meta = pd.read_csv("data/metadata.csv")

paths = ["data/train_labels.csv", "data/acoustic_test_labels.csv"]

for path in paths:
    print(path)
    df_labels = pd.read_csv(path)

    labels = ["diagnosis_control","diagnosis_mci","diagnosis_adrd"]

    for label in labels:
        print(label)
        df = df_labels[df_labels[label] == 1.0]
        uids = list(df["uid"])
        print("Number of Samples", len(uids))
        meta_label = df_meta[df_meta["uid"].isin(uids)]

        gender = meta_label[meta_label["gender"] == "female"]
        print(f"Gender (%female) {len(gender)/len(uids) * 100:.3f}")

        ages =  list(meta_label["age"])
        print(f"Age {np.mean(ages):.3f}±{np.std(ages):.3f}")
        
    print()


