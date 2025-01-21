import pandas as pd

df = pd.read_csv("data/transcripts.csv")
df_label = pd.read_csv("data/train_labels.csv")
uids = list(df_label["uid"])

alexa_uids_train = []
alexa_uids_test = []

for index, row in df.iterrows():
    if "alexa" in str(row["transcript"]).lower():
        if row["uid"] in uids:
            alexa_uids_train.append(row["uid"])
        else:
            alexa_uids_test.append(row["uid"])

print(f"Number of Alexa samples in Train Dataset is {len(alexa_uids_train)}")
print(f"Number of Alexa samples in Test Dataset is {len(alexa_uids_test)}")

print("Alexa uids are", alexa_uids_train)