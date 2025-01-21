import whisper
import torch
import glob 
from tqdm import tqdm
import os
import pandas as pd


if os.path.exists("data/transcripts.csv"):
    df = pd.read_csv("data/transcripts.csv")
else:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    whisper_model = whisper.load_model("large", device = device)

    train_paths = sorted(glob.glob("data/train_audios/*.mp3"))
    test_paths = sorted(glob.glob("data/test_audios/*.mp3"))

    texts = []
    uids = []
    lans = []

    for path in tqdm(train_paths):
        subject_name = os.path.basename(path)[:-4]
        result = whisper_model.transcribe(path)
        text = result["text"]
        lan = result["language"]
        texts.append(str(text))
        uids.append(subject_name)
        lans.append(lan)

    for path in tqdm(test_paths):
        subject_name = os.path.basename(path)[:-4]
        result = whisper_model.transcribe(path)
        text = result["text"]
        lan = result["language"]
        texts.append(str(text))
        uids.append(subject_name)
        lans.append(lan)

    dic = {"uid": uids, "transcript": texts, "language": lans}
    df = pd.DataFrame(dic)
    df.to_csv("data/transcripts.csv", index = False)

languages = ["en", "es", "zh", "gl"]
labels = ["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"]
df_label = pd.read_csv("data/train_labels.csv")
controls = list(df_label[df_label["diagnosis_control"] == 1.0])

print("TRAIN")
for lan in languages:
    for label in labels:
        lan_uids = list(df[df["language"] == lan]["uid"])
        label_uids = list(df_label[df_label[label] == 1.0]["uid"])

        cnt = 0
        for uid in lan_uids:
            if uid in label_uids:
                cnt += 1

        print(f"Number of {lan} spekaers that have {label} is {cnt}")

print("TEST")
for lan in languages:
    
    lan_uids = list(df[df["language"] == lan]["uid"])
    train_uids = list(df_label["uid"])

    cnt = 0
    for uid in lan_uids:
        if uid not in train_uids:
            cnt += 1

    print(f"Number of {lan} spekaers that in Test is {cnt}")