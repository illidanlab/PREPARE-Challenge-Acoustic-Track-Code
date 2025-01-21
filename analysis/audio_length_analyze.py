import torchaudio
import glob 
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle

types = ["train", "test"]
uids = []
uid2length = {}

for t in types:
    paths = sorted(glob.glob(f"data/{t}_audios/*.mp3"))
    
    t = "T" + t[1:]
    print(f"{t} Data")

    # Number of audio files
    print(f"Number of {t} audio files is {len(paths)}")

    # Distribution of audio length figure
    lengths = []
    for path in tqdm(paths):
        subject_name = os.path.basename(path)[:-4]
        waveform, sample_rate = torchaudio.load(path)
        lengths.append(waveform.shape[1]//sample_rate)
        uid2length[subject_name] = waveform.shape[1]//sample_rate
        if waveform.shape[1]//sample_rate < 10:
            uids.append(subject_name)

    plt.hist(lengths, range = (0, 30))
    plt.title(f"Distribution of Audio Length in {t} Dataset")
    plt.xlabel("Length (seconds)")
    plt.ylabel("Number of Audio Files")
    plt.savefig(f"figures/{t}_audio_length_distribution.png", dpi = 300)
    plt.clf()

    # Distribution of audio length table
    ranges = [[0, 10], [10, 20], [20, 31]]
    for r in ranges:
        cnt = 0
        for length in lengths:
            if r[0] <= length and length < r[1]:
                cnt += 1
        print(f"Number of Audio Files Having Length Between {r[0]} and {r[1]} is {cnt}")

print("uids less than 10 seconds are", uids)

with open("features/length.pkl", "wb") as f:
    pickle.dump(uid2length, f)