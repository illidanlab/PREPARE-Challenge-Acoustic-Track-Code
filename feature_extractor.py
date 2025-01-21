import librosa
import numpy as np
from scipy.stats import skew, kurtosis
import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import glob 
import numpy 
import pickle
from tqdm import tqdm

def mfcc_helper(path):
    '''log spectrogram -> mfcc features
    mfcc: original 13 frequencies
    delta_mfcc, delta2_mfcc: vel. and accel. features
    M: 39 mfcc, delta, delta2 features'''
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return M

def extract_embeddings(wav_file_path, processor, model, device):
    waveform, sample_rate = torchaudio.load(wav_file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(device)  
    with torch.no_grad():
        outputs = model(input_values)
        embeddings = outputs.last_hidden_state

    return embeddings.mean(dim=1).cpu().detach().numpy()

def extract_mfcc():
    train_paths = sorted(glob.glob("data/train_audios/*.mp3"))
    test_paths = sorted(glob.glob("data/test_audios/*.mp3"))
    id2features = {}

    for path in tqdm(train_paths):
        subject_name = os.path.basename(path)[:-4]
        mfcc = mfcc_helper(path)
        minima, maxima, mean, std, skewness, kurtos = mfcc.min(axis = 1), mfcc.max(axis = 1), mfcc.mean(axis = 1), mfcc.std(axis = 1), skew(mfcc, axis = 1), kurtosis(mfcc, axis = 1)
        features = np.concatenate((minima,maxima,mean, std, skewness, kurtos), axis=0)
        id2features[subject_name] = features

    for path in tqdm(test_paths):
        subject_name = os.path.basename(path)[:-4]
        mfcc = mfcc_helper(path)
        minima, maxima, mean, std, skewness, kurtos = mfcc.min(axis = 1), mfcc.max(axis = 1), mfcc.mean(axis = 1), mfcc.std(axis = 1), skew(mfcc, axis = 1), kurtosis(mfcc, axis = 1)
        features = np.concatenate((minima,maxima,mean, std, skewness, kurtos), axis=0)
        id2features[subject_name] = features

    with open("features/MFCC.pkl", "wb") as f:
        pickle.dump(id2features, f)

def extract_wav2vec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = sorted(glob.glob("data/train_audios/*.mp3"))
    test_paths = sorted(glob.glob("data/test_audios/*.mp3"))
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60")
    model.to(device)
    model.eval()

    id2features = {}

    for path in tqdm(train_paths):
        subject_name = os.path.basename(path)[:-4]
        id2features[subject_name] = numpy.array(extract_embeddings(path, processor, model, device)[0])

    for path in tqdm(test_paths):
        subject_name = os.path.basename(path)[:-4]
        id2features[subject_name] = numpy.array(extract_embeddings(path, processor, model, device)[0])

    with open("features/Wav2Vec.pkl", "wb") as f:
        pickle.dump(id2features, f)

def extract(acoustic_type):
    if acoustic_type == "Wav2Vec":
        extract_wav2vec()
    if acoustic_type == "MFCC":
        extract_mfcc()

            
