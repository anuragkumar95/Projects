import numpy as np
import pysptk
from scipy.io import wavfile
import librosa
import librosa.display
import random
from tqdm import tqdm_notebook

#Function for padding mfcc
def padding(egs):
    max_val = max([e.shape[0] for e in egs])
    padded = np.array([np.pad(e, ((0, max_val-e.shape[0]), (0, 0)) , 'constant') for e in egs])
    return padded

def pad_features(path):
    folder = os.fsencode(path)
    features = []
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        features.append((filename, np.load(path + filename)))
    
    feats = [f[1] for f in features]
    feats = padding(feats)
    features = [[x[0], y[0] for x, y in zip(features, feats)]
    for name, feature in features:
        np.save(path + name, feature)

def create_features(path):
    save_path = '../data/'
    folder = os.fsencode(path)

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        save_filename = filename.split('.')[0]
        
        print("Saving MFCCs......")
        #Creating mfccs
        frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
        frames *= pysptk.blackman(frame_length)
        mgc = pysptk.mgcep(frames, order, 0.0, -1.0)
        logH = pysptk.mgc2sp(mgc, 0.0, -1.0, frame_length).real
        mfcc = librosa.feature.mfcc(y=x, sr=fs, S=logH.T, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
        #creating deltas 
        delta = np.zeros(mfcc.shape)
        for i in range(mfcc.shape[1]-1):
            delta[:,i] = mfcc[:,i+1] - mfcc[:,i]
        #creating ddeltas
        ddelta = np.zeros(delta.shape)
        for i in range(delta.shape[1]-1):
            ddelta[:,i] = delta[:,i+1] - delta[:,i]

        #Removing nan values
        mfcc = np.nan_to_num(mfcc)
        delta = np.nan_to_num(delta)
        ddelta = np.nan_to_num(ddelta)

        #Normalizing the values
        mfcc = mfcc/np.max(mfcc)
        delta = delta/np.max(delta) 
        ddelta = ddelta/np.max(ddelta)

        np.save(save_path + "MFCC/" + save_filename, mfcc)
        np.save(save_path + "DMFCC/" + save_filename, delta)
        np.save(save_path + "DDMFCC/" + save_filename, ddelta)

        #Padding features
        pad_features(save_path + "MFCC/")
        pad_features(save_path + "DMFCC/")
        pad_features(save_path + "DDMFCC/")

        print("Features created.........")
