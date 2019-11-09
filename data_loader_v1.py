import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import shutil
import os
import sys
import warnings

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

RAV = "./ravdess-emotional-speech-audio/" #name of file

dir_list = os.listdir(RAV)  #list of actors

# getting rid of unneeded directories
try:
    dir_list.remove('.DS_Store')
except:
    # '.DS_Store' not in directory
    pass
try:
    dir_list.remove('audio_speech_actors_01-24')
except:
    # 'audio_speech_actors_01-24' not in directory
    pass

dir_list.sort() #list of "Actor_1", "Actor_2" ...

RAVDESS = {
    "modality" :        {'01' : "full-AV", '02' : "video-only", '03' : "audio-only"},
    "vocal channel" :   {'01' : "speech", '02' : "song"},
    "emotion" :         {'01' : "neutral", '02' : "calm", '03' : "happy", '04' : "sad",
                         '05' : "angry", '06' : "fearful", '07' : "disgust", '08' : "surprised"},
    "intensity" :       {'01' : "normal", '02' : "strong"},
    "statement" :       {'01' : "Kids are talking by the door", '02' : "Dogs are sitting by the door"},
    "repetition" :      {'01' : 1, '02' : 2},
    "gender" :          {0: "Female", 1 : "Male"}
}

#creating the required folders
try:
    os.mkdir("./RAVDESS")
    for gender in RAVDESS["gender"].values():
        os.mkdir(f"./RAVDESS/{gender}")
        for mood in RAVDESS["emotion"].values():
            os.mkdir(f"./RAVDESS/{gender}/{mood}")
except:
    # Folders have already been created
    pass

for Actor in dir_list: #for loops through the actor
    fname = os.listdir(os.path.join(RAV, Actor))
    for f in fname:
        part = f.split('.')[0].split('-') # part = [['03', '01', '02', '01', '01', '01', '08']
        gender = RAVDESS["gender"][int(part[6])%2]
        mood = RAVDESS["emotion"][part[2]]
        shutil.move(os.path.join(RAV, Actor, f), f"./RAVDESS/{gender}/{mood}/")