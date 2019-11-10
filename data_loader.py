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
import json

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

RAV = "./data/ravdess-emotional-speech-audio/" # path to files
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

dir_list.sort() #["Actor_1", "Actor_2", ..., "Actor_24"]

RAVDESS_metadata = json.load(open("./data/RAVDESS_metadata.json", "r"))

#creating the required folders
try:
    os.mkdir("./data/RAVDESS")
    print("created ./data/RAVDESS")
except:
    print("./data/RAVDESS already exists")
for gender in RAVDESS_metadata["gender"]:
    try:
        os.mkdir(f"./data/RAVDESS/{gender}")
        print(f"created ./data/RAVDESS/{gender}")
    except:
        print(f"./data/RAVDESS/{gender} already exists")
    for emotion in RAVDESS_metadata["emotion"].values():
        try:
            os.mkdir(f"./data/RAVDESS/{gender}/{emotion}")
            print(f"created ./data/RAVDESS/{gender}/{emotion}")
        except:
            print(f"./data/RAVDESS/{gender}/{emotion} already exists")


for Actor in dir_list: #for loops through the actor
    fname = os.listdir(os.path.join(RAV, Actor))
    for f in fname:
        parts = f.split('.')[0].split('-') # part = [['03', '01', '02', '01', '01', '01', '08']
        gender = RAVDESS_metadata["gender"][int(parts[6])%2]
        emotion = RAVDESS_metadata["emotion"][parts[2]]
        shutil.move(os.path.join(RAV, Actor, f), f"./data/RAVDESS/{gender}/{emotion}/")