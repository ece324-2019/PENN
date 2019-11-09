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

RAV = "./Data_loading/ravdess-emotional-speech-audio/" #name of file

dir_list = os.listdir(RAV)  #list of actors
dir_list.sort() #list of "Actor_1", "Actor_2" ...

#creating the required folders
os.mkdir("./RAVDESS")
os.mkdir("./RAVDESS/Male")
os.mkdir("./RAVDESS/Male/neutral")
os.mkdir("./RAVDESS/Male/calm")
os.mkdir("./RAVDESS/Male/happy")
os.mkdir("./RAVDESS/Male/sad")
os.mkdir("./RAVDESS/Male/angry")
os.mkdir("./RAVDESS/Male/fearful")
os.mkdir("./RAVDESS/Male/disgust")
os.mkdir("./RAVDESS/Male/suprised")
os.mkdir("./RAVDESS/Female")
os.mkdir("./RAVDESS/Female/neutral")
os.mkdir("./RAVDESS/Female/calm")
os.mkdir("./RAVDESS/Female/happy")
os.mkdir("./RAVDESS/Female/sad")
os.mkdir("./RAVDESS/Female/angry")
os.mkdir("./RAVDESS/Female/fearful")
os.mkdir("./RAVDESS/Female/disgust")
os.mkdir("./RAVDESS/Female/suprised")

for i in dir_list: #for loops through the actor
    fname = os.listdir(os.path.join(RAV, i))
    for f in fname:
        part = f.split('.')[0].split('-') # part = [['03', '01', '02', '01', '01', '01', '08']
        #HERE ARE THE GUYS, ALL 8 EMOTIONS
        if (part[2] == '01') and (int(part[6])%2 ==1): #male and neutral
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/neutral/")
        if (part[2] == '02') and (int(part[6])%2 ==1): #male and calm
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/calm/")
        if (part[2] == '03') and (int(part[6])%2 ==1): #male and happy
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/happy/")
        if (part[2] == '04') and (int(part[6])%2 ==1): #male and sad
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/sad/")
        if (part[2] == '05') and (int(part[6])%2 ==1): #male and angry
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/angry/")
        if (part[2] == '06') and (int(part[6])%2 ==1): #male and fearful
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/fearful/")
        if (part[2] == '07') and (int(part[6])%2 ==1): #male and disgust
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/disgust/")
        if (part[2] == '08') and (int(part[6])%2 ==1): #male and suprise
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Male/suprised/")
        #DONE THE GUYS ONTO THE FEMALES NOW
        if (part[2] == '01') and (int(part[6])%2 ==0): #female and neutral
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/neutral/")
        if (part[2] == '02') and (int(part[6])%2 ==0): #female and calm
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/calm/")
        if (part[2] == '03') and (int(part[6])%2 ==0): #female and happy
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/happy/")
        if (part[2] == '04') and (int(part[6])%2 ==0): #female and sad
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/sad/")
        if (part[2] == '05') and (int(part[6])%2 ==0): #female and angry
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/angry/")
        if (part[2] == '06') and (int(part[6])%2 ==0): #female and fearful
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/fearful/")
        if (part[2] == '07') and (int(part[6])%2 ==0): #female and disgust
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/disgust/")
        if (part[2] == '08') and (int(part[6])%2 ==0): #female and suprise
            shutil.move(os.path.join(RAV, i, f), "./RAVDESS/Female/suprised/")


