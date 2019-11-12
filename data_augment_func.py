import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd

np.random.seed(400)
int2gender = {
    "01": "Male","03": "Male","05": "Male","07": "Male","09": "Male","11": "Male","13": "Male","15": "Male","17": "Male","19": "Male",
    "21": "Male", "23": "Male",
    "02": "Female","04": "Female","06": "Female","08": "Female","10": "Female","12": "Female","14": "Female","16": "Female","18": "Female",
    "20": "Female","22": "Female","24": "Female"}
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
def prepare_data_no_aug_mfcc(folder, mfcc_num, sampling_rate, audio_duration):
    emotion = []
    X = np.empty(shape =(1440, mfcc_num, 216, 1))
    cnt = 0
    file_list = os.listdir(folder)
    for file in file_list:
        file_path = os.path.join(folder,file)
        part = file.split('.')[0].split('-')
        data , _ = librosa.load(file_path, sr = sampling_rate, res_type = "kaiser_fast", duration = audio_duration)
        MFCC = librosa.feature.mfcc(data, sr = sampling_rate, n_mfcc = mfcc_num)
        MFCC = np.expand_dims(MFCC, axis=-1)
        X[cnt,] = MFCC
        emotion.append(int2gender[part[6]] + "_" +int2emotion[part[2]])
        cnt += 1
    return X, emotion

mfcc_number = 30
sr = 44100
n_mfcc = 30
dur = 2.5
path = "./big_set/"

# data, label_array = prepare_data_no_aug_mfcc(path, n_mfcc, sr, dur)
#
# label = np.asarray(label_array)
#
# np.save("./data_test", data)
# np.save("./label_test",label)

mfcc = np.load("./data_test.npy", allow_pickle = True)
label = np.load("./label_test.npy", allow_pickle = True)

train_data, valid_data, train_label, valid_label = train_test_split(mfcc
                                                     , label
                                                     , test_size=0.20
                                                     , shuffle=True
                                                     , random_state=42
                                                    )
lb = LabelEncoder()
#convert (2 genders * 8 emotions = 16 possible labels) into numbers 0 to 15
train_label = lb.fit_transform(train_label)
valid_label = lb.fit_transform(valid_label)

#normalize data
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean)/std
valid_data = (valid_data - mean)/std

model = CNN
