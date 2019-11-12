import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import IPython.display as ipd  # To play sound in the notebook



RAV = "./RAVDESS/"
df_mfcc = pd.DataFrame(columns=['feature'])
df_label = pd.DataFrame(columns=['label'])

dir_list = os.listdir(RAV)
dir_list.sort() #Female -> Male

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

counter  = 0
# Loop through genders in order
for gender in dir_list:
    #generate list of emotions in each gender in order
    emotion_list = os.listdir(os.path.join(RAV, gender))
    emotion_list.sort()
    # loop through list of emotions
    for emotion in emotion_list:
        #generate list of all files within an emotion in order
        fname_list = os.listdir(os.path.join(RAV,gender,emotion))
        fname_list.sort()
        #loop through all files
        for f in fname_list:
            #get the file name, part is a list with the important info of each file
            file = os.path.join(RAV,gender,emotion,f)
            part = f.split('.')[0].split('-')
            #Convert .wav file to a integer array using Librosa
            X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
            sample_rate = np.array(sample_rate)
            #Take integer array and convert it to a MFCC
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=30),axis=0)
            #Add mfcc representation of recording as well as its gender and emotion label to panda frames
            df_mfcc.loc[counter] = [mfccs]
            df_label.loc[counter] = int2gender[part[6]] + "_" +int2emotion[part[2]]
            counter = counter + 1

#need to get rid of the missing values (NA) in the feature column so have to split that up
expanded_mfcc = pd.DataFrame(df_mfcc['feature'].values.tolist())
expanded_mfcc =expanded_mfcc.fillna(0)

#Concatenate into a single dataframe with shape = 1440 by 217
# column 0 as the label then column 1 to column 216 are the 217 values of the MFCC array
# index = 0 to 1440, feature = MFCC array with length = 13, label = "gender_emotion"
df = pd.concat([df_label,expanded_mfcc],axis =1)


train_data, valid_data, train_label, valid_label = train_test_split(df.drop(['label'],axis=1),
                                                                    df.label,
                                                                    test_size=0.20,
                                                                    shuffle=True,
                                                                    random_state=100)

#Need to normalize data, normalize valid data based off of normalization of training data
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean)/std
valid_data = (valid_data - mean)/std

print(type(train_data))
print(train_data.shape)
print(train_data)









