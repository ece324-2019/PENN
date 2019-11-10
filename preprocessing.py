import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import IPython.display as ipd  # To play sound in the notebook
import json


RAV = "./data/RAVDESS/"
df_mfcc = pd.DataFrame(columns=['feature'])
df_label = pd.DataFrame(columns=['label'])

dir_list = os.listdir(RAV)
dir_list.sort() # ['Female', 'Male']

RAVDESS_metadata = json.load(open("./data/RAVDESS_metadata.json", "r"))

counter  = 0
# Loop through genders in order
for gender in dir_list:
    
    # generate list of emotions in each gender in order
    emotion_list = os.listdir(os.path.join(RAV, gender))
    emotion_list.sort()
    # loop through list of emotions
    for emotion in emotion_list:
        
        # generate list of all files within an emotion in order
        fname_list = os.listdir(os.path.join(RAV,gender,emotion))
        fname_list.sort()
        #loop through all files
        for f in fname_list:
            
            # get the file name, part is a list with the important info of each file
            file = os.path.join(RAV,gender,emotion,f)
            part = f.split('.')[0].split('-')
            # Convert .wav file to a integer array using Librosa
            X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
            sample_rate = np.array(sample_rate)
            # Take integer array and convert it to a MFCC
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=30),axis=0)
            # Add mfcc representation of recording as well as its gender and emotion label to panda frames
            df_mfcc.loc[counter] = [mfccs]
            gender = RAVDESS_metadata["gender"][int(part[6])%2]
            emotion = RAVDESS_metadata["emotion"][part[2]]
            df_label.loc[counter] = f"{gender.lower()}_{emotion.lower()}"
            counter += 1

#need to get rid of the missing values (NA) in the feature column so have to split that up
expanded_mfcc = pd.DataFrame(df_mfcc['feature'].values.tolist())
expanded_mfcc =expanded_mfcc.fillna(0)

# Concatenate into a single dataframe with shape = 1440 by 217
# column 0 as the label then column 1 to column 216 are the 217 values of the MFCC array
# index = 0 to 1440, feature = MFCC array with length = 13, label = "gender_emotion"
df = pd.concat([df_label,expanded_mfcc], axis=1)

# Training and validation data
train_data, valid_data, train_label, valid_label = train_test_split(    df.drop(['label'], axis=1),
                                                                        df["label"],
                                                                        test_size=0.20,
                                                                        shuffle=True,
                                                                        random_state=100
                                                                    )

# Overfit Data
overfit_data = df.sample(n=50, random_state=100)
overfit_label = overfit_data["label"]
overfit_data = overfit_data.drop(['label'], axis=1)

# Need to normalize data, normalize valid data based off of normalization of training data
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean)/std
train_data.insert(0, "label", train_label)

mean = np.mean(valid_data, axis=0)
std = np.std(valid_data, axis=0)
valid_data = (valid_data - mean)/std
valid_data.insert(0, "label", valid_label)

mean = np.mean(overfit_data, axis=0)
std = np.std(overfit_data, axis=0)
overfit_data = (overfit_data - mean)/std
overfit_data.insert(0, "label", overfit_label)

# Saving to csv
train_data.to_csv(path_or_buf='./train.tsv', sep='\t', index=False)
valid_data.to_csv(path_or_buf='./validation.tsv', sep='\t', index=False)
overfit_data.to_csv(path_or_buf='./overfit.tsv', sep='\t', index=False)