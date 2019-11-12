import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import shutil
import IPython.display as ipd  # To play sound in the notebook
import json
import librosa
import librosa.display


def RAVDESS_reordering():
    RAV = "../data/ravdess-emotional-speech-audio/"
    dir_list = os.listdir(RAV)
    dir_list.sort() #list of "Actor_1", "Actor_2" ...

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

    try:
        os.mkdir("../raw_data/RAVDESS")
        print("created ../raw_data/RAVDESS")
    except:
        print("../raw_data/RAVDESS already exists")
    for i in dir_list: #for loops through the actor
        fname_list = os.listdir(os.path.join(RAV, i))
        fname_list.sort()
        for f in fname_list:
            shutil.move(os.path.join(RAV, i, f), "../raw_data/RAVDESS")


def RAVDESS_mfcc_conversion():
    RAV = "../raw_data/RAVDESS/"
    df_mfcc = pd.DataFrame(columns=['feature'])
    df_label = pd.DataFrame(columns=['label'])

    dir_list = os.listdir(RAV)
    dir_list.sort() #Female -> Male

    RAVDESS_metadata = json.load(open("../raw_data/RAVDESS_metadata.json", "r"))

    X = np.empty(shape=(1440, 30, 216, 1))

    # generate list of all files within an emotion in order
    fname_list = os.listdir(os.path.join(RAV))
    fname_list.sort()
    # loop through all files
    for index, f in enumerate(fname_list):    
        
        # get the file name, part is a list with the important info of each file
        file = os.path.join(RAV, f)
        part = f.split('.')[0].split('-')
        
        # Convert .wav file to a integer array using Librosa
        data, _ = librosa.load(file, res_type='kaiser_fast', sr=44100, duration=2.5)
        MFCC = librosa.feature.mfcc(data, sr=44100, n_mfcc=30)
        # MFCC.shape = (30, 216)

        # Add mfcc representation of recording as well as its gender and emotion label to panda frames
        df_mfcc.loc[index] = [MFCC.flatten()]
        gender = RAVDESS_metadata["gender"][int(part[6])%2]
        emotion = RAVDESS_metadata["emotion"][part[2]]
        df_label.loc[index] = f"{gender.lower()}_{emotion.lower()}"

    # need to get rid of the missing values (NA) in the feature column so have to split that up
    expanded_mfcc = pd.DataFrame(df_mfcc['feature'].values.tolist())
    expanded_mfcc = expanded_mfcc.fillna(0)
    
    # Concatenate into a single dataframe
    df = pd.concat([df_label, expanded_mfcc], axis=1)

    return df

def split_data(df):

    # Integer encoding Labels
    le = LabelEncoder()
    df_label_ints = le.fit_transform( df["label"] )
    
    # creating a directory we will need later
    try:
        os.mkdir("../data")
        print("created ./data")
    except:
        print("../data already exists")
    
    # Saving Mapping in order to reconstruct label from encoding
    Mapping = dict(zip( le.classes_, le.transform(le.classes_) ))
    Mapping = {str(Mapping[label]) : label for label in Mapping}
    with open("../data/Mapping.json", "w+") as g:
        json.dump(Mapping, g, indent=4)
    
    # replacing cateogory labels with integers
    df["label"] = df_label_ints

    # Training and Validation Data
    train_data, valid_data, train_label, valid_label = train_test_split(    df.drop(["label"], axis=1),
                                                                            df["label"],
                                                                            test_size=0.20,
                                                                            shuffle=True,
                                                                            random_state=100)

    # Overfit Data
    overfit_data = df.sample(n=50, random_state=100)
    overfit_label = overfit_data["label"]
    overfit_data = overfit_data.drop(["label"], axis=1)

    # Need to normalize data, normalize valid data based off of normalization of training data
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = (train_data - mean)/std
    valid_data = (valid_data - mean)/std
    overfit_data = (overfit_data - mean)/std

    # Saving to tsv
    train_data.to_csv(path_or_buf='../data/train_data.tsv', sep='\t', index=True, header=True)
    train_label.to_csv(path_or_buf='../data/train_label.tsv', sep='\t', index=True, header=True)
    valid_data.to_csv(path_or_buf='../data/valid_data.tsv', sep='\t', index=True, header=True)
    valid_label.to_csv(path_or_buf='../data/valid_label.tsv', sep='\t', index=True, header=True)
    overfit_data.to_csv(path_or_buf='../data/overfit_data.tsv', sep='\t', index=True, header=True)
    overfit_label.to_csv(path_or_buf='../data/overfit_label.tsv', sep='\t', index=True, header=True)

if __name__ == "__main__":
    #RAVDESS_reordering()
    df = RAVDESS_mfcc_conversion()
    #split_data(df)