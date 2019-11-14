# manipulating audio
import librosa
import librosa.display

# manipulating data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# manipulating files
import os
import shutil
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 100

def RAVDESS_reordering():
    RAV = f"{ROOT}/raw_data/ravdess-emotional-speech-audio/"
    
    try:
        dir_list = os.listdir(RAV)
    except:
        #print(f"{ROOT}/ravdess-emotional-speech-audio/ does not exists")
        try:
            dir_list = os.listdir(f"{ROOT}/raw_data")
            print("Data files has already been reordered")
        except:
            print("You have not imported the RAVDESS dataset. Go to https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio, download the data, and put it in the PENN directory")
        return None
    
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
        os.mkdir(f"{ROOT}/raw_data/RAVDESS")
        print(f"created {ROOT}/raw_data/RAVDESS")
    except:
        print(f"{ROOT}/raw_data/RAVDESS already exists")
    
    for i, actor in enumerate(dir_list): #for loops through the actor
        fname_list = os.listdir(os.path.join(RAV, actor))
        fname_list.sort()
        for f in fname_list:
            shutil.move(os.path.join(RAV, actor, f), f"{ROOT}/raw_data/RAVDESS")


def RAVDESS_mfcc_conversion(sr=44100, n_mfcc=30, duration=2.5):
    RAV = f"{ROOT}/raw_data/RAVDESS/"
    df_mfcc = pd.DataFrame(columns=['feature'])
    df_label = pd.DataFrame(columns=['label'])

    dir_list = os.listdir(RAV)
    dir_list.sort() #Female -> Male

    RAVDESS_metadata = json.load(open(f"{ROOT}/raw_data/RAVDESS_metadata.json", "r"))

    # generate list of all files within an emotion in order
    fname_list = os.listdir(os.path.join(RAV))
    fname_list.sort()
    audio_length = 0
    # loop through all files
    for index, f in enumerate(fname_list):    
        
        # get the file name, part is a list with the important info of each file
        file = os.path.join(RAV, f)
        part = f.split('.')[0].split('-')
        
        # Convert .wav file to a integer array using Librosa
        data, _ = librosa.load(file, res_type='kaiser_fast', sr=sr, duration=duration)
        MFCC = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
        n_mfcc, audio_length = MFCC.shape   # (30, 216) for default inputs

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

    return df, n_mfcc, audio_length

def split_data(df, n_mfcc, audio_length):
    
    # Integer encoding Labels
    le = LabelEncoder()
    df_label_ints = le.fit_transform( df["label"] )

    # replacing cateogory labels with integers
    df["label"] = df_label_ints

    # Getting Mapping in order to reconstruct label from encoding
    # This will be saved to a json file `Metadata.json` later
    Mapping = dict(zip( le.classes_, le.transform(le.classes_) ))
    Mapping = {str(Mapping[label]) : label for label in Mapping}

    # creating an equal distribution of labels
    Data_Splits = {"train" : {}, "valid" : {}, "test" : {}}
    for int_category in Mapping:
        train_category_df, test_category_df = train_test_split(  
                                                            df.loc[ df["label"] == int(int_category) ], 
                                                            test_size=0.20,
                                                            random_state=SEED
                                                        )
        train_category_df, valid_category_df = train_test_split(  
                                                            train_category_df.loc[ train_category_df["label"] == int(int_category) ], 
                                                            test_size=0.20,
                                                            random_state=SEED
                                                        )

        Data_Splits["train"][Mapping[int_category]] = train_category_df
        Data_Splits["valid"][Mapping[int_category]] = valid_category_df
        Data_Splits["test"][Mapping[int_category]] = test_category_df
    
    # printing amount of data in each category
    for dataset in Data_Splits:
        print(f"{dataset.title()} Data")
        total = 0
        for category in Data_Splits[dataset]:
            curr = Data_Splits[dataset][category].shape[0]
            print(f"\t{category}:\t\t{curr}")
            total += curr
        print(f"\tTotal: {int(total)}")
    
    # concatinating everything together and separating out labels
    train_data = pd.concat( Data_Splits["train"].values(), axis=0 )
    train_label = train_data["label"]
    train_data = train_data.drop(["label"], axis=1)

    valid_data = pd.concat( Data_Splits["valid"].values(), axis=0 )
    valid_label = valid_data["label"]
    valid_data = valid_data.drop(["label"], axis=1)

    test_data = pd.concat( Data_Splits["test"].values(), axis=0 )
    test_label = test_data["label"]
    test_data = test_data.drop(["label"], axis=1)

    # Overfit Data with equal distribution of labels
    overfit_data = pd.DataFrame(columns=df.columns)
    for int_category in Mapping:
        category_df = df.loc[ df["label"] == int(int_category) ].sample(n=10, random_state=SEED)
        overfit_data = pd.concat( (overfit_data, category_df), axis=0 )
    overfit_label = overfit_data["label"]
    overfit_data = overfit_data.drop(["label"], axis=1)

    # Need to normalize data, normalize other based off of normalization of training data
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = (train_data - mean)/std
    valid_data = (valid_data - mean)/std
    test_data = (test_data - mean)/std
    overfit_data = (overfit_data - mean)/std
    
    # TODO: We might not have done the mean correctly
    #print("Mean:", mean)
    #print("Standard Deviation:", std)
    mean = 10
    std = 10

    # creating a directory we will need later
    try:
        os.mkdir(f"{ROOT}/data")
        print(f"created {ROOT}/data")
    except:
        print(f"{ROOT}/data already exists")

    # Saving to tsv
    train_data.to_csv(path_or_buf=f"{ROOT}/data/train_data.tsv", sep='\t', index=True, header=True)
    train_label.to_csv(path_or_buf=f"{ROOT}/data/train_label.tsv", sep='\t', index=True, header=True)
    valid_data.to_csv(path_or_buf=f"{ROOT}/data/valid_data.tsv", sep='\t', index=True, header=True)
    valid_label.to_csv(path_or_buf=f"{ROOT}/data/valid_label.tsv", sep='\t', index=True, header=True)
    test_data.to_csv(path_or_buf=f"{ROOT}/data/test_data.tsv", sep='\t', index=True, header=True)
    test_label.to_csv(path_or_buf=f"{ROOT}/data/test_label.tsv", sep='\t', index=True, header=True)
    overfit_data.to_csv(path_or_buf=f"{ROOT}/data/overfit_data.tsv", sep='\t', index=True, header=True)
    overfit_label.to_csv(path_or_buf=f"{ROOT}/data/overfit_label.tsv", sep='\t', index=True, header=True)

    # saving relavent metadata
    Metadata = {"mapping" : Mapping, "n_mfcc" : n_mfcc, "audio_length" : audio_length, "mean" : mean, "std" : std}
    with open(f"{ROOT}/data/Metadata.json", "w+") as g:
        json.dump(Metadata, g, indent=4)

if __name__ == "__main__":
    #RAVDESS_reordering()
    df, n_mfcc, audio_length = RAVDESS_mfcc_conversion()
    split_data(df, n_mfcc, audio_length)