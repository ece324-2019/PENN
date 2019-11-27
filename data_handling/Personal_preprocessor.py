from .Preprocessor import Preprocessor

import pandas as pd
from sklearn.model_selection import train_test_split
from pydub import AudioSegment

import os
import shutil
import json


class Personal_Preprocessor(Preprocessor):
    name = "Personal_Preprocessor"
    dataset = "Personal"
    sample_rate = 44100
    duration = 2.5

    def __init__(self, raw_data_dir="Original_Personal", data_dir="Personal", metadata_file="Personal_metadata.json", seed=None, n_mfcc=30):
        Preprocessor.__init__(self, seed=seed, n_mfcc=n_mfcc)
        self.original_path = os.path.join(self.ROOT, "raw_data", raw_data_dir)
        self.path = os.path.join(self.ROOT, "raw_data", data_dir)
        metadata_path = os.path.join(self.ROOT, "raw_data", metadata_file)
        self.Metadata = json.load(open(metadata_path, "r"))

    def rearrange(self):
        fname_list = self.create_new_data_directory()
        
        try:
            for f in fname_list:
                final_file_path, file_extension = os.path.splitext(f)
                filepath = os.path.join(self.original_path, f)
                finalpath = os.path.join(self.path, final_file_path + ".wav")
                track = AudioSegment.from_file(filepath, "m4a")
                file_handle = track.export(finalpath, format='wav')
        except Exception as e:
            # data files already in the correct configuration
            pass

        self.check_dataset_created()

    def parse_file_name(self, f_name):
        #               0               1               2
        # parts = ["Actor Name", "Sentence Index", "Emotion"]
        parts = f_name.split('.')[0].split('-')
        gender = self.Metadata["gender"]
        emotion = self.Metadata["emotion"][parts[2]]
        actor = parts[0]
        
        skip = False
        
        return skip, actor, gender, emotion
    
    def split(self, df):
        # Getting reverse mapping because we have to label encode manually
        Metadata = json.load(open(f"{self.ROOT}/data/Metadata.json", "r"))
        Reverse_Mapping = {}
        for key, val in Metadata["mapping"].items():
            Reverse_Mapping[val] = int(key)
        
        # integer encoding labels
        df["label"] = df["label"].apply(lambda emotion: Reverse_Mapping[emotion])

        # splitting into datasets
        Data_Splits = {"training" : {}, "validation" : {}}
        for int_category in Metadata["mapping"]:
            
            # We have no female data and we didn't do fear
            if "female" in Metadata["mapping"][int_category] or "fear" in Metadata["mapping"][int_category]:
                continue

            print(int_category, Metadata["mapping"][int_category])

            train_category_df, valid_category_df = train_test_split(  
                                                                df.loc[ df["label"] == int(int_category) ], 
                                                                test_size=0.25,
                                                                random_state=self.seed
                                                            )

            Data_Splits["training"][Metadata["mapping"][str(int_category)]] = train_category_df
            Data_Splits["validation"][Metadata["mapping"][str(int_category)]] = valid_category_df

        # printing amount of data in each category
        print()
        for dataset in Data_Splits:
            print(f"{dataset.title()} Data")
            total = 0
            for category in Data_Splits[dataset]:
                curr = Data_Splits[dataset][category].shape[0]
                print(f"\t{category:20s} {curr}")
                total += curr
            print(f"Total: {int(total)}")
            print()
        
        # concatinating everything together and separating out labels
        train_data = pd.concat( Data_Splits["training"].values(), axis=0 )
        train_label = train_data[["label", "length"]]
        train_data = train_data.drop(["label", "actor", "length"], axis=1)

        valid_data = pd.concat( Data_Splits["validation"].values(), axis=0 )
        valid_label = valid_data[["label", "length"]]
        valid_data = valid_data.drop(["label", "actor", "length"], axis=1)

        return train_data, train_label, valid_data, valid_label

if __name__ == "__main__":
    Personal = Personal_Preprocessor(seed=100)
    Personal.rearrange()
    #df, n_mfcc, audio_length = RAVDESS.mfcc_conversion()
    #le = RAVDESS.split_data(df, n_mfcc, audio_length, append=False)
