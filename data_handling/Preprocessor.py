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
import json

class Preprocessor(object):

    """ An abtract class meant to be implemented for importing data from multiple databases
    """

    extra = ['.DS_Store']

    def __init__(self, seed=None, n_mfcc=30):
        self.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.seed = seed
        self.n_mfcc = n_mfcc

    def rearrange(self):
        raise NotImplementedError("")

    def parse_file_name(self, f_name):
        raise NotImplementedError("")

    def create_new_data_directory(self):
        
        # getting sorted list of files in the directory
        try:
            dir_list = os.listdir(self.original_path)
        except:
            #print(f"{self.original_path} does not exists")
            try:
                target_dir_list = os.listdir(self.path)
                print("Data files has already been reordered")
                return None # dataset already exists
            except:
                print("The dataset has not been imported")
                return None # original dataset does not exist

        try:
            target_dir_list = os.listdir(self.path)
            print("Data files has already been reordered")
            return None # dataset already exists
        except:
            pass

        # sorting directory list so actors are in order
        dir_list.sort()

        # getting rid of unneeded directories
        for f in self.extra:
            try:
                dir_list.remove(f)
            except:
                # file not in directory
                pass

        # creating new directory for reordered data
        try:
            os.mkdir(self.path)
            print(f"created {self.path}")
        except:
            print(f"{self.path} already exists")
        
        return dir_list

    def check_dataset_created(self):
        try:
            dir_list = os.listdir(self.path)
        except:
            print(f"Something went wrong and {self.path} was not created")
        
        try:
            dir_list.remove('.DS_Store')
        except:
            # '.DS_Store' not in directory
            pass
        
        print(f"{self.path} contains {len(dir_list)} total data files")

    def get_audio_data(self):
        print("Loading raw audio data...")

        df_audio = pd.DataFrame(columns=["audio"])
        df_label = pd.DataFrame(columns=["label", "actor", "length"])

        # generate list of all files within an emotion in order
        fname_list = os.listdir(self.path)
        fname_list.sort()

        idx = 0 # can't use enumerate because we sometimes skip
        # loop through all files
        for f in fname_list:
            
            # get the file name, part is a list with the important info of each file
            skip, actor, gender, emotion = self.parse_file_name(f)

            if skip:
                continue

            # Convert .wav file to a integer array using Librosa
            audio_file = os.path.join(self.path, f)
            data, _ = librosa.load(audio_file, res_type='kaiser_fast', sr=self.Metadata["sample rate"], duration=self.Metadata["duration"])

            # Add to pandas dataframes
            df_audio.loc[idx] = [ data ]
            df_label.loc[idx] = [ f"{gender.lower()}_{emotion.lower()}", actor, 0 ]
            idx += 1
        
        # removing header from dataframe (makes it nicer when we want to load stuff with out model)
        df_audio = pd.DataFrame(df_audio["audio"].values.tolist())

        # replace NaNs with 0's
        df_audio= df_audio.fillna(0)

        return pd.concat([df_label, df_audio], axis=1)


    """ Augmentation """
    def pitch(self, data_array):
        bins_per_octave = 12 # standard/ default number for music/sound
        pitch_pm = 2 # factor
        pitch_change = pitch_pm * 2 * np.random.uniform()
        data_array = librosa.effects.pitch_shift(   data_array.astype('float64'), self.Metadata["sample rate"], 
                                                    n_steps=pitch_change, 
                                                    bins_per_octave=bins_per_octave
                                                )
        return data_array

    def white_noise(self, data_array):
        # The number below is just how much minimum amplitude you want to add. Should limit the value between 0 and 0.056 from some testing.
        noise_amp = 0.06 * np.random.uniform() * np.amax(data_array)
        data_array = data_array.astype('float64') + noise_amp * np.random.normal(size=data_array.shape[0])
        return data_array

    def shift(self, data_array):
        # Shifts the data left or right randomly depending on the low and high value.
        s_range = int( 500 * np.random.uniform(low=-90, high=90) )
        return np.roll(data_array, s_range)

    def volume(self, data_array):
        # Gives back the exact same shape but scaled amplitude down or up.
        # Will basically create quieter and louder versions of our dataset.
        dyn_change = np.random.uniform(low=-0.5 ,high=3)
        return data_array * dyn_change
    
    def augment(self, df, frac=0.1):
        print("Augmenting...")
        if frac <= 0:
            return df

        # sampling a fraction of the total data
        pitch_data = df.sample(frac=frac)
        white_noise_data = df.sample(frac=frac)
        shift_data = df.sample(frac=frac)
        volume_data = df.sample(frac=frac)

        # looping through each augmentation type
        Data = [pitch_data, white_noise_data, shift_data, volume_data]
        Aug_functions = [self.pitch, self.white_noise, self.shift, self.volume]
        Name = ["pitch", "white noise", "shifting", "volume"]
        for data, f, name in zip(Data, Aug_functions, Name):
            
            # extracting labels and actors
            labels = data["label"].to_numpy()
            actors = data["actor"].to_numpy()
            audio_lengths = data["length"].to_numpy()
            np_data = data.drop(["label", "actor", "length"], axis=1).to_numpy(dtype=np.float32)
            
            # looping through each sample
            for np_sample, label, actor, audio_length in zip(np_data, labels, actors, audio_lengths):

                df_label = pd.DataFrame({"label" : [label], "actor" : [actor], "length" : [audio_length]})
                df_aug_data = pd.DataFrame( [f(np_sample)] ) # creates a dataframe with one row
                
                # concatinating augmented results with corresponding labels to original dataframe
                df_aug_data = pd.concat( [df_label, df_aug_data], axis=1 )
                df = pd.concat( [df, df_aug_data], ignore_index=True )
            
            print(f"{name.title()} augmentation complete")
        print()
        return df


    """ Mel-Frequency Cepstrum Conversion """
    def mfcc_conversion(self, df):
        print("Converting to MFCC Representation...")
        
        df_mfcc = pd.DataFrame(columns=["audio"])
        df_label = df[["label", "actor", "length"]]
        np_data = df.drop(["label", "actor", "length"], axis=1).to_numpy()

        for i, data in enumerate(np_data):
            
            # actual MFCC conversion
            MFCC = librosa.feature.mfcc(np.asfortranarray(data), sr=self.Metadata["sample rate"], n_mfcc=self.n_mfcc)
            n_mfcc, audio_length = MFCC.shape   # (30, 216) for default inputs

            # Add mfcc representation of recording as well as its gender and emotion label to panda frames
            df_mfcc.loc[i] = [ MFCC.flatten() ]
            df_label.loc[i]["length"] = audio_length

        # removing header from dataframe (makes it nicer when we want to load stuff with out model)
        expanded_mfcc = pd.DataFrame(df_mfcc["audio"].values.tolist())

        # replace NaNs with 0's
        expanded_mfcc = expanded_mfcc.fillna(0)

        return pd.concat([df_label, expanded_mfcc], axis=1)
    

    """ Making datasets """
    def split_data(self, df, le=None, append=True, equalize=True):
        print("Splitting Data...")

        # Integer encoding Labels and replace category labels
        if le == None:
            le = LabelEncoder()
            df["label"] = le.fit_transform( df["label"] )
        else:
            df["label"] = le.transform( df["label"] )

        # getting list of unique labels in dataframe
        # since we are using multiple datasets, there might be labels in the label encoder 
        # that are not in the current dataset
        Labels = list(sorted(df["label"].unique()))

        # Getting Mapping in order to reconstruct label from encoding
        # This will be saved to a json file `Metadata.json` later
        Mapping = dict(zip( le.classes_, le.transform(le.classes_) ))
        Mapping = {str(Mapping[label]) : label for label in Mapping}

        # Splitting Testing data from rest by actor
        #   i.e. Actors 1, 2, 3, and 4 go into Testing Data
        test_data = pd.DataFrame()
        if len(self.test_actors) != 0:
            test_actors = df["actor"].isin(self.test_actors)
            test_data = df.loc[ test_actors ]
            df = df.drop(df[test_actors].index, axis=0)
        
        # separating into each unique label
        Data_Splits = {"training" : {}, "validation" : {}}
        for int_category in Labels:

            train_category_df, valid_category_df = train_test_split(  
                                                                df.loc[ df["label"] == int(int_category) ], 
                                                                test_size=0.20,
                                                                random_state=self.seed
                                                            )

            Data_Splits["training"][Mapping[str(int_category)]] = train_category_df
            Data_Splits["validation"][Mapping[str(int_category)]] = valid_category_df
        
        # getting equal numbers for classes
        if equalize:
            for dataset in Data_Splits:
                min_class_size = min([df.shape[0] for df in Data_Splits[dataset].values()])
                for category in Data_Splits[dataset]:
                    Data_Splits[dataset][category] = Data_Splits[dataset][category].sample(n=min_class_size, random_state=self.seed)

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
        
        # had to do it separately for test data
        print("Test Data")
        if len(self.test_actors) != 0:
            category_counts = test_data["label"].value_counts()
            total = 0
            for category_int in Labels:
                curr = category_counts[int(category_int)]
                print(f"\t{Mapping[str(category_int)]:20s} {curr}")
                total += curr
            print(f"Total: {int(total)}")
        else:
            for category_int in Labels:
                print(f"\t{Mapping[str(category_int)]:20s} {0}")
            print(f"Total: {0}")
        print()
        
        # concatinating everything together and separating out labels
        train_data = pd.concat( Data_Splits["training"].values(), axis=0 )
        train_label = train_data[["label", "length"]]
        train_data = train_data.drop(["label", "actor", "length"], axis=1)

        valid_data = pd.concat( Data_Splits["validation"].values(), axis=0 )
        valid_label = valid_data[["label", "length"]]
        valid_data = valid_data.drop(["label", "actor", "length"], axis=1)

        test_label = pd.DataFrame()
        if len(self.test_actors) != 0:
            test_label = test_data[["label", "length"]]
            test_data = test_data.drop(["label", "actor", "length"], axis=1)

        # Overfit Data with equal distribution of labels
        overfit_data = pd.DataFrame(columns=df.columns)
        for int_category in Labels:
            category_df = df.loc[ df["label"] == int(int_category) ].sample(n=10, random_state=self.seed)
            overfit_data = pd.concat( (overfit_data, category_df), axis=0 )
        overfit_label = overfit_data[["label", "length"]]
        overfit_data = overfit_data.drop(["label", "actor", "length"], axis=1)

        print("Saving Data...")

        # creating a directory we will need later
        try:
            os.mkdir(f"{self.ROOT}/data")
            print(f"created {self.ROOT}/data")
        except:
            print(f"{self.ROOT}/data already exists")

        # Saving to tsv
        mode = 'a' if append else 'w'       # a = append, w = overwrite
        header = False if append else True

        train_data.to_csv(path_or_buf=f"{self.ROOT}/data/train_data.tsv", sep='\t', mode=mode, index=True, header=header)
        train_label.to_csv(path_or_buf=f"{self.ROOT}/data/train_label.tsv", sep='\t', mode=mode, index=True, header=header)
        valid_data.to_csv(path_or_buf=f"{self.ROOT}/data/valid_data.tsv", sep='\t', mode=mode, index=True, header=header)
        valid_label.to_csv(path_or_buf=f"{self.ROOT}/data/valid_label.tsv", sep='\t', mode=mode, index=True, header=header)
        if len(self.test_actors) != 0:
            test_data.to_csv(path_or_buf=f"{self.ROOT}/data/test_data.tsv", sep='\t', mode=mode, index=True, header=header)
            test_label.to_csv(path_or_buf=f"{self.ROOT}/data/test_label.tsv", sep='\t', mode=mode, index=True, header=header)
        overfit_data.to_csv(path_or_buf=f"{self.ROOT}/data/overfit_data.tsv", sep='\t', mode=mode, index=True, header=header)
        overfit_label.to_csv(path_or_buf=f"{self.ROOT}/data/overfit_label.tsv", sep='\t', mode=mode, index=True, header=header)

        # getting maximum audio length
        max_audio_length = 0
        try:
            max_audio_length = max([train_label["length"].max(), valid_label["length"].max(), test_label["length"].max()])
        except:
            max_audio_length = max([train_label["length"].max(), valid_label["length"].max()])

        # getting total counts for each dataset
        total_train = train_data.shape[0]
        total_valid = valid_data.shape[0]
        total_test = test_data.shape[0]

        # saving relavent metadata
        Metadata = {}
        if append:
            Metadata = json.load(open(f"{self.ROOT}/data/Metadata.json", "r"))
            
            # updating max audio length
            Metadata["max audio length"] = max(Metadata["max audio length"], max_audio_length)

            # updating totals
            Metadata["total training data"] += total_train
            Metadata["total validation data"] += total_valid
            Metadata["total test data"] += total_test

        else:
            Metadata = {    "mapping" : Mapping, 
                            "n_mfcc" : self.n_mfcc, 
                            "max audio length" : max_audio_length,
                            "total training data" : total_train,
                            "total validation data" : total_valid,
                            "total test data" : total_test
                        }
        
        with open(f"{self.ROOT}/data/Metadata.json", "w+") as g:
            json.dump(Metadata, g, indent=4)
        
        print(f"{self.dataset} processing complete")
        print()
        return le
