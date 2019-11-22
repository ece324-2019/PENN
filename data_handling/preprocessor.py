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

    def __init__(self, seed=None):
        self.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.seed = seed

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
                dir_list = os.listdir(self.path)
                print("Data files has already been reordered")
            except:
                print("The dataset has not been imported")
            return None

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
        
        print(f"{len(dir_list)} total data files now in the directory {self.path}")

    def mfcc_conversion(self, sr=44100, n_mfcc=30, duration=2.5):
        
        print("Creating Dataframes")
        df_mfcc = pd.DataFrame(columns=["feature"])
        df_label = pd.DataFrame(columns=["label", "actor"])

        # generate list of all files within an emotion in order
        fname_list = os.listdir(self.path)
        fname_list.sort()

        audio_length = 0
        cnt = 0 # can't use enumerate because we sometimes skip
        
        # loop through all files
        print("Iterating over files")
        for f in fname_list:
            
            # get the file name, part is a list with the important info of each file
            skip, actor, gender, emotion = self.parse_file_name(f)

            if skip:
                continue

            # Convert .wav file to a integer array using Librosa
            audio_file = os.path.join(self.path, f)
            data, _ = librosa.load(audio_file, res_type='kaiser_fast', sr=sr, duration=duration)
            MFCC = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
            n_mfcc, audio_length = MFCC.shape   # (30, 216) for default inputs

            # Add mfcc representation of recording as well as its gender and emotion label to panda frames
            df_mfcc.loc[cnt] = [ MFCC.flatten() ]
            df_label.loc[cnt] = [ f"{gender.lower()}_{emotion.lower()}", actor ]
            cnt += 1
            
        print("Loaded data into dataframe\n")

        # need to get rid of the missing values (NA) in the feature column so have to split that up
        expanded_mfcc = pd.DataFrame(df_mfcc["feature"].values.tolist())
        expanded_mfcc = expanded_mfcc.fillna(0)
        
        # Concatenate into a single dataframe
        df = pd.concat([df_label, expanded_mfcc], axis=1)

        return df, n_mfcc, audio_length


    """ Augmentation """
    def pitch(self, data_array, sample_rate):
        bins_per_octave = 12 # standard/ default number for music/sound
        pitch_pm = 2 # factor
        pitch_change = pitch_pm * 2 * np.random.uniform()
        data_array = librosa.effects.pitch_shift(   data_array.astype('float64'), sample_rate, 
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
        # The audio will roll around like the lin alg quiz question. 
        # I though that was bad as the sentence wont make sense but since we dont even look at the words so this is probably good.
        # ...wut ^^
        s_range = int( 500 * np.random.uniform(low=-90, high=90) )
        return np.roll(data_array, s_range)

    def volume(self, data_array):
        # Gives back the exact same shape but scaled amplitude down or up.
        # Will basically create quieter and louder versions of our dataset.
        dyn_change = np.random.uniform(low=-0.5 ,high=3)
        return data_array * dyn_change
    
    def augment(self, df, frac=0.1):
        # sampling a fraction of the total data
        pitch_data = df.sample(frac=frac)
        white_noise_data = df.sample(frac=frac)
        shift_data = df.sample(frac=frac)
        volume_data = df.sample(frac=frac)

        # looping through each augmentation type
        Data = [pitch_data, white_noise_data, shift_data, volume_data]
        Aug_functions = [lambda data: self.pitch(data, self.sample_rate), self.white_noise, self.shift, self.volume]
        Name = ["pitch", "white noise", "shifting", "volume"]
        for data, f, name in zip(Data, Aug_functions, Name):
            
            # extracting labels and actors
            labels, actors = data["label"].to_numpy(), data["actor"].to_numpy()
            np_data = data.drop(["label", "actor"], axis=1).to_numpy(dtype=np.float32)
            
            # looping through each sample
            for np_sample, label, actor in zip(np_data, labels, actors):
                # contatinating augmented results with corresponding label and actor to original dataframe
                np_aug_df = pd.DataFrame([ f(np_sample) ]) # creates a dataframe with one row
                np_aug_df = pd.concat( [pd.DataFrame({"label" : [label], "actor" : [actor]}), np_aug_df], axis=1 )
                df = pd.concat( [df, np_aug_df], ignore_index=True )
            print(f"{name.title()} augmentation complete")

        return df

    """ Making datasets """
    def split_data(self, df, n_mfcc, audio_length, le=None, append=True):
        
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
        for dataset in Data_Splits:
            min_class_size = min([df.shape[0] for df in Data_Splits[dataset].values()])
            for category in Data_Splits[dataset]:
                Data_Splits[dataset][category] = Data_Splits[dataset][category].sample(n=min_class_size, random_state=self.seed)

        # printing amount of data in each category
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
        category_counts = test_data["label"].value_counts()
        total = 0
        for category_int in Labels:
            curr = category_counts[int(category_int)]
            print(f"\t{Mapping[str(category_int)]:20s} {curr}")
            total += curr
        print(f"Total: {int(total)}")
        print()
        
        # concatinating everything together and separating out labels
        train_data = pd.concat( Data_Splits["training"].values(), axis=0 )
        train_label = train_data["label"]
        train_data = train_data.drop(["label", "actor"], axis=1)

        valid_data = pd.concat( Data_Splits["validation"].values(), axis=0 )
        valid_label = valid_data["label"]
        valid_data = valid_data.drop(["label", "actor"], axis=1)

        test_label = test_data["label"]
        test_data = test_data.drop(["label", "actor"], axis=1)

        # Overfit Data with equal distribution of labels
        overfit_data = pd.DataFrame(columns=df.columns)
        for int_category in Labels:
            category_df = df.loc[ df["label"] == int(int_category) ].sample(n=10, random_state=self.seed)
            overfit_data = pd.concat( (overfit_data, category_df), axis=0 )
        overfit_label = overfit_data["label"]
        overfit_data = overfit_data.drop(["label", "actor"], axis=1)

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
        test_data.to_csv(path_or_buf=f"{self.ROOT}/data/test_data.tsv", sep='\t', mode=mode, index=True, header=header)
        test_label.to_csv(path_or_buf=f"{self.ROOT}/data/test_label.tsv", sep='\t', mode=mode, index=True, header=header)
        overfit_data.to_csv(path_or_buf=f"{self.ROOT}/data/overfit_data.tsv", sep='\t', mode=mode, index=True, header=header)
        overfit_label.to_csv(path_or_buf=f"{self.ROOT}/data/overfit_label.tsv", sep='\t', mode=mode, index=True, header=header)

        # saving relavent metadata
        Metadata = {}
        if append:
            pass
        else:
            Metadata = {"mapping" : Mapping, "n_mfcc" : n_mfcc, "audio_length" : audio_length, "mean" : mean, "std" : std}
            with open(f"{self.ROOT}/data/Metadata.json", "w+") as g:
                json.dump(Metadata, g, indent=4)
        
        return le
    