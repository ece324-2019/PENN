from .preprocessor import Preprocessor

import os
import shutil
import json

class SAVEE_Preprocessor(Preprocessor):

    name = "SAVEE_Preprocessor"
    dataset = "SAVEE"

    def __init__(self, raw_data_dir="AudioData", data_dir="SAVEE", metadata_file="SAVEE_metadata.json", seed=None):
        Preprocessor.__init__(self, seed=seed)

        self.extra += ['AudioData', 'Info.txt']
        self.test_actors = ['DC']

        self.original_path = os.path.join(self.ROOT, "raw_data", raw_data_dir)
        self.path = os.path.join(self.ROOT, "raw_data", data_dir)
        
        metadata_path = os.path.join(self.ROOT, "raw_data", metadata_file)
        self.Metadata = json.load(open(metadata_path, "r"))
    
    def reordering(self):    
        dir_list = self.create_new_data_directory()

        for actor in dir_list: #for loops through the actor
            fname_list = os.listdir(os.path.join(self.original_path, actor))
            fname_list.sort()
            for f in fname_list:
                shutil.copy(os.path.join(self.original_path, actor, f), self.path)
                new_f = f"{actor}-{f[:-6]}-{f[-6:]}"
                os.rename( os.path.join(self.path, f), os.path.join(self.path, new_f) )
        
        self.check_dataset_created()

    def parse_file_name(self, f_name):
        #           0           1           2
        # parts = ["Actor", "Gender", "Statement"]
        parts = f_name.split('.')[0].split('-')
        gender = self.Metadata["gender"]                 # Male
        emotion = self.Metadata["emotion"][parts[1]]
        actor = parts[0]

        skip = False
        
        return skip, actor, gender, emotion

if __name__ == "__main__":
    
    SAVEE = SAVEE_Preprocessor(seed=100)
    #SAVEE.reordering()
    df, n_mfcc, audio_length = SAVEE.mfcc_conversion()
    le = SAVEE.split_data(df, n_mfcc, audio_length, append=False)
    
    #print( le.transform(["male_happy"]) )