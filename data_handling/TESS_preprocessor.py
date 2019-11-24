from .preprocessor import Preprocessor

import os
import shutil
import json

class TESS_Preprocessor(Preprocessor):

    name = "TESS_Preprocessor"
    dataset = "TESS"

    def __init__(self, raw_data_dir="TESS Toronto emotional speech set data", data_dir="TESS", metadata_file="TESS_metadata.json", seed=None, n_mfcc=30):
        Preprocessor.__init__(self, seed=seed, n_mfcc=n_mfcc)

        self.extra += ['TESS Toronto emotional speech set data']
        self.test_actors = []

        self.original_path = os.path.join(self.ROOT, "raw_data", raw_data_dir)
        self.path = os.path.join(self.ROOT, "raw_data", data_dir)
        
        metadata_path = os.path.join(self.ROOT, "raw_data", metadata_file)
        self.Metadata = json.load(open(metadata_path, "r"))
    
    def rearrange(self):    
        dir_list = self.create_new_data_directory()

        try:
            for actor_emotion in dir_list: #for loops through the folders
                fname_list = os.listdir(os.path.join(self.original_path, actor_emotion))
                fname_list.sort()
                for f in fname_list:
                    shutil.copy(os.path.join(self.original_path, actor_emotion, f), self.path)
                    
                    new_f = f.replace('_', '-', 2)
                    # error in the dataset
                    if new_f == "OA-bite-neutral.wav":
                        new_f = "OAF-bite-neutral.wav"
                    
                    os.rename( os.path.join(self.path, f), os.path.join(self.path, new_f) )
        except Exception as e:
            # data files already in the correct configuration
            pass

        self.check_dataset_created()
    
    def parse_file_name(self, f_name):
        #           0        1          2
        # parts = ["Actor", "Word", "emotion"]
        parts = f_name.split('.')[0].split('-')
        gender = self.Metadata["gender"]
        emotion = self.Metadata["emotion"][parts[2]]
        actor = parts[0]

        skip = False
        
        return skip, actor, gender, emotion

if __name__ == "__main__":
    
    TESS = TESS_Preprocessor(seed=100)
    TESS.rearrange()
    #df, n_mfcc, audio_length = TESS.mfcc_conversion()
    #le = TESS.split_data(df, n_mfcc, audio_length, append=False)