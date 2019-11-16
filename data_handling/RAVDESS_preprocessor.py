from .preprocessor import Preprocessor

import os
import shutil
import json

class RAVDESS_Preprocessor(Preprocessor):

    name = "RAVDESS_Preprocessor"
    dataset = "RAVDESS"

    def __init__(self, raw_data_dir="ravdess-emotional-speech-audio", data_dir="RAVDESS", metadata_file="RAVDESS_metadata.json", seed=None):
        Preprocessor.__init__(self, seed=seed)

        self.extra += ['audio_speech_actors_01-24']
        self.test_actors = ['01', '02', '03']

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
        
        self.check_dataset_created()

    def parse_file_name(self, f_name):
        #               0               1           2           3           4               5           6
        # parts = ["Modality", "Vocal Channel", "Emotion", "Intensity", "Statement", "Repetition", "Gender"]
        parts = f_name.split('.')[0].split('-')
        gender = self.Metadata["gender"][int(parts[6])%2]
        emotion = self.Metadata["emotion"][parts[2]]
        actor = parts[-1]

        skip = False
        if parts[3] == '01':
            skip = True
        
        return skip, actor, gender, emotion

if __name__ == "__main__":
    
    RAVDESS = RAVDESS_Preprocessor(seed=100)
    #RAVDESS.reordering()
    df, n_mfcc, audio_length = RAVDESS.mfcc_conversion()
    le = RAVDESS.split_data(df, n_mfcc, audio_length, append=False)