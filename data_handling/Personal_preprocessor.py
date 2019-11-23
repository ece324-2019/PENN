from preprocessor import Preprocessor

import os
import shutil
import json
from pydub import AudioSegment


class Personal_Preprocessor(Preprocessor):
    name = "Personal_Preprocessor"
    dataset = "Personal"
    sample_rate = 44100
    duration = 2.5

    def __init__(self, raw_data_dir="Personal", data_dir="Personal_wav",metadata_file="Personal_metadata.json", seed=None):
        Preprocessor.__init__(self, seed=seed)
        self.original_path = os.path.join(self.ROOT, "raw_data", raw_data_dir)
        self.path = os.path.join(self.ROOT, "raw_data", data_dir)
        metadata_path = os.path.join(self.ROOT, "raw_data", metadata_file)
        self.Metadata = json.load(open(metadata_path, "r"))

    def convert_to_wav(self):
        fname_list = self.create_new_data_directory()
        fname_list.sort()
        for f in fname_list:
            (final_file_path, file_extension) = os.path.splitext(f)
            filepath = os.path.join(self.original_path,f)
            finalpath = os.path.join(self.path, final_file_path + ".wav")
            track = AudioSegment.from_file(filepath, "m4a")
            file_handle = track.export(finalpath, format='wav')
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

if __name__ == "__main__":
    Personal = Personal_Preprocessor(seed=100)
    Personal.convert_to_wav()
    #df, n_mfcc, audio_length = RAVDESS.mfcc_conversion()
    #le = RAVDESS.split_data(df, n_mfcc, audio_length, append=False)
