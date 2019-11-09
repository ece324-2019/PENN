# pip install my-voice-analysis
# copy the contents of the `__init__.py` function and change the name to `myspsolution.py`
# Also need to copy `mysolution.praat` to the same directory

from pydub import AudioSegment
import myspsolution as mysp
import pandas as pd
import numpy as np

import os
import json

RAVDESS = {
    "modality" :        {'01' : "full-AV", '02' : "video-only", '03' : "audio-only"},
    "vocal channel" :   {'01' : "speech", '02' : "song"},
    "emotion" :         {'01' : "neutral", '02' : "calm", '03' : "happy", '04' : "sad",
                         '05' : "angry", '06' : "fearful", '07' : "disgust", '08' : "surprised"},
    "intensity" :       {'01' : "normal", '02' : "strong"},
    "statement" :       {'01' : "Kids are talking by the door", '02' : "Dogs are sitting by the door"},
    "repetition" :      {'01' : 1, '02' : 2},
    "gender" :          {0: "Female", 1 : "Male"}
}

class baseline(object):

    def __init__(self):
        pass

    def __call__(self, file_name, path):
        Features = mysp.mysptotal(file_name, path)

    def get_audio_features(self, data):
        
        Features = mysp.mysptotal(data[0], self.path)
        """
        Features = {
            "number_of_syllables" : [],
            "number_of_pauses" : [],
            "rate_of_speech" : [],
            "articulation_rate" : [],
            "speaking_duration" : [],
            "original_duration" : [],
            "balance" : [],
            "f0_mean" : [],
            "f0_std" : [],
            "f0_median" : [],
            "f0_min" : [],
            "f0_max" : [],
            "f0_quantile25" : [],
            "f0_quan75" : []
        }
        """
        # f0 = fundemantal frequency distribution

        for file_name in dataset:
            Features.append( mysp.mysptotal(file_name, path) )
        
        Features.to_csv("./features.csv")

    def get_statistics(self, file_path):
        # convert to framerate of 44kHz
        sample = AudioSegment.from_wav(file_path)
        sample = sample.set_frame_rate(44000)
        
        # get statistics
        sample.export("./temp.wav", format="wav")
        stats = mysp.mysptotal("temp", ".")
        os.remove("temp.wav")
        return stats
    
    def process_RAVDESS_data(self):
        base_path = "../RAVDESS/"
        
        Feature_Statistics = {}
        for gender in os.listdir(base_path):
            print(gender)
            Feature_Statistics[gender] = {}
            
            for mood in os.listdir(base_path + f"{gender}/"):
                print("\t", mood)
                Features_df = pd.DataFrame()
                
                for clip in os.listdir(base_path + f"{gender}/{mood}"):
                    path = base_path + f"{gender}/{mood}/{clip}"
                    print("\t\tProcessed: ", path)
                    
                    stat = self.get_statistics(path)
                    if Features_df.empty:
                        Features_df = stat
                    else:
                        Features_df = Features_df.append(stat, ignore_index=True)

                # idk why I have to do this, but it doesn't work otherwise
                Features_df.to_csv("temp.csv")
                df = pd.read_csv("temp.csv")

                Feature_Statistics[gender][mood] = df.describe(include='all').to_dict()
                del Feature_Statistics[gender][mood]["Unnamed: 0"]

        # Writing to json file
        with open('Feature_Statistics.json', 'w+') as g:
            json.dump(Feature_Statistics, g, indent=4, sort_keys=True)
        
        os.remove("temp.csv")
    
    def predict(self, test_file):
        
        stats = model.get_statistics(test_file)

        Feature_Statistics = json.load(open('Feature_Statistics.json', 'r'))
        z_score = {}
        for gender in Feature_Statistics:
            z_score[gender] = {}

            for mood in Feature_Statistics[gender]:
                z_score[gender][mood] = {}

                for feature in Feature_Statistics[gender][mood]:
                    mean = Feature_Statistics[gender][mood][feature]["mean"]
                    std = Feature_Statistics[gender][mood][feature]["std"]
                    try:
                        z_score[gender][mood][feature] = (float(stats[feature]) - mean) / std
                    except:
                        z_score[gender][mood][feature] = 0.0

        for gender in z_score:
            for mood in z_score[gender]:
                z_score[gender][mood] = sum( z_score[gender][mood].values() )
    
        min_z = z_score["Female"]["angry"]
        min_feature = "Female_angry"
        for gender in z_score:
            for mood in z_score[gender]:
                if z_score[gender][mood] < min_z:
                    min_z = z_score[gender][mood]
                    min_feature = f"{gender}_{mood}"

        print(z_score)
        print(min_feature)
        return min_feature

if __name__ == "__main__":
    model = baseline()
    #model.process_RAVDESS_data()

    #test_file = "../RAVDESS/Female/disgust/03-01-07-01-02-01-12.wav"
    test_file = "../RAVDESS/Male/angry/03-01-05-01-01-02-19.wav"
    model.predict(test_file)

    