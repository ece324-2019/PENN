# pip install my-voice-analysis
# copy the contents of the `__init__.py` function and change the name to `myspsolution.py`
# Also need to copy `mysolution.praat` to the same directory

import myspsolution as mysp
import pandas as pd

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

    def get_statistics(self):
        Features = pd.read_csv("./features.csv")
        stats = Features.describe()



if __name__ == "__main__":

    model = baseline()
    model(10)
