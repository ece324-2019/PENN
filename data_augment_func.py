import librosa
import numpy as np

np.random.seed(400)
def pitch_tuning(data, sample_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                      sample_rate,
                                       n_steps=pitch_change,
                                      bins_per_octave=bins_per_octave)
    return data

def shifting(data):
    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500
    return np.roll(data, s_range)