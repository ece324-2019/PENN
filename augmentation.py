import numpy as np
import pandas as pd
import random
import itertools
import librosa
import matplotlib.pyplot as plt

def plot_audio_file(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

def pitch(data, sample_rate):
    #changes the pitch of the audio
    bins_per_octave = 12 #standard/ default number for music/sound
    pitch_pm = 2 #factor
    pitch_change =  pitch_pm * 2*(np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),sample_rate, n_steps=pitch_change,bins_per_octave=bins_per_octave)
    return data

def white_noise(data):
    #the number below is just how much minimum amplitude you want to add. should limit the value between 0 and 0.056 from some testing.
    noise_amp = 0.06*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data

def shift(data):
    #shifts the data left or right randomly depending on the low and high value. The audio will roll around
    #like the lin alg quiz question. I though that was bad as the sentence wont make sense but since we dont even look at the words
    #this is probably good.
    s_range = int(np.random.uniform(low=-90, high = 90)*500)
    return np.roll(data, s_range)

def random_change(data):
    #gives back the exact same shape but scaled amplitude down or up.
    #Will basically create quieter and louder versions of our dataset.
    dyn_change = np.random.uniform(low=-0.5 ,high=3)
    return (data * dyn_change)

'''Testing on a single file and its plot'''

# data_example = "./raw_data/RAVDESS/03-01-01-01-01-01-01.wav"
# data_array, _ = librosa.load(data_example, res_type='kaiser_fast', sr=44100, duration=2.5)
# MFCC = librosa.feature.mfcc(data_array, sr=44100, n_mfcc=30)
# data_array_shifted = pitch(data_array,44100)
# plot_audio_file(data_array)
# plot_audio_file(data_array_shifted)