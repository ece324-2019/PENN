import librosa
from pydub import AudioSegment
from pydub.playback import play

import numpy as np

def mfcc_conversion(file_path, sr=44100, n_mfcc=30, duration=2.5):
        
    # Convert .wav file to a integer array using Librosa
    data, _ = librosa.load(file_path, res_type='kaiser_fast', sr=sr, duration=duration)
    MFCC = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
    
	# MFCC is a 2D numpy array of shape n_mfcc by audio_length (which is really the number of samples)
    n_mfcc, audio_length = MFCC.shape

    # replacing NAN's with 0
    MFCC = np.nan_to_num(MFCC)
    
    return MFCC, n_mfcc, audio_length

def augment_pitch(data_array, sample_rate=44100):
        bins_per_octave = 12 # standard/ default number for music/sound
        pitch_pm = 2 # factor
        pitch_change = pitch_pm * 2 * np.random.uniform()
        data_array = librosa.effects.pitch_shift(   data_array.astype('float64'), sample_rate, 
                                                    n_steps=pitch_change, 
                                                    bins_per_octave=bins_per_octave
                                                )
        return data_array

def augment_white_noise(data_array):
    # The number below is just how much minimum amplitude you want to add. Should limit the value between 0 and 0.056 from some testing.
    noise_amp = 0.06 * np.random.uniform() * np.amax(data_array)
    data_array = data_array.astype('float64') + noise_amp * np.random.normal(size=data_array.shape[0])
    return data_array

def augment_shift(data_array):
    # Shifts the data left or right randomly depending on the low and high value.
    s_range = int( 500 * np.random.uniform(low=-90, high=90) )
    return np.roll(data_array, s_range)

def augment_volume(data_array):
    # Gives back the exact same shape but scaled amplitude down or up.
    # Will basically create quieter and louder versions of our dataset.
    dyn_change = np.random.uniform(low=-0.5 ,high=3)
    return data_array * dyn_change

def play_mp3(audio_path):
    sound = AudioSegment.from_mp3(audio_path)
    play(sound)

def play_wav(audio_path):
    sound = AudioSegment.from_wav(audio_path)
    play(sound)

def wav_to_mp3(audio_file_path, output_path="test.mp3"):
    clip = AudioSegment.from_wav(audio_file_path)
    clip.export(output_path, format="mp3")

if __name__ == "__main__":
   
    sample_1 = "./raw_data/RAVDESS/03-01-06-01-01-01-17.wav"        # guy, actually does kinda sound fearful
    sample_2 = "./raw_data/TESS/YAF-which-fear.wav"                 # weird girl
    sample_3 = "./raw_data/RAVDESS/03-01-06-02-01-01-17.wav"        # guy, very expressive
    #play_wav("raw_data/RAVDESS/03-01-04-01-01-01-01.wav")
    #play_wav("raw_data/TESS/OAF-mood-disgust.wav")
    wav_to_mp3("raw_data/TESS/OAF-mood-disgust.wav")