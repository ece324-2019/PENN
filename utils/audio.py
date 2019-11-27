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
    
    
    
    #audio_file_path = "./raw_data/RAVDESS/03-01-06-01-01-01-15.wav"
    #print(get_silence(audio_file_path, threshold=0))
    #plot_waveform(audio_file_path)
    #plot_MFCC(audio_file_path)
    #audio = AudioSegment.from_wav(audio_file_path)
    #print(len(audio) / 1000)
    #wav_to_mp3(l[1])