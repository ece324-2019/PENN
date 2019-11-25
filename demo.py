# Manipulate audio
import librosa
from pydub import AudioSegment
from pydub.playback import play
#conda install -c anaconda pyaudio
import pyaudio
import wave

# Manipulate model
import torch
from torch.nn import Softmax

# Manipulate data
import pandas as pd
import numpy as np

import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def record(output_file_name, length=2.5, sample_rate=44100, channels=2, audio_format=pyaudio.paInt16, chunk=1024):
    audio = pyaudio.PyAudio()

    # let user get ready to record
    while True:
        inp = input("Press ENTER to start recording")
        break

    # start recording
    stream = audio.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)
    print("\n\nRecording...")
    frames = []
    for i in range(int(sample_rate / chunk * length)+1):
        data = stream.read(chunk)
        frames.append(data)

    # stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Finished recording")

    # saving audio
    waveFile = wave.open(output_file_name, 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(audio_format))
    waveFile.setframerate(sample_rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def play_mp3(audio_path):
    sound = AudioSegment.from_mp3(audio_path)
    play(sound)

def play_wav(audio_path):
    sound = AudioSegment.from_wav(audio_path)
    play(sound)

def wav_to_mp3(audio_file_path):
    clip = AudioSegment.from_wav(audio_file_path)
    clip.export("test.mp3", format="mp3")

def mfcc_conversion(file_path, sr=44100, n_mfcc=30, duration=2.5):
        
    # Convert .wav file to a integer array using Librosa
    data, _ = librosa.load(file_path, res_type='kaiser_fast', sr=sr, duration=duration)
    MFCC = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
    
	# MFCC is a 2D numpy array of shape n_mfcc by audio_length (which is really the number of samples)
    n_mfcc, audio_length = MFCC.shape

    # replacing NAN's with 0
    MFCC = np.nan_to_num(MFCC)
    
    return MFCC, n_mfcc, audio_length

if __name__ == "__main__":
    
    record("demo.wav")
    MFCC, n_mfcc, audio_length = mfcc_conversion("demo.wav")

    # Metadata file
    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))

    # normalizing the data with the training mean and std
    
    MFCC = MFCC.flatten()
    MFCC = (MFCC - np.array(Metadata["mean"])) / np.array(Metadata["std"])
    MFCC = MFCC.reshape((n_mfcc, audio_length))
    
    # getting data in proper type
    MFCC = torch.from_numpy(MFCC).reshape(1, n_mfcc, audio_length)

    # load model
    #model_name = "trained_model"
    model_name = "cnn"
    model = torch.load(f"{model_name}.pt")
    
    # Get prediction and softmax to turn into probability
    prediction = Softmax(dim=0)( model(MFCC.float()) )
    
    # Display nicely
    print()
    for i, pred in enumerate(prediction):
        print(f"{Metadata['mapping'][str(i)]}:\t{pred:.4f}")
    print()
    p = int(torch.argmax(prediction, dim=0))
    print(f"Prediction: {Metadata['mapping'][str(p)]}")
    