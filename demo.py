from utils.audio import *

# Manipulate audio
from pydub import AudioSegment
import pyaudio
import wave

# Manipulate model
import torch
from torch.nn import Softmax

# Manipulate data
import numpy as np

# Manipulate files
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

def demo(model_name):
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
    model = torch.load(model_name)
    
    # Get prediction and softmax to turn into probability
    prediction = Softmax(dim=0)( model(MFCC.float()) )
    
    # Display nicely
    print()
    for i, pred in enumerate(prediction):
        print(f"{Metadata['mapping'][str(i)] + ':':20s}{pred:.4f}")
    print()
    p = int(torch.argmax(prediction, dim=0))
    print(f"Prediction: {Metadata['mapping'][str(p)]}")

if __name__ == "__main__":
    args = get_args()
    demo(args.model_name)