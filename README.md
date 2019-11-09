# PENN

Creating virtual environment with [Anaconda](https://www.anaconda.com/distribution/)
```
$ conda create --name penn python=3.7
$ source activate penn
```

Installing [PyTorch](https://pytorch.org/)
```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```

Installing other Python packages required
```
$ pip install -r requirements.txt
```

## Downloading Data

We used the RAVDESS Emotional speech audio dataset found at this link
[RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio/data#)

Download the dataset and unzip the file. The data should be in a directory named `ravdess-emotional-speech-audio`. Move this directory into the root directory `PENN`

### About the RAVDESS Dataset

File Naming
* Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
* Vocal channel (01 = speech, 02 = song).
* Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
* Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* Repetition (01 = 1st repetition, 02 = 2nd repetition).
* Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Modality - vocal channel = emotion - emotional intensity - statement - repetition - actor

## Software we downloaded
Downloading Librosa
```
$ conda install -c conda-forge librosa
```

Downloading PyDub
```
$ conda install -c conda-forge pydub
```

My-Voice-Analysis library is included in the project
