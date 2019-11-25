# PENN

## Table of Contents

## Repository Structure
```
└── PENN
  ├── Baseline
  |   └── model.py
  ├── CNN
  |   └── model.py
  ├── RNN
  |   └── model.py
  ├── data_handling
  |   ├── load_data.py
  |   ├── Preprocessor.py
  |   ├── RAVDESS_preprocessor.py
  |   ├── SAVEE_preprocessor.py
  |   ├── TESS_preprocessor.py
  |   └── Personal_preprocessor.py
  ├── raw_data
  |   ├── RAVDESS_metadata.json
  |   ├── SAVEE_metadata.json
  |   ├── TESS_metadata.json
  |   └── Personal_metadata.json
  ├── __init__.py
  ├── preprocess.py
  ├── main.py
  ├── demo.py
  ├── args.py
  ├── utils.py
  └── trained_model.pt
```

## Setting Up Environment
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
We are using the following emotional speech audio datasets:
* [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio/)
* [SAVEE](https://www.kaggle.com/barelydedicated/savee-database)
* [TESS](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)


Download the datasets and unzip the files. You should then have the following folders with data
* RAVDESS: `ravdess-emotional-speech-audio`
* SAVEE: `AudioData`
* TESS: `TESS Toronto emotional speech set data`
Copy these folders into the `PENN/raw_data` directory

Execute the following:
```
$ python preprocess.py
```
This will take a very long time as it is processing, MFCC converting, augmenting, and normalizing all the data. The following will be created.
* `raw_data/RAVDESS` directory with the data from the `ravdess-emotional-speech-audio` reformatted in a more convienent way
* `raw_data/SAVEE` directory with the data from the `AudioData` reformatted
* `raw_data/TESS` directory with the data from the `TESS Toronto emotional speech set data` reformatted
* `data` directory containing `.tsv` files of the different datasets
You may delete the `ravdess-emotional-speech-audio`, `AudioData`, `TESS Toronto emotional speech set data` folders if you would like.

To use the model modify the `main.py` code as needed and execute
```
$ python main.py --model cnn --epoch 30 --save
```
This will save the model in as `trained_model.pt`

### About the RAVDESS Dataset

File Naming
* Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
* Vocal channel (01 = speech, 02 = song).
* Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
* Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* Repetition (01 = 1st repetition, 02 = 2nd repetition).
* Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Modality - vocal channel - emotion - emotional intensity - statement - repetition - actor

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
