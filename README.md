# PENN
**P**rediction **E**motions using a **N**eural **N**etwork

The goal of PENN is to be able to take in an audio file of a person speaking and accurately predicting their emotion. PENN can predict the emotions angry, disgust, fear, happy, neutral, sad, and surprised across both males and females with a top 2 accuracy of 95%. This repository was developed by Gabriel Deza and Eric Keilty as part of the ECE324 final project at the University of Toronto.

Link to [video presentation](https://www.youtube.com/watch?v=FgxWvIUK_i8&feature=emb_logo) <br>
Link to [report](https://www.eecg.utoronto.ca/~jayar/ece324/2019/penn.pdf)

## Table of Contents
* [Repository Structure](#repository-structure)
* [Setting Up Environment](#setting-up-environment)
  * [Software Required](#software-required)
* [Downloading Data](#downloading-data)
  * [About the RAVDESS Dataset](#about-the-ravdess-dataset)
  * [About the SAVEE Dataset](#about-the-savee-dataset)
  * [About the TESS Dataset](#about-the-tess-dataset)
* [Using the Model](#using-the-model)
  * [Summary](#summary)
  * [Preprocessing](#preprocessing)
  * [Training](#training)
  * [Demo](#demo)


## Repository Structure
```
└── PENN
  ├── __init__.py
  ├── Baseline
  |   └── model.py
  ├── CNN
  |   └── model.py
  ├── RNN
  |   └── model.py
  ├── raw_data
  |   ├── Personal
  |   |   └── .wav files
  |   ├── RAVDESS_metadata.json
  |   ├── SAVEE_metadata.json
  |   ├── TESS_metadata.json
  |   └── Personal_metadata.json
  ├── data_handling
  |   ├── load_data.py
  |   ├── Preprocessor.py
  |   ├── RAVDESS_preprocessor.py
  |   ├── SAVEE_preprocessor.py
  |   ├── TESS_preprocessor.py
  |   └── Personal_preprocessor.py
  ├── utils
  |   ├── audio.py
  |   ├── plot.py
  |   └── train.py
  ├── args.py
  ├── preprocess.py
  ├── pre_train.py
  ├── fine_tune.py
  ├── demo.py
  ├── pretrained_mlp.pt
  ├── pretrained_rnn.pt
  ├── pretrained_cnn.py
  └── finetune.pt
```

## Setting Up Environment
Creating virtual environment with [Anaconda](https://www.anaconda.com/distribution/)
```
$ conda create --name penn python=3.7
$ source activate penn
```

Installing other Python packages required using anaconda
```
$ conda install --yes --file requirements.txt
```

Installing other Python packages required using pip
```
$ pip install -r requirements.txt
```

### Software Required
The required software should be installed by using the command above and `requirements.txt`. However, if this doesn't work here is how you can download the software manually using anaconda.

Installing [PyTorch](https://pytorch.org/)
```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```

Installing [Librosa](https://librosa.github.io/librosa/)
```
$ conda install -c conda-forge librosa
```

Installing [PyDub](https://github.com/jiaaro/pydub)
```
$ conda install -c conda-forge pydub
```

Installing [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/)
```
$ conda install -c anaconda pyaudio
```

## Downloading Data
We are using the following emotional speech audio datasets. Download and unzip the datasets. You should then have the following folders.
* [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio/): `ravdess-emotional-speech-audio`
* [SAVEE](https://www.kaggle.com/barelydedicated/savee-database): `AudioData`
* [TESS](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess): `TESS Toronto emotional speech set data`

Copy these folders into the `PENN/raw_data` directory

### About the RAVDESS Dataset

There are 24 actors/actresses (half male and half female) who say two different statements in 8 different and 2 different intensities. A summary of the file naming is given below.

`03-01-05-01-02-01-17.wav` = 'Modality'-'vocal channel'-'emotion'-'emotional intensity'-'statement'-'repetition'-'actor'
* Modality: 01 = full-AV, 02 = video-only, 03 = audio-only
* Vocal channel: 01 = speech, 02 = song
* Emotion: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
* Emotional intensity: 01 = normal, 02 = strong, NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement: 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
* Repetition: 01 = 1st repetition, 02 = 2nd repetition
* Actor/Actress: 01 to 24. Odd numbered actors are male, even numbered actors are female

Total data: 60 x 24 = 1440

### About the SAVEE Dataset

There are 4 male actors who say a numebr of different phrases. A summary of the file naming is given below.

`DC/h03.wav` = 'Actor'/'emotion' 'statement'
* Actors: DC, JE, JK, KL
* Emotion: a = angry, d = disgust, f = fear, h = happy, n = neutral, sa = sad, su = surprised
* Statement: 15 TIMIT sentences per emotion = 3 common + 2 emotion-specific + 10 generic sentences

Total data: 120 x 4 = 480

### About the TESS Dataset

There are 2 female actresses who say a number of different words. A summary of the file naming is given below.

`OAF_happy/OAF_hall_happy.wav` = 'Actress'\_'emotion'/'Actress'\_'word'\_'emotion'
* Actress: OAF, YAF
* Emotion: angry, disgust, fear, happy, ps (pleasent-surprised), sad, neutral
* Word: back, bar, base, bath, bean, beg, bite, boat, bone, book, bought, burn, cab, calm, came, cause, chain, chair, chalk, chat, check, cheek, chief, choice, cool, dab, date, dead, death, deep, dime, dip, ditch, dodge, dog, doll, door, fail, fall, far, fat, fit, five, food, gap, gas, gaze, germ, get, gin, goal, good, goose, gun, half, hall, hash, hate, have, haze, hire, hit, hole, home, hurl, hush, jail, jar, join, judge, jug, juice, keen, keep, keg, kick, kill, king, kite, knock, late, laud, lean, learn, lease, lid, life, limb, live, loaf, long, lore, lose, lot, love, luck, make, match, merge, mess, met, mill, mob, mode, mood, moon, mop, mouse, nag, name, near, neat, nice, note, numb, pad, page, pain, pass, pearl, peg, perch, phone, pick, pike, pole, pool, puff, rag, raid, rain, raise, rat, reach, read, red, ring, ripe, road, room, rose, rot, rough, rush, said, sail, search, seize, sell, shack, shall, shawl, sheep, shirt, should, shout, size, soap, soup, sour, south, sub, such, sure, take, talk, tape, team, tell, thin, third, thought, thumb, time, tip, tire, ton, tool, tough, turn, vine, voice, void, vote, wag, walk, wash, week, wheat, when, which, whip, white, wife, wire, witch, yearn, yes, young, youth

## Using the Model

### Summary

```
$ python preprocess.py
$ python pre_train.py --model cnn --lr 0.01 --batch-size 100 --epochs 50 --eval-every 10 --save-as pretrained_model
$ python python fine_tune.py --model-name pretrained_model --save-as finetuned_model
$ python demo.py --model-name finetuned_model
```

### Preprocessing

The data is in `.wav` files. We need to convert it to a format that is easier for the neural network to use. This is done using the `preprocess.py` script.

Execute the following:
```
$ python preprocess.py
```
This will take a very, **very** long time as it is loading, MFCC converting, augmenting, and normalizing all the data. The following will be created.
* `raw_data/RAVDESS` directory with the data from the `ravdess-emotional-speech-audio` reformatted in a more convienent way
* `raw_data/SAVEE` directory with the data from the `AudioData` reformatted
* `raw_data/TESS` directory with the data from the `TESS Toronto emotional speech set data` reformatted
* `data` directory containing `.tsv` files of the different datasets
You may delete the `ravdess-emotional-speech-audio`, `AudioData`, `TESS Toronto emotional speech set data` folders if you would like.

For more details on exactly how the preprocessor works, see the README in the directory `data_handling`

### Training

Training the model will consist of 2 parts: pre-training and fine-tuning. The pre-training will use the dataset created by executing `preprocess.py`. The fine-tuning will use our personally recorded dataset. Why do we need to fine-tune the model? If you try to just use the pre-trained model, it will not work when you try to demo. This is because different microphones process audio data a little differently and the neural network will not be able to recognize it. Therefore, fine-tuning was implemented so it can retain what it has learned about classifying audio data, but also work on your computer microphone. The fine-tuning data was recorded on a MacBook pro and will work on any other Mac. If you have another computer then you will need to obtain data recorded on your specific type of laptop in order for the demo to work.  

To pre-train, use `pre_train.py`. There are a number of commandline arguments that can be used. Go to `args.py` to see them for yourself.
* `--model`: ["mlp", "average", "cnn", "rnn"] --> which architecture is used
* `--lr`: any positive float --> learning rate of the model
* `--batch-size`: any positive integer --> batch size of the data
* `--epochs`: any positive integer --> number of epochs the model will train on
* `--eval-every`: any positive integer --> how many batches occur until we evaluate the current state of the model
* `--overfit`: no parameters --> use the overfit data (for debugging purposes)
* `--save-as`: any string --> save the model after training with the name as the string inputted

The following are the defaults
```
$ python pre_train.py --model cnn --lr 0.01 --batch-size 100 --epochs 50 --eval-every 10 --save-as pretrained_model
```

To fine-tune, use `fine_tune.py`. The following are commandline arguments for this script.
* `--model-name`: any string --> use the pre-trained model with name equal to the string inputted
* `--save-as`: any string --> save the model after training with the name as the string inputted

```
$ python fine_tune.py --model-name pretrained_model --save-as finetuned_model
```

### Demo

To demo with your own recording, use `demo.py`. The following commandline argument is used for this script
* `--model-name`: any string --> use the model with name equal to the string inputted

```
$ python demo.py --model-name finetuned_model
```

The script will ask you to record a sample, convert that sample to a `.wav` file, run that audio through the model, and produce the model's prediction.
