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

### About the Dataset



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