import librosa
import librosa.display
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(x, train_error=None, valid_error=None, title=None):
    if train_error != None:
        plt.plot(x, train_error, label="Training Error")
    if valid_error != None:
        plt.plot(x, valid_error, label="Validation Error")
    
    if title == None:
        plt.title("Training Loss")
    else:
        plt.title(title)
    
    plt.xlabel("Epochs")
    plt.xlim(left=0)
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()
    plt.clf()

def plot_accuracy(x, train_accuracy=None, valid_accuracy=None, title=None):
    if train_accuracy != None:
        plt.plot(x, train_accuracy, label="Training Accuracy")
    if valid_accuracy != None:
        plt.plot(x, valid_accuracy, label="Validation Accuracy")
    
    if title == None:
        plt.title("Accuracy")
    else:
        plt.title(title)

    plt.xlabel("Epochs")
    plt.xlim(left=0)
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(linestyle='-', axis='y')
    plt.legend(loc="lower right")
    plt.show()
    plt.clf()

def plot_confusion_matrix(confusion_matrix, class_names, title=None, figsize=(10,7), fontsize=14):
    """ Prints a confusion matrix, as returned by `sklearn.metrics.confusion_matrix`, as a heatmap.
    Arguments
    ---------
    confusion_matrix: `numpy.ndarray`
        The numpy.ndarray object returned from a call to `sklearn.metrics.confusion_matrix`.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    Returns
    -------
   `matplotlib.figure.Figure`
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame( confusion_matrix, index=class_names, columns=class_names )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.subplots_adjust(left=0.27, bottom=0.25, right=0.97, top=0.95)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if title != None:
        plt.title(title)

    plt.show()

def plot_waveform(audio_file_path):
    
    data_array, _ = librosa.load(audio_file_path, res_type='kaiser_fast', sr=44100, duration=2.5)
    audio = AudioSegment.from_wav(audio_file_path)
    audio_length = len(audio) / 1000

    fig = plt.figure(figsize=(14, 8))
    file_name = audio_file_path.split("/")[-1]
    plt.title(f"Waveform: {file_name}")
    plt.xlabel('time')
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0, audio_length, len(data_array)), data_array)
    plt.show()

def plot_MFCC(audio_file_path):
    data_array, _ = librosa.load(audio_file_path, res_type='kaiser_fast', sr=44100, duration=2.5)
    MFCC = librosa.feature.mfcc(data_array, sr=44100, n_mfcc=100)
    MFCC = np.nan_to_num(MFCC)
    
    librosa.display.specshow(MFCC, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()