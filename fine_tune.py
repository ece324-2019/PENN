import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from data_handling.Personal_preprocessor import Personal_Preprocessor
from main import training_loop

import numpy as np
import pandas as pd

# Plots and summary statistics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from utils import *

import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def transfer_learning():
    
    model = torch.load("cnn.pt")        # loads model with all parameters frozen
    model.fc = nn.Linear(3*35, 14)      # re-initialize last linear layer

    hyperparameters = {
                    "optimizer" : torch.optim.Adam,
                    "loss_fnc" : nn.CrossEntropyLoss(),
                    "epochs" : 20,
                    "batch_size" : 10,
                    "lr" : 0.001,
                    "eval_every" : 1
    }

    # preprocessing the personally collected data
    Personal = Personal_Preprocessor(seed=100, n_mfcc=30)
    Personal.rearrange()
    df = Personal.get_audio_data()
    df = Personal.mfcc_conversion(df)
    train_data, train_label, valid_data, valid_label = Personal.split(df)

    # loading training data
    train_label = torch.from_numpy( train_label.to_numpy(dtype=int) ).squeeze()
    train_label, train_length = train_label[:, 0], train_label[:, 1]
    train_data = torch.from_numpy( train_data.to_numpy(dtype=np.float32) ).reshape(-1, Personal.n_mfcc, max(train_length))

    # loading validation data
    valid_label = torch.from_numpy( valid_label.to_numpy(dtype=int) ).squeeze()
    valid_label, valid_length = valid_label[:, 0], valid_label[:, 1]
    valid_data = torch.from_numpy( valid_data.to_numpy(dtype=np.float32) ).reshape(-1, Personal.n_mfcc, max(valid_length))

    # creating iterators
    train_iter = DataLoader(TensorDataset(train_data, train_label, train_length), batch_size=hyperparameters["batch_size"], shuffle=True)
    valid_iter = DataLoader(TensorDataset(valid_data, valid_label, valid_length), batch_size=hyperparameters["batch_size"])
    test_iter = valid_iter      # not enough data to have a separate dataset

    final_train_loss, final_train_acc, \
    final_valid_loss, final_valid_acc, \
    final_test_loss, final_test_acc = training_loop(    model, 
                                                        train_iter, valid_iter, test_iter, 
                                                        save=False,
                                                        **hyperparameters
                                                    )

    # summary statistics
    print("Model Summary:")
    summary(model, input_size=(Personal.n_mfcc, 216))
    print()
    print()
    
    predictions = torch.Tensor()
    labels = torch.Tensor()

    for batch, batch_labels, batch_lengths in test_iter:
        batch_predictions = model(batch.float())
        batch_predictions = torch.argmax(batch_predictions, dim=1)

        predictions = torch.cat( (predictions, batch_predictions.float()), dim=0 )
        labels = torch.cat( (labels, batch_labels.float()), dim=0 )
    
    predictions = predictions.detach().numpy().astype(int)
    labels = labels.detach().numpy().astype(int)
    CM = confusion_matrix(labels, predictions) 
    print("Confusion Matrix :")
    print(CM)
    print()

    # plotting confusion matrix nicer
    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))
    named_labels = []
    for label in labels:
        named_label = Metadata["mapping"][str(label)] 
        if named_label not in named_labels:
            named_labels.append(named_label)
    plot_confusion_matrix(CM, named_labels)


if __name__ == "__main__":
    transfer_learning()