# commanline arguments
from args import get_args

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from data_handling.Personal_preprocessor import Personal_Preprocessor
from utils.training import *

import numpy as np
import pandas as pd

# Plots and summary statistics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_personal_iters(batch_size):
    # preprocessing the personally collected data
    Personal = Personal_Preprocessor(seed=100, n_mfcc=30)
    Personal.rearrange()
    df = Personal.get_audio_data()
    #df = Personal.augment(df, frac=1)
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
    train_iter = DataLoader(TensorDataset(train_data, train_label, train_length), batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(TensorDataset(valid_data, valid_label, valid_length), batch_size=batch_size)
    test_iter = valid_iter      # not enough data to have a separate dataset

    return train_iter, valid_iter, test_iter

def fine_tune(model_name, save_as=None):

    # TODO: implemented fine-tuning for each model
    try:
        model = torch.load(f"{model_name}.pt")                    # loads model with all parameters frozen
        model.fc = nn.Linear(3*model.n_kernels, model.n_classes)    # re-initialize last linear layer
    except:
        raise TypeError("The loaded model needs to be a CNN")

    hyperparameters = {
                    "optimizer" : torch.optim.Adam,
                    "loss_fnc" : nn.CrossEntropyLoss(),
                    "epochs" : 20,
                    "batch_size" : 10,
                    "lr" : 0.001,
                    "eval_every" : 1
    }

    train_iter, valid_iter, test_iter = get_personal_iters(hyperparameters["batch_size"])

    final_train_loss, final_train_acc, \
    final_valid_loss, final_valid_acc, \
    final_test_loss, final_test_acc = training_loop(    model, 
                                                        train_iter, valid_iter, test_iter, 
                                                        save_as=save_as,
                                                        **hyperparameters
                                                    )

    # summary statistics
    print("Model Summary:")
    summary(model, input_size=(30, 216))
    print()
    print()
    
    predictions = torch.Tensor()
    labels = torch.Tensor()

    train_predictions, train_labels = get_predictions_and_labels(model, train_iter)
    valid_predictions, valid_labels = get_predictions_and_labels(model, valid_iter)

    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))

    # plotting confusion matrices
    CM = confusion_matrix(train_labels, train_predictions) 
    named_labels = []
    for label in train_labels:
        named_label = Metadata["mapping"][str(label)] 
        if named_label not in named_labels:
            named_labels.append(named_label)
    plot_confusion_matrix(CM, list(reversed(named_labels)), title="Transfer Learning")

    CM = confusion_matrix(valid_labels, valid_predictions) 
    named_labels = []
    for label in valid_labels:
        named_label = Metadata["mapping"][str(label)] 
        if named_label not in named_labels:
            named_labels.append(named_label)
    plot_confusion_matrix(CM, list(reversed(named_labels)), title="Validation Data")

def no_transfer_learning(model_name, save_as):
    
    model = torch.load(model_name)

    train_iter, valid_iter, test_iter = get_personal_iters(batch_size=10)
    train_predictions, train_labels = get_predictions_and_labels(model, train_iter)

    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))

    # plotting confusion matrices
    CM = confusion_matrix(train_labels, train_predictions)
    print(CM) 
    print(type(train_labels))
    print(train_predictions)
    named_labels = []
    for label in np.append(train_labels, train_predictions):
        named_label = Metadata["mapping"][str(label)] 
        if named_label not in named_labels:
            named_labels.append(named_label)
    print(named_labels)
    plot_confusion_matrix(CM, list(reversed(named_labels)), title="No Transfer Learning")

    if save_as != None:
        torch.save(model, f"{save_as}.pt")
        print(f"Model saved as '{save_as}.pt'")

if __name__ == "__main__":
    
    args = get_args()
    
    model_name, save_as = None, None
    if args.model_name != None:
        model_name = args.model_name[:-3] if args.model_name[-3:] == '.pt' else args.model_name
    if args.save_as != None:
        save_as = args.save_as[:-3] if args.save_as[-3:] == '.pt' else args.save_as
    fine_tune(model_name, save_as)
    #no_transfer_learning(model_name, save_as)