import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data(overfit=False):
    
    if overfit:
        overfit_data = pd.read_csv(f"{ROOT}/data/overfit_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        overfit_label = pd.read_csv(f"{ROOT}/data/overfit_label.tsv", sep='\t', index_col=0).reset_index(drop=True)
        return overfit_data, overfit_label, overfit_data, overfit_label, overfit_data, overfit_label
    else:
        train_data = pd.read_csv(f"{ROOT}/data/train_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        train_label = pd.read_csv(f"{ROOT}/data/train_label.tsv", sep='\t', index_col=0).reset_index(drop=True)

        valid_data = pd.read_csv(f"{ROOT}/data/valid_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        valid_label = pd.read_csv(f"{ROOT}/data/valid_label.tsv", sep='\t', index_col=0).reset_index(drop=True)

        test_data = pd.read_csv(f"{ROOT}/data/test_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        test_label = pd.read_csv(f"{ROOT}/data/test_label.tsv", sep='\t', index_col=0).reset_index(drop=True)

        return train_data, train_label, valid_data, valid_label, test_data, test_label

def one_hot_encode(df_label):
    
    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))
    
    tensor_label_ints = torch.from_numpy( df_label.values ).transpose(0, 1)
    tensor_labels_onehot = torch.zeros(tensor_label_ints.shape[1], len(Metadata["mapping"]))
    tensor_labels_onehot[range(tensor_labels_onehot.shape[0]), tensor_label_ints] = 1

    return tensor_labels_onehot

def load_data(batch_size, n_mfcc, audio_length, overfit=False):

    train_data, train_label, valid_data, valid_label, test_data, test_label = get_data(overfit=overfit)

    print(train_data.shape)

    train_data = torch.from_numpy( train_data.to_numpy(dtype=np.float32) ).reshape(-1, n_mfcc, audio_length)
    train_label = torch.from_numpy( train_label.to_numpy(dtype=int) ).squeeze()
    #train_label = one_hot_encode(train_label)
    
    valid_data = torch.from_numpy( valid_data.to_numpy(dtype=np.float32) ).reshape(-1, n_mfcc, audio_length)
    valid_label = torch.from_numpy( valid_label.to_numpy(dtype=int) ).squeeze()
    #valid_label = one_hot_encode(valid_label)

    test_data = torch.from_numpy( test_data.to_numpy(dtype=np.float32) ).reshape(-1, n_mfcc, audio_length)
    test_label = torch.from_numpy( test_label.to_numpy(dtype=int) ).squeeze()
    #test_label = one_hot_encode(test_label)
    
    train_iter = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(TensorDataset(valid_data, valid_label), batch_size=batch_size)
    test_iter = DataLoader(TensorDataset(test_data, test_label), batch_size=batch_size)

    return train_iter, valid_iter, test_iter

if __name__ == "__main__":
    
    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))

    """
    train_data, train_label, valid_data, valid_label = get_data()
    print(valid_data)
    print(valid_label)
    """

    """
    overfit_data, overfit_label, _, _ = get_data(overfit=True)
    print(overfit_data)
    print(overfit_label)
    """

    train_iter, valid_iter, test_iter = load_data(64, Metadata["n_mfcc"], Metadata["audio_length"])
    for i, (batch, labels) in enumerate(train_iter):
        if i > 2:
            break
        print(batch)
        print(batch.size())
        print(labels)
        print()
