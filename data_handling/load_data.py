import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import json
import os

# TODO: Make it so it doesn't matter what file you run this from

def get_data(overfit=False):
    
    if overfit:
        overfit_data = pd.read_csv("./data/overfit_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        overfit_label = pd.read_csv("data/overfit_label.tsv", sep='\t', index_col=0).reset_index(drop=True)
        return overfit_data, overfit_label, overfit_data, overfit_label
    else:
        train_data = pd.read_csv("./data/train_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        train_label = pd.read_csv("./data/train_label.tsv", sep='\t', index_col=0).reset_index(drop=True)

        valid_data = pd.read_csv("./data/valid_data.tsv", sep='\t', index_col=0).reset_index(drop=True)
        valid_label = pd.read_csv("./data/valid_label.tsv", sep='\t', index_col=0).reset_index(drop=True)

        return train_data, train_label, valid_data, valid_label

def one_hot_encode(df_label):
    
    Mapping = json.load(open("./data/Mapping.json", "r"))
    
    tensor_label_ints = torch.from_numpy( df_label.values ).transpose(0, 1)
    tensor_labels_onehot = torch.zeros(tensor_label_ints.shape[1], len(Mapping))
    tensor_labels_onehot[range(tensor_labels_onehot.shape[0]), tensor_label_ints] = 1

    return tensor_labels_onehot

def load_data(batch_size):

    train_data, train_label, valid_data, valid_label = get_data()
    
    train_data = torch.from_numpy( train_data.to_numpy() )
    train_label = torch.from_numpy( train_label.to_numpy() ).squeeze()
    #train_label = one_hot_encode(train_label)
    
    valid_data = torch.from_numpy( valid_data.to_numpy() )
    valid_label = torch.from_numpy( valid_label.to_numpy() ).squeeze()
    #valid_label = one_hot_encode(valid_label)
    
    train_iter = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(TensorDataset(valid_data, valid_label), batch_size=batch_size)

    return train_iter, valid_iter

if __name__ == "__main__":
    
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

    load_data(64)