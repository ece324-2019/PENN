from data_handling.RAVDESS_preprocessor import RAVDESS_Preprocessor
from data_handling.SAVEE_preprocessor import SAVEE_Preprocessor
from data_handling.TESS_preprocessor import TESS_Preprocessor

import pandas as pd

import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def normalize():
    print("Normalizing Data")
    
    # normalizing training data
    train_data = pd.read_csv(f"{ROOT}/data/train_data.tsv", sep='\t', index_col=0)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean)/std
    train_data.to_csv(path_or_buf=f"{ROOT}/data/train_data.tsv", sep='\t', mode='w', index=True, header=True)

    # normalizing all other data with the training mean and std
    valid_data = pd.read_csv(f"{ROOT}/data/valid_data.tsv", sep='\t', index_col=0)
    valid_data = (valid_data - mean)/std
    valid_data.to_csv(path_or_buf=f"{ROOT}/data/valid_data.tsv", sep='\t', mode='w', index=True, header=True)

    test_data = pd.read_csv(f"{ROOT}/data/test_data.tsv", sep='\t', index_col=0)
    test_data = (test_data - mean)/std
    test_data.to_csv(path_or_buf=f"{ROOT}/data/test_data.tsv", sep='\t', mode='w', index=True, header=True)

    overfit_data = pd.read_csv(f"{ROOT}/data/overfit_data.tsv", sep='\t', index_col=0)
    overfit_data = (overfit_data - mean)/std
    overfit_data.to_csv(path_or_buf=f"{ROOT}/data/overfit_data.tsv", sep='\t', mode='w', index=True, header=True)

    # saving mean and std deviation
    Metadata = json.load(open(f"{ROOT}/data/Metadata.json", "r"))
    Metadata["mean"] = mean.tolist()
    Metadata["std"] = std.tolist()
    with open(f"{ROOT}/data/Metadata.json", "w+") as g:
        json.dump(Metadata, g, indent=4)

if __name__ == "__main__":
    le = None

    print("Processing RAVDESS dataset")
    RAVDESS = RAVDESS_Preprocessor(seed=100, n_mfcc=30)
    #RAVDESS.rearrange()
    df = RAVDESS.mfcc_conversion()

    """
    # only augmenting male data to keep dataset balanced
    male_labels = [f"male_{emotion.lower()}" for emotion in RAVDESS.Metadata["emotion"].values()]
    df_males = df.loc[df["label"].isin(male_labels)] #only males
    df_females = df[~df["label"].isin(male_labels)]  #only females
    df_males = RAVDESS.augment(df_males, frac=0.65)
    df = pd.concat([df_males, df_females], ignore_index=True)
    """

    le = RAVDESS.split_data(df, le=le, append=False, equalize=False)

    """
    print("Processing SAVEE dataset")
    SAVEE = SAVEE_Preprocessor(seed=100, n_mfcc=30)
    #SAVEE.rearrange()
    df = SAVEE.mfcc_conversion()
    df = SAVEE.augment(df, frac=1)
    le = SAVEE.split_data(df, le=le, append=True)

    print("Processing TESS dataset")
    TESS = TESS_Preprocessor(seed=100, n_mfcc=30)
    #TESS.rearrange()
    df = TESS.mfcc_conversion()
    le = TESS.split_data(df, le=le, append=True)
    """

    # Normalize all data (not just the TESS data)
    normalize()