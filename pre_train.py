# commanline arguments
from args import get_args

# preprocessing
from data_handling.load_data import *

# models
from Baseline.model import MLP, Average
from CNN.model import CNN
from RNN.model import RNN

# Pytorch
import torch
import torch.nn as nn

# Training Loop and evaluation functions
from utils.training import *

# Summary Statistics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torchsummary import summary

import json


def pre_train(args):
    
    Metadata = json.load(open(f"./data/Metadata.json", "r"))
    n_mfcc = Metadata["n_mfcc"]
    #audio_length = Metadata["max audio length"]
    audio_length = 216
    n_classes = len(Metadata["mapping"])

    model_name = args.model

    model = None
    hyperparameters = {}

    """ Specification for each model """
    if model_name.lower() == "mlp":
        model = MLP(input_size=n_mfcc*audio_length, output_size=n_classes)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : args.epochs,
            "batch_size" : args.batch_size,
            "lr" : 0.001 if args.lr == -1 else args.lr,
            "eval_every" : 10 if args.eval_every == -1 else args.eval_every
        }
        print("Created MLP baseline model")
    elif model_name.lower() == "average":
        model = Average(input_size=audio_length, output_size=n_classes)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : args.epochs,
            "batch_size" : args.batch_size,
            "lr" : 0.1 if args.lr == -1 else args.lr,
            "eval_every" : 10 if args.eval_every == -1 else args.eval_every
        }
        print("Created Averaging CNN baseline model")
    elif model_name.lower() == "cnn":
        model = CNN(n_mfcc=n_mfcc, n_classes=n_classes)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : args.epochs,
            "batch_size" : args.batch_size,
            "lr" : 0.001 if args.lr == -1 else args.lr,
            "eval_every" : 10 if args.eval_every == -1 else args.eval_every
        }
        print("Created CNN model")
    elif model_name.lower() == "rnn":
        model = RNN(n_mfcc=n_mfcc, n_classes=n_classes, hidden_size=100)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : args.epochs,
            "batch_size" : args.batch_size,
            "lr" : 0.01 if args.lr == -1 else args.lr,
            "eval_every" : 10 if args.eval_every == -1 else args.eval_every
        }
        print("Created RNN model")
    else:
        raise ValueError(f"Model '{model_name}' does not exist")
    

    train_iter, valid_iter, test_iter = load_data(  args.batch_size, 
                                                    n_mfcc, 
                                                    overfit=args.overfit
                                                )
    
    final_train_loss, final_train_acc, \
    final_valid_loss, final_valid_acc, \
    final_test_loss, final_test_acc = training_loop(    model, 
                                                        train_iter, valid_iter, test_iter, 
                                                        save_as=args.save_as, 
                                                        **hyperparameters
                                                    )

    # Final evaluation of Validation and Test Data
    print()
    print()
    print(f"Training Loss: {final_train_loss:.4f}\tTraining Accuracy: {final_train_acc*100:.2f}")
    print(f"Validation Loss: {final_valid_loss:.4f}\tValidation Accuracy: {final_valid_acc*100:.2f}")
    print(f"Testing Loss: {final_test_loss:.4f}\tTesting Accuracy: {final_test_acc*100:.2f}")
    print()
    print()

    # summary statistics
    print("Model Summary:")
    summary(model, input_size=(n_mfcc, audio_length))
    print()
    print()
    
    train_predictions, train_labels = get_predictions_and_labels(model, train_iter)
    valid_predictions, valid_labels = get_predictions_and_labels(model, valid_iter)
    test_predictions, test_labels = get_predictions_and_labels(model, test_iter)

    # plotting confusion matrices
    CM = confusion_matrix(train_labels, train_predictions) 
    plot_confusion_matrix(CM, list(Metadata["mapping"].values()), title="Training Data")

    CM = confusion_matrix(valid_labels, valid_predictions) 
    plot_confusion_matrix(CM, list(Metadata["mapping"].values()), title="Validation Data")

    CM = confusion_matrix(test_labels, test_predictions) 
    plot_confusion_matrix(CM, list(Metadata["mapping"].values()), title="Testing Data")

    # Accuracy for top 2 guesses
    acc = evaluate_top_k(model, valid_iter, 2)
    print(f"Accuracy for top 2 guesses: {100*acc:2f}")

if __name__ == "__main__":
    # get commandline arguments
    args = get_args()

    # run main program
    pre_train(args)