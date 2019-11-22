# commanline arguments
from args import get_args

# preprocessing
from data_handling.load_data import *
from data_handling.RAVDESS_preprocessor import RAVDESS_Preprocessor
from data_handling.SAVEE_preprocessor import SAVEE_Preprocessor
from data_handling.TESS_preprocessor import TESS_Preprocessor

# models
from baseline.model import MLP, Average
from CNN.model import CNN
from RNN.model import RNN

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Plots and summary statistics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from utils import *

import json

def process_datasets(*dataset_processors):
    le = None
    append = False
    for DATASET in dataset_processors:
        print("Processing", DATASET.dataset)
        df, n_mfcc, audio_length = DATASET.mfcc_conversion()
        le = DATASET.split_data(df, n_mfcc, audio_length, le=le, append=append)
        append = True
        print(n_mfcc, audio_length)

def evaluate(model, data_loader, loss_fnc):
    running_loss = 0.0
    total_batches = 0.0
    correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for batch, labels in data_loader:

            predictions = model(batch.float())
            
            running_loss += loss_fnc(input=predictions, target=labels)
            total_batches += 1

            corr = ( torch.argmax(predictions, dim=1) == labels )
            correct += int(corr.sum())
            total_samples += labels.size(0)

    return float(running_loss) / total_batches, float(correct) / total_samples

def training_loop(model, train_iter, valid_iter, test_iter, optimizer, loss_fnc, epochs, batch_size, lr, eval_every, save=False):
    
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)

    training_error = []
    validation_error = []
    training_acc = []
    validation_acc = []

    print("Start Training")
    # training loop
    evaluated_data = 0
    total_batches = 0
    running_loss = 0.0
    running_acc = 0.0
    for e in range(epochs):
        for i, (batch, labels) in enumerate(train_iter):
            evaluated_data += labels.size()[0]
            total_batches += 1

            # re-initializing optimizer
            optimizer.zero_grad()

            # forward + backward + update
            predictions = model(batch.float())
            loss = loss_fnc(input=predictions, target=labels)
            loss.backward()
            optimizer.step()

            # accumulated loss for the batch
            running_loss += loss

            # accumulated accuracy
            corr = torch.argmax(predictions, dim=1) == labels
            running_acc += int(corr.sum())
            
            # calculate stats
            if total_batches % eval_every == 0:
                model.eval()
                loss, acc = evaluate(model, valid_iter, loss_fnc)
                
                training_error.append( running_loss / eval_every )
                validation_error.append( loss )

                training_acc.append( running_acc / evaluated_data  )
                validation_acc.append( acc )
                
                print(f"epoch: {e+1:4d}\tbatch: {i+1:5d}\tloss: {training_error[-1]:.4f}\tAcc: {acc:.4f}")

                evaluated_data = 0
                running_loss = 0.0
                running_acc = 0.0
                model.train()
            
    print("End Training")

    model_name = model.__class__.__name__

    # Creating plots
    plot_loss(np.linspace(0, epochs, len(training_error)), 
                train_error=training_error,
                valid_error=validation_error, 
                title=model_name)
    plot_accuracy(np.linspace(0, epochs, len(validation_acc)), 
                train_accuracy=training_acc,
                valid_accuracy=validation_acc, 
                title=model_name)
    
    # Final evaluation of Validation and Test Data
    print()
    print()
    print(f"Training Loss: {training_error[-1]:.4f}\tTraining Accuracy: {training_acc[-1]*100:.2f}")
    loss, acc = evaluate(model, valid_iter, loss_fnc)
    print(f"Validation Loss: {loss:.4f}\tValidation Accuracy: {acc*100:.2f}")
    loss, acc = evaluate(model, test_iter, loss_fnc)
    print(f"Testing Loss: {loss:.4f}\tTesting Accuracy: {acc*100:.2f}")
    print()
    print()

    if save:
        torch.save(model, f"{model_name.lower()}.pt")
        print(f"Model saved as '{model_name.lower()}.pt'")

def main(args):
    
    Metadata = json.load(open(f"./data/Metadata.json", "r"))
    n_mfcc = Metadata["n_mfcc"]
    audio_length = Metadata["audio_length"]
    n_classes = len(Metadata["mapping"])

    model_name = args.model

    model = None
    hyperparameters = {}

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

    train_iter, valid_iter, test_iter = load_data(  hyperparameters["batch_size"], 
                                                    n_mfcc, audio_length, 
                                                    overfit=args.overfit
                                                )
    
    training_loop(  model, 
                    train_iter, valid_iter, test_iter, 
                    save=args.save, 
                    **hyperparameters
                )

    # summary statistics
    print("Model Summary:")
    #summary(model, input_size=(1, n_mfcc, audio_length))
    print()
    print()
    
    predictions = torch.Tensor()
    labels = torch.Tensor()

    for batch, batch_labels in test_iter:
        batch_predictions = model(batch.float())
        batch_predictions = torch.argmax(batch_predictions, dim=1)

        predictions = torch.cat( (predictions, batch_predictions.float()), dim=0 )
        labels = torch.cat( (labels, batch_labels.float()), dim=0 )
    
    predictions = predictions.detach().numpy().astype(int)
    labels = labels.detach().numpy().astype(int)
    results = confusion_matrix(labels, predictions) 
    print('Confusion Matrix :')
    print(results) 
    """
    print('Accuracy Score :', accuracy_score(labels, predictions))
    print()
    print()
    print('Report : ')
    print(classification_report(labels, predictions))
    """

    # plotting confusion matrix nicer
    plot_confusion_matrix(results, list(Metadata["mapping"].values()))

if __name__ == "__main__":
    args = get_args()
    
    if args.preprocess:
        
        RAVDESS = RAVDESS_Preprocessor(seed=100)
        df, n_mfcc, audio_length = RAVDESS.mfcc_conversion()
        df = RAVDESS.augment(df)
        RAVDESS.split_data(df, n_mfcc, audio_length, le=None, append=False)

        """ data preprocessing 
        RAVDESS = RAVDESS_Preprocessor(seed=100)
        SAVEE = SAVEE_Preprocessor(seed=100)
        TESS = TESS_Preprocessor(seed=100)
        #RAVDESS.rearrange()
        #SAVEE.rearrange()
        #TESS.rearrange()
        #process_datasets(RAVDESS, SAVEE, TESS)
        df, n_mfcc, audio_length = TESS.mfcc_conversion()
        le = TESS.split_data(df, n_mfcc, audio_length, le=None, append=False)
        """
    else:
        main(args)