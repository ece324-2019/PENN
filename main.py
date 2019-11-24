# commanline arguments
from args import get_args

# preprocessing
from data_handling.load_data import *

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

def evaluate(model, data_loader, loss_fnc):
    running_loss = 0.0
    total_batches = 0.0
    correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for batch, labels, lengths in data_loader:

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
        for i, (batch, labels, lengths) in enumerate(train_iter):
            evaluated_data += labels.size(0)
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
    
    if save:
        torch.save(model, f"{model_name.lower()}.pt")
        print(f"Model saved as '{model_name.lower()}.pt'")

    final_train_loss, final_train_acc = training_error[-1], training_acc[-1]        # Training
    final_valid_loss, final_valid_acc = evaluate(model, valid_iter, loss_fnc)       # Validation
    final_test_loss, final_test_acc = evaluate(model, test_iter, loss_fnc)          # Testing
    
    return final_train_loss, final_train_acc, final_valid_loss, final_valid_acc, final_test_loss, final_test_acc


def main(args):
    
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
                                                        save=args.save, 
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
    print("Accuracy Score :")
    print(accuracy_score(labels, predictions))
    print()
    print()
    #print("Report : ")
    #print(classification_report(CM, predictions))

    # plotting confusion matrix nicer
    plot_confusion_matrix(results, list(Metadata["mapping"].values()))

if __name__ == "__main__":
    # get commandline arguments
    args = get_args()

    # run main program
    main(args)