from baseline.model import MLP, Average
from CNN.model import CNN

from data_handling.load_data import *
from utils import *

import torch
import torch.nn as nn

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

    # Creating plots
    plot_loss(np.linspace(0, epochs, len(training_error)), 
                train_error=training_error,
                valid_error=validation_error, 
                title=None)
    plot_accuracy(np.linspace(0, epochs, len(validation_acc)), 
                train_accuracy=training_acc,
                valid_accuracy=validation_acc, 
                title=None)
    
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
        torch.save(model, f"{model.__class__.__name__}.pt")
        print(f"Model saved as '{model.__class__.__name__}.pt'")

def main():
    
    Metadata = json.load(open(f"./data/Metadata.json", "r"))
    n_mfcc = Metadata["n_mfcc"]
    audio_length = Metadata["audio_length"]
    n_classes = len(Metadata["mapping"])

    model_name = "mlp"
    #model_name = "average"
    #model_name = "cnn"

    model = None
    hyperparameters = {}

    if model_name.lower() == "mlp":
        model = MLP(input_size=n_mfcc*audio_length, output_size=n_classes)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : 100,
            "batch_size" : 64,
            "lr" : 0.001,
            "eval_every" : 10
        }
    elif model_name.lower() == "average":
        model = Average(input_size=audio_length, output_size=n_classes)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : 500,
            "batch_size" : 64,
            "lr" : 0.1,
            "eval_every" : 10
        }
    elif model_name.lower() == "cnn":
        model = CNN(30)
        hyperparameters = {
            "optimizer" : torch.optim.Adam,
            "loss_fnc" : nn.CrossEntropyLoss(),
            "epochs" : 100,
            "batch_size" : 64,
            "lr" : 0.1,
            "eval_every" : 10
        }
    else:
        raise ValueError(f"Model '{model_name}' does not exist")

    train_iter, valid_iter, test_iter = load_data(hyperparameters["batch_size"], n_mfcc, audio_length, overfit=False)
    training_loop(model, train_iter, valid_iter, test_iter, **hyperparameters)

if __name__ == "__main__":
    main()