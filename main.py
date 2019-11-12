from baseline.model import Baseline
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

            corr = torch.argmax(predictions, dim=1) == labels
            correct += int(corr.sum())
            total_samples += labels.size(0)

    return float(running_loss) / total_batches, float(correct) / total_samples

def training_loop(model, train_iter, valid_iter, loss_fnc, epochs, batch_size, lr, eval_every):
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_error = []
    validation_error = []
    training_acc = []
    validation_acc = []

    print("Start Training")
    # training loop
    total_batches = 0
    running_loss = 0.0
    running_acc = 0.0
    for e in range(epochs):
        for i, (batch, labels) in enumerate(train_iter):
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
                
                training_error.append( running_loss/eval_every )
                validation_error.append( loss )

                training_acc.append( running_acc / (eval_every*labels.size()[0])  )
                validation_acc.append( acc )
                
                print(f"epoch: {e+1:4d}\tbatch: {i+1:5d}\tloss: {training_error[-1]:.4f}\tAcc: {acc:.4f}")

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

def main():
    
    model = Baseline(input_size=216, output_size=16)

    hyperparameters = {
        "loss_fnc" : nn.CrossEntropyLoss(),
        "epochs" : 100,
        "batch_size" : 64,
        "lr" : 0.001,
        "eval_every" : 10
    }

    train_iter, valid_iter = load_data(hyperparameters["batch_size"])
    training_loop(model, train_iter, valid_iter, **hyperparameters)

if __name__ == "__main__":
    main()