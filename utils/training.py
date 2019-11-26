import torch

from utils.plot import *

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

def evaluate_top_k(model, data_loader, k):
    correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for batch, labels, lengths in data_loader:

            predictions = model(batch.float())

            top_k = torch.topk(predictions, k, dim=1).indices
            for i in range(k):
                corr = top_k[:, i] == labels
                correct += int(corr.sum())
            total_samples += labels.size(0)

    return float(correct) / total_samples


def get_predictions_and_labels(model, data_loader):
    predictions = torch.Tensor()
    labels = torch.Tensor()
    for batch, batch_labels, batch_lengths in data_loader:
        batch_predictions = model(batch.float())
        batch_predictions = torch.argmax(batch_predictions, dim=1)

        predictions = torch.cat( (predictions, batch_predictions.float()), dim=0 )
        labels = torch.cat( (labels, batch_labels.float()), dim=0 )
    
    predictions = predictions.detach().numpy().astype(int)
    labels = labels.detach().numpy().astype(int)

    return predictions, labels


def training_loop(  model, train_iter, valid_iter, test_iter, \
                    optimizer, loss_fnc, epochs, batch_size, lr, \
                    eval_every, save_as=None, plot=True):
    
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
    
    if save_as != None:
        torch.save(model, f"{save_as}.pt")
        print(f"Model saved as '{save_as}.pt'")

    model.eval()
    final_train_loss, final_train_acc = evaluate(model, train_iter, loss_fnc)       # Training
    final_valid_loss, final_valid_acc = evaluate(model, valid_iter, loss_fnc)       # Validation
    final_test_loss, final_test_acc = evaluate(model, test_iter, loss_fnc)          # Testing
    
    return final_train_loss, final_train_acc, final_valid_loss, final_valid_acc, final_test_loss, final_test_acc