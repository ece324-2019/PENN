import numpy as np
import matplotlib.pyplot as plt

def plot_loss(x, train_error=None, valid_error=None, title=None):
    if train_error != None:
        plt.plot(x, train_error, label="Training Error")
    if valid_error != None:
        plt.plot(x, valid_error, label="Validation Error")
    
    if title == None:
        plt.title("Training Loss")
    else:
        plt.title(title)
    
    plt.xlabel("Epochs")
    plt.xlim(left=0)
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()
    plt.clf()

def plot_accuracy(x, train_accuracy=None, valid_accuracy=None, title=None):
    if train_accuracy != None:
        plt.plot(x, train_accuracy, label="Training Accuracy")
    if valid_accuracy != None:
        plt.plot(x, valid_accuracy, label="Validation Accuracy")
    
    if title == None:
        plt.title("Accuracy")
    else:
        plt.title(title)

    plt.xlabel("Epochs")
    plt.xlim(left=0)
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(linestyle='-', axis='y')
    plt.legend(loc="lower right")
    plt.show()
    plt.clf()