import torch
import torch.nn as nn

class RNN(nn.Module):

    """ a Recurrent Neural Network
        Nothing fancy, just a Gated Recurrent Unit with a dense layer at the end

        Note: we do not need to Softmax at the end because we are using nn.CrossEntropyLoss(), which does that automatically
    """

    def __init__(self, n_mfcc, n_classes, hidden_size=100):
        super(RNN, self).__init__()
        self.GRU = nn.GRU(input_size=n_mfcc, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, hidden = self.GRU(x)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)