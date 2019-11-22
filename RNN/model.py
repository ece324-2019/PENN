import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, n_mfcc, n_classes, hidden_size=100):
        super(RNN, self).__init__()
        self.GRU = nn.GRU(input_size=n_mfcc, hidden_size=hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, hidden = self.GRU(x)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)