import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, n_mfcc, n_classes, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.GRU = nn.GRU(input_size=n_mfcc, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, hidden = self.GRU(x)
        hidden = hidden.squeeze(0)
        return nn.Softmax(dim=1)( self.fc(hidden).squeeze(1) )