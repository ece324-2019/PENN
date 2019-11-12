import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):

    """ Very simple MLP
    """

    def __init__(self, input_size, output_size, hidden_layers=[], seed=None):
        super(Baseline, self).__init__()

        # seed for reproducability
        if seed != None:
            torch.manual_seed(seed)

        # Creating MLP
        if len(hidden_layers) == 0:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Softmax()
            )
        elif len(hidden_layers) == 1:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                F.relu(),
                nn.Linear(hidden_layers[-1], output_size),
                nn.Softmax()
            )
        else:
            Hidden = []
            for i in range(len(hidden_layers)-1):
                Hidden.append( nn.Linear(hidden_layers[i], hidden_layers[i+1]) )
                Hidden.append( F.relu() )
            print(Hidden)
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                F.relu(),
                *Hidden,
                nn.Linear(hidden_layers[-1], output_size),
                nn.Softmax()
            )
        
        #print(self.mlp)
    
    def forward(self, x):
        return self.mlp(x)