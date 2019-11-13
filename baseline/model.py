import torch
import torch.nn as nn

class MLP(nn.Module):

    """ Very simple MLP
    """

    def __init__(self, input_size, output_size, hidden_layers=[], seed=None):
        super(MLP, self).__init__()

        # seed for reproducability
        if seed != None:
            torch.manual_seed(seed)

        # Creating MLP
        if len(hidden_layers) == 0:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Softmax(dim=1)
            )
        elif len(hidden_layers) == 1:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[-1], output_size),
                nn.Softmax(dim=1)
            )
        else:
            Hidden = []
            for i in range(len(hidden_layers)-1):
                Hidden.append( nn.Linear(hidden_layers[i], hidden_layers[i+1]) )
                Hidden.append( nn.ReLU() )
            print(Hidden)
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                nn.ReLU(),
                *Hidden,
                nn.Linear(hidden_layers[-1], output_size),
                nn.Softmax(dim=1)
            )
        
        #print(self.mlp)
    
    def forward(self, x):
        return self.mlp(x.view(x.size(0), -1))

class Average(nn.Module):

    def __init__(self, input_size, output_size, seed=None):
        super(Average, self).__init__()

        # seed for reproducability
        if seed != None:
            torch.manual_seed(seed)
        
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        print(x.size())
        average = x.mean(1)
        return nn.Softmax(dim=1)(self.fc(average).squeeze())