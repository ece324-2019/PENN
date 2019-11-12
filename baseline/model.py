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
                nn.Softmax()
            )
        elif len(hidden_layers) == 1:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[-1], output_size),
                nn.Softmax()
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
                nn.Softmax()
            )
        
        #print(self.mlp)
    
    def forward(self, x):
        return self.mlp(x)

class Average(nn.Module):

    def __init__(self, input_size, output_size, seed=None):
        super(Average, self).__init__()

        # seed for reproducability
        if seed != None:
            torch.manual_seed(seed)
        
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        #x = torch.reshape(x, (x.size()[0], 30, 216))
        #average = x.mean(1)
        x = torch.reshape(x, (x.size()[0], 216, 30))
        average = x.mean(2)
        return nn.Softmax( self.fc(average).squeeze() )