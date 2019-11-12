import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_mfcc):
        super(CNN, self).__init__()
        n_classes = 16 #didnt add up calm and neutral yet
        kernerl_dim = (4,10)
        #input is of size (30, 216,1) ie: 2D matrix of 30 MFCC bands by 216 audio length
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = n, out_channels = 32, kernel_size = kernel_dim),
                    nn.BatchNorm2d(n),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_dim),
                    nn.Dropout(p = 0.2))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        return x






