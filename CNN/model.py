import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_mfcc, n_classes, n_kernels=50):
        super(CNN, self).__init__()
        # In n_classes we can group up Calm and Neutral to get 14 instead of 16 labels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=(n_mfcc,10), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=(n_mfcc, 50), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(n_kernels, 16),
            nn.Softmax(dim=1)
        )
        
    
    # calulates output size
    def _output_size(self, inp_size, kernel_size, stride, padding):
        h, w = inp_size
        h_output = int((h - kernel_size + 2*padding) / stride) + 1
        w_output = int((w - kernel_size + 2*padding) / stride) + 1
        return h_output, w_output

    def forward(self, x):
        x = x.unsqueeze(1)
        
        conv1_output = self.conv1(x)
        pool1 = nn.MaxPool2d(kernel_size=(1, conv1_output.size(3)), stride=1, padding=0)
        conv1_output = pool1(conv1_output).squeeze()
        
        #conv2_output = self.conv1(x)
        #pool2 = nn.MaxPool2d(kernel_size=(1, conv2_output.size(3)), stride=1, padding=0)
        #conv2_output = pool2(conv2_output).squeeze()
        
        #output = torch.cat((conv1_output, conv2_output), 1)

        return self.fc(conv1_output)
