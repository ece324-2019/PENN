import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_mfcc, n_classes):
        super(CNN, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(1,4)),
                    nn.MaxPool2d(kernel_size=(1,10), stride=(1,2)),
                    nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1,4)),
                    nn.MaxPool2d(kernel_size=(1,10), stride=(1,2)),
                    nn.ReLU()
                )
       
    
    # calulates output size
    def _output_size(self, inp_size, kernel_size, stride, padding):
        h, w = inp_size
        h_output = int((h - kernel_size + 2*padding) / stride) + 1
        w_output = int((w - kernel_size + 2*padding) / stride) + 1
        return h_output, w_output

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        for i in range(1):
            x = self.conv2(x)
        x = x.view(x.size(0), -1)
        fc = nn.Linear(x.size(1), self.n_classes)
        return nn.Softmax(dim=1)( fc(x) )
