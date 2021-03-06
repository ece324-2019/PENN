import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
   
    """ A Convolutional Neural Network
        it has two kernels that operate independently, 
        the idea being the smaller kernel will pick up on short bursts and the larger kernel will pick up on longer slurs
        For our particular sample rate, a kernel size of 10 ~ 0.1s and kernel size of 40 ~ 0.4s
        These independent kernels are then concatinated and run through a dense layer

        Note: we do not need to Softmax at the end because we are using nn.CrossEntropyLoss(), which does that automatically
    """
    
    def __init__(self, n_mfcc, n_classes, n_kernels=35):
        super(CNN, self).__init__()
        # In n_classes we can group up Calm and Neutral to get 14 instead of 16 labels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=(n_mfcc, 10), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2*n_kernels, kernel_size=(n_mfcc, 40), stride=4, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(3*n_kernels, n_classes)
        )

        # need to access for fine-tuning
        self.n_kernels = n_kernels
        self.n_classes = n_classes

    # calulates output size
    def _output_size(self, inp_size, kernel_size, stride=1, padding=0, dilation=1):
        
        # being able to take in either single number, list, or tuple
        kernel_size = kernel_size if type(kernel_size) == tuple or type(kernel_size) == list else (kernel_size, kernel_size)
        stride = stride if type(stride) == tuple or type(stride) == list else (stride, stride)
        padding = padding if type(padding) == tuple or type(padding) == list else (padding, padding)
        dilation = dilation if type(dilation) == tuple or type(dilation) == list else (dilation, dilation)

        h, w = inp_size
        h_output = int((h - dilation[0] * (kernel_size[0] - 1) + 2*padding[0] - 1) / stride[0]) + 1
        w_output = int((w - dilation[1] * (kernel_size[1] - 1) + 2*padding[1] - 1) / stride[1]) + 1
        return h_output, w_output

    def forward(self, x):
        x = x.unsqueeze(1)
        
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(x)
        pool1 = nn.MaxPool2d(kernel_size=(1, conv1_output.size(3)), stride=1, padding=0)
        pool2 = nn.MaxPool2d(kernel_size=(1, conv2_output.size(3)), stride=1, padding=0)
        conv1_output = pool1(conv1_output)
        conv2_output = pool2(conv2_output)
        #print(conv1_output.size())
        #print(conv2_output.size())
        output = torch.cat((conv1_output, conv2_output), 1)
        output = output.squeeze()
        #print(output.size())
        return self.fc(output)