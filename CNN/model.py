import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_mfcc):
        super(CNN, self).__init__()
        n_classes = 16 # We can group up Calm and Neutral to get 14 instead of 16 labels
        n_kernels = 32
        # input is of size (batch_size, 30, 216, 1)
        # applying 32 kernels of size of (4,10)
        self.conv1 = nn.Sequential(
<<<<<<< HEAD
                            nn.Conv2d(in_channels = 1, out_channels = n_kernels, kernel_size = (4,10)),
                            nn.BatchNorm2d(n_kernels),
                            nn.MaxPool2d(kernel_size=(4, 10), stride=(2, 2), padding = (1,4)),
                            nn.ReLU(),
                            nn.Dropout(p=0.2)
                            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=(4, 10)),
            nn.BatchNorm2d(n_kernels),
            nn.MaxPool2d(kernel_size=(4, 10), stride=(2, 2), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout(p=0.2)
                            )
        self.end = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(n_kernels),
            nn.ReLU(),
            nn.Dropout(p=0.2)
                            )
    def forward(self, x):
        batch_size = x.size()[0]
        #shape coming in [batch size, 30*216]
        x = torch.reshape(x, (batch_size,1, 30, 216))
        #print("Initial Size",x.size()) #[64,1,30,216]
        x = self.conv1(x)
        #print("After Conv1",x.size())  #[64,32,6,207]
        x = self.conv2(x)
        #print("after conv2", x.size()) #[64,32,5,47]
        x = self.conv2(x)
        #print("after conv3", x.size())  # [64,32,1,19]
        x = x.view(x.size()[0], -1)
        #print("after flattening x", x.size()) #[64, 608]
        self.fc1 = nn.Linear(x.size()[1],64)
        x = self.fc1(x)
        #print("after linear", x.size())
        #x= self.end(x)
        #print("after end", x.size())
        self.linear = nn.Sequential(nn.Linear(x.size()[1],16),nn.Softmax())
        x= self.linear(x)
        #print("at the very end", x.size())
=======
                    nn.Conv2d(in_channels=n_mfcc, out_channels=32, kernel_size=(4,10)),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.MaxPool2d(4),
                    nn.Dropout(p=0.2))
        self.output = nn.Sequential(
                        nn.Dropout(p=0.2),
                        nn.BatchNorm2d(4),
                        nn.ReLU(),
                        nn.Dropout(p=0.2)
                        )
    
    def forward(self, x):
        x = torch.reshape(x, (x.size()[0], 30, 216))
        print(x.size())
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        self.linear =  nn.Linear(x.size[0], 64)
        x = self.linear(x)
        x = self.output(x)
        self.end = nn.Sequential(nn.Linear(x.size[0], 16), nn.Softmax())
        x = self.end(x)
>>>>>>> 94464217cfbce72d8cc63d103703efac9cc98b4a
        return x
