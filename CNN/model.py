import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_mfcc):
        super(CNN, self).__init__()
        n_classes = 16
        # We can group up Calm and Neutral to get 14 instead of 16 labels
        # input is of size (batch_size, 30, 216, 1)
        # applying 32 kernels of size of (4,10)
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = n_mfcc, out_channels = 32, kernel_size = (4,10)),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.MaxPool2d(4),
                    nn.Dropout(p = 0.2))
        self.output = nn.Sequential(
                        nn.Dropout( p =0.2),
                        nn.BatchNorm2d(4),
                        nn.ReLU(),
                        nn.Dropout(p=0.2)
                        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        self.linear =  nn.Linear(x.size[0],64)
        x = self.linear(x)
        x = self.output(x)
        self.end = nn.Sequential(nn.Linear(x.size[0], 16),nn.Softmax())
        x = self.end(x)
        return x
