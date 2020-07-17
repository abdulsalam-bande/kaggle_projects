import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size=10)
        self.batchnorm_conv1 = nn.BatchNorm2d(14)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=10)
        self.batchnorm_conv2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=10)
        self.batchnorm_conv3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=1600, out_features=128)
        self.batchnorm_fc1 = nn.BatchNorm1d(128)


        self.fc2 = nn.Linear(in_features=128,out_features=5)
        
    def forward(self,x):
        # Convolution Layer 1
        x = self.conv1(x)
        x = self.batchnorm_conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(5,5))


        #Convolution Layer 2
        x = self.conv2(x)
        x = self.batchnorm_conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(5,5))


        #Convolution Layer3
        x = self.conv3(x)
        x = self.batchnorm_conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(5,5))

        x = x.view(-1,self.num_flat_features(x))

        #Fully connected layer 1
        x = self.fc1(x)
        x = self.batchnorm_fc1(x)
        x = F.relu(x)

        #Fully Connected layer2
        x = self.fc2(x)

        return x

    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_fetures = 1
        for s in size:
            num_fetures *=s
        return num_fetures
    
    