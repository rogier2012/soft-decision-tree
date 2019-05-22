import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        features = []
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        features.append(x)  # C1
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        features.append(x)  # C3
        x = F.max_pool2d(x, 2)
        features.append(x)  # S4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, features


net = LeNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

net.to(device)