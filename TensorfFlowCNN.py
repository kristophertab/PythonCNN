import torch.nn as nn
import torch.nn.functional as F

def my_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def my_relu(x):
    return x.clamp(min=0)

def my_liner(x):
    return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = my_relu(self.fc1(x))
        x = my_relu(self.fc2(x))
        x = self.fc3(x)
        return x