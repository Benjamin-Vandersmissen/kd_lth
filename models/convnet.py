import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.pool = torch.nn.AvgPool2d(2)
        self.lin = torch.nn.Linear(1600, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)
        return x