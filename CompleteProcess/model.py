from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

# 搭建NerualNetwork
class Cifar10_network(nn.Module):
    def __init__(self):
        super(Cifar10_network, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x