from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("E:\\python_Vsc\\dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

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
    
test_network = Cifar10_network()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, labels = data
    outputs = test_network(imgs)
    # print(outputs)
    # print(labels)
    result_loss = loss(outputs, labels)
    result_loss.backward()  # 该函数将最终结果求出的loss值反向传播到网络的各节点，求出各节点的loss和对应的grad值，并存储在各个节点中
    # print(result_loss)
