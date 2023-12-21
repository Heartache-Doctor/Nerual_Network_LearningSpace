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

optim = torch.optim.SGD(test_network.parameters(), lr=0.01)   # 选择使用梯度下降优化器

for epoch in range(20):
    running_loss = 0.0  # 可视化对相同数据的每一轮累计误差值
    for data in dataloader:
        imgs, labels = data
        outputs = test_network(imgs)
        result_loss = loss(outputs, labels)
        optim.zero_grad()   # 使用优化器将network结点梯度初始化全为0
        result_loss.backward()
        optim.step()    # 对各个结点进行梯度调优
        running_loss += result_loss
    # 每次循环梯度都必须清零, 因为需要被调优的参数在本次梯度step调优更新后, 就不再需要这一轮的梯度数据了
    print(running_loss)
# 对dataloader中的数据集进行20轮训练