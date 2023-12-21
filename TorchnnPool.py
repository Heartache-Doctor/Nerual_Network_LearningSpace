"""
最大池化函数(MaxPool2d)参数说明
kernel_size: 池化核的大小,为单个数值则代表(x*x),为元组时代表(x*y)
ceil_mode: 对边缘数据是否池化选择.True->ceil_mode,将边缘的残缺部分依然进行池化运算,False->floor_mode,忽略该残缺部分
stride: 偏移量,可为单个数据也可为一个元组,默认情况下为kernel_size的大小

最大池化: 将池化核覆盖的子图取最大值作为res
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d

dataset = torchvision.datasets.CIFAR10("E:\\python_Vsc\\dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.pool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.pool1(x)
        return x

test_network = MyNetwork()
for data in dataloader:
    img, label = data
    output = test_network.forward(img)
    # print(output.shape)