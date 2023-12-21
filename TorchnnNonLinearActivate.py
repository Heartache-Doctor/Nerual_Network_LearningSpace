"""
ReLU函数参数说明
implace: True->将非线性变换后的值赋予给输入变量; False->将非线性变换后的值return,不改变输入变量(default = False)

ReLU函数对于input和output的变量shape没有要求
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import ReLU

dataset = torchvision.datasets.CIFAR10("E:\\python_Vsc\\dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.relu1 = ReLU()

    def forward(self, x):
        x = self.relu1(x)
        return x

test_network = MyNetwork()
for data in dataloader:
    img, label = data
    output = test_network(img)
    # print(output.shape)