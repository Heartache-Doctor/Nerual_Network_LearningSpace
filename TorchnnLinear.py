"""
线性变换在网络层与层之间的数据传输中很常见。

Linear函数参数说明:
in_features: 输入样本的大小[一般要折算成线性的一维向量来考虑长度]
out_features: 输出的样本需要的大小[一般指的是下一层神经网络有多少的神经元]
bias: False->线性变换不加入bias; True->线性变换加入bias计算(default = True)
"""

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("E:\\python_Vsc\\dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


test_network = MyNetwork()
for data in dataloader:
    img, target = data
    # output = torch.reshape(img, (1,1,1,-1))    # 不确定的参数直接设置为-1
    output = torch.flatten(img) # 效果和上面注释语句一致
    output = test_network(output)
    # print(output)
    