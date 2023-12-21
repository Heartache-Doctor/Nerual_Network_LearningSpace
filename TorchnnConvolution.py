"""
conv2d函数参数说明
stride: 卷积核的位移量,可为单个数值,也可以是一个元组(sH,sW).单个数值情况下,无论上下还是左右,偏移量都是这个值;元组情况下,一个负责横向位移量,一个负责纵向位移量
input[被卷的]: 有对应的shape要求:(batch_size, in_channels, height, width)
weight[卷积核]: 有对应的shape要求:(out_channels, in_channels/groups, height, width);其中out_channels代表输出通道数(=卷积核个数),groups代表卷积共享组数
padding: 表示填充的范围大小
padding_mode: 决定填充的内容
dilation: 空洞取集  eg.以往取3*3的子图和卷积核做运算;现在取的是5*5子图,但只取该5*5中的9个数据,每个中间间隔一格(dilation = 1)
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d

dataset = torchvision.datasets.CIFAR10("E:\\python_Vsc\\dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)  # 创造一层卷积层：输入的通道有三层，输出的通道有六层，卷积核的大小为3*3，偏移量为1，填充为0
    
    def forward(self, x):
        x = self.conv1(x)
        return x
    # 输入x，返回经过conv1卷积运算后的结果

test_nn = MyNetwork()

for data in dataloader:
    imgs, targets = data
    output = test_nn(imgs)
    # print(output.shape)
# 对每个数据集的每个img进行卷积计算操作