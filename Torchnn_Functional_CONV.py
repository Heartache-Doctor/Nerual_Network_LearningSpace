"""
torch.nn.functional是没有封装较为原始的torch.nn库。

conv2d函数的参数说明
stride: 卷积核的位移量，可为单个数值，也可以是一个元组(sH,sW)。单个数值情况下，无论上下还是左右，偏移量都是这个值；元组情况下，一个负责横向位移量，一个负责纵向位移量
input[被卷的]: 有对应的shape要求:(batch_size, in_channels, height, width)
weight[卷积核]: 有对应的shape要求:(out_channels, in_channels/groups, height, width);其中out_channels代表输出通道数(=卷积核个数)，groups代表卷积共享组数
padding: 表示填充的范围大小
padding_mode: 决定填充的内容
dilation: 空洞取集  eg.以往取3*3的子图和卷积核做运算；现在取的是5*5子图，但只取该5*5中的9个数据，每个中间间隔一格(dilation = 1)
"""

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
input = torch.reshape(input, (1, 1, 5, 5))  # 将input和kernel的shape变得符合conv2d函数的参数
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, stride=1)  # 此处对conv2d函数的参数书写并不完整
# print(output)