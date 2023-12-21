import torch
from torch import nn


class MyNetwork(nn.Module):    # modules是nn下的一个可继承父类
    def __init__(self):
        super(MyNetwork, self).__init__()   # 继承父类中的__init__函数

    def forward(self, input):   #定义现在的神经网络中，在不同层间传递输入信息的function
        output = input + 1
        return output
    


test = MyNetwork()
input = torch.tensor(1)
output = test(input)
print(output)
