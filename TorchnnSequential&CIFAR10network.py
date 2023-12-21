from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch
from torch.utils.tensorboard import SummaryWriter

class Cifar10_network(nn.Module):
    def __init__(self):
        super(Cifar10_network, self).__init__()
        """
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1) # 其中stride和padding的数值要根据cifar10的模型和torch.nn官网公式推导而出
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()    # 展平数据
        self.linear1 = Linear(in_features=1024, out_features=64)    # 将展平的数据集线性映射到长度为64的平坦数据集
        self.linear2 = Linear(in_features=64, out_features=10)  # 将数据集再次线性映射，获得长度为10的最终十个分类组
        """
        # 以上的模型均可用下述的Sequential替代：
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
        """
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        """
        # 以上的forward过程均可用以下一句话代替
        x = self.model(x)
        return x

test_network = Cifar10_network()

input = torch.ones((64, 3, 32, 32)) # 生成一系列的值为1, size为(64,3,32,32)的测试数据集；用以测试上述网络是否正确
output = test_network(input)
# print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(test_network, input)   # 将创建的网络可视化的展示在tensorboard上
writer.close()