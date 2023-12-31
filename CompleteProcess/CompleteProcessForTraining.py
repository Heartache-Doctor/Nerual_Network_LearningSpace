import torchvision
from torch.utils.data import DataLoader
from model import Cifar10_network
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
nerual_network = Cifar10_network()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(nerual_network.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0    # 用来记录训练次数
total_test_step = 0 # 用来记录测试次数
epoch = 10  # 设置训练轮数

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("------第{}轮训练：------".format(i+1))

    # 训练步骤开始：
    nerual_network.train()  # 将网络设置为训练模式
    for data in train_dataloader:
        img, label = data
        outputs = nerual_network(img)
        loss = loss_fn(outputs, label)
        
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss：{}".format(total_train_step, loss))

            writer.add_scalar("train_loss", loss, total_train_step)

    # 测试步骤开始：
    nerual_network.eval()   # 将网络设置为训练模式
    total_test_loss = 0
    total_accuracy = 0  # 累计正确预测次数
    
    with torch.no_grad():   # 测试阶段不改变梯度
        for data in test_dataloader:
            img, label = data
            outputs = nerual_network(img)
            loss = loss_fn(outputs, label)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == label).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss为: {}".format(total_test_loss))
    print("整体测试集上的正确率为：{}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(nerual_network, "../model/CIFAR10_model{}.pth".format(i))    # 每轮训练后，保存模型

writer.close()


