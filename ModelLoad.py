import torch
import torchvision

# Load Method 1:
model = torch.load("model\\vgg16_method1.pth")
print(model)

# Load Method 2:
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("model\\vgg16_method2.pth"))
# model = torch.load("model\\vgg16_method2.pth")  # 此时load出的model变量是字典形式的参数罢了，需要拉取一个空白的模型架构把参数导入
print(vgg16)


"""
陷阱:
load自己写的model, 要在load前将这个model的class重申一下, 否则会报错说找不到对应的class. 或者直接把定义model的py文件import一下
eg. from filename import *
"""