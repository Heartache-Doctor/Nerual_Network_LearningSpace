import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()    # 从torchvision库中拉取vgg16模型架构+训练后的参数

# 我们想让这个能分1000个类别的模型，现在能使用到CIFAR10数据集上[finetunning]

vgg16.classifier.add_module("7", nn.Linear(in_features=1000, out_features=10))    # finetunning，将原先的1000个类别，再次线性转换为10个类别
del vgg16.classifier[7] # 删除刚刚添加的第7层finetunning
vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10)  # finetunning的另一种改法，不添加新层，在原先的最后一层线性层上直接修改
print(vgg16)