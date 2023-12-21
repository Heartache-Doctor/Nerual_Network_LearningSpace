import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])  #使用totensor工具将PIL图片数据转化为Tensor图片数据

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_trans, download=True)
#在torch官网的dataset中查找对应dataset所需要的参数为什么


"""
print(test_set[0])  #取出实验数据集的第一个元素，发现返回两个值，一个是PIL图片文件，一个是label对应的classes序号
print(test_set.classes) #所有label共同组成的一个列表[classes]

img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target]) #从classes中根据序号target提取对应的label名
img.show()
"""
writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test", img, i)

writer.close()