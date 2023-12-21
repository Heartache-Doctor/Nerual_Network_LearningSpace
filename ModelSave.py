import torchvision
import torch

vgg16 = torchvision.models.vgg16()
# Save Method 1:
torch.save(vgg16, "./model/vgg16_method1.pth")  # 可以保存模型架构 + 参数

# Save Method 2:[官方推荐，空间要求小]
torch.save(vgg16.state_dict(), "./model/vgg16_method2.pth") # 以字典(dict)数据格式保存模型中的参数