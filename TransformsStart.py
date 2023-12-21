from PIL import Image
img_path = "dataset\\beeVSant\\train\\ants\\0013035.jpg"
img = Image.open(img_path)  # PIL类型图片

import cv2
cv_img = cv2.imread(img_path)   # numpy.ndarray类型图片

from torchvision import transforms  #用来进行图片处理
# transforms实质为一个py文件，其本质是一个图片处理的工具箱。

# 通过ToTensor类：
# 1 学习transforms工具使用方法
# 2 并理解tensor数据类型和普通数据类型区别

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()    #创建一个ToTensor类的对象
img_tensor = tensor_trans(img)  #调用call函数，将PIL类型图片转换为tensor类型图片
# print(img_tensor)

writer.add_image("tensor_img", img_tensor, 2)   #添加tensor类型图片文件
writer.close()

