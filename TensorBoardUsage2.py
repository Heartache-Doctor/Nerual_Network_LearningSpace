from torch.utils.tensorboard import SummaryWriter   #导入SummaryWriter类
import numpy as np
from PIL import Image

# SummaryWriter类的使用
writer = SummaryWriter("logs")  #创建SummaryWriter类对象，logs参数代表将生成的事件文件保存到当前路径下的logs文件夹

# writer.add_image()函数的使用(常用以观察不同阶段的训练结果)
# 由于add_image()函数的img_tensor形参规定了类，所以传统的Image.open返回的图片文件不适用，需要进行强制转换

image_path = "dataset\\beeVSant\\train\\ants\\0013035.jpg"

img_PIL = Image.open(image_path)    #这里的img变量对应的类型是PIL类，和add_image()函数不匹配
img_array = np.array(img_PIL)   #把img_PIL转换为numpy.ndarray类
print(img_array.shape)  #查看强制转换为np型后的数据的数据格式；发现和add_image()函数格式存在差异，故需要在函数后加入对应的数据格式参数

writer.add_image("Graph", img_array, 1, dataformats='HWC')  #形参分别对应：图像对应title，图片数据，迭代轮数global_step(与add_scalar中的x轴一个道理)，数据格式参数

writer.close()