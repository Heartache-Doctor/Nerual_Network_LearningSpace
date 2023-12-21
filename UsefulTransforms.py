from PIL import Image
img = Image.open("dataset\\beeVSant\\train\\ants\\5650366_e22b7e1065.jpg")

from torchvision import transforms

# ToTensor类
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)

# Normalize类   (对图片进行归一化)
# print(img_tensor[0][0][0])  #tensor型图片文件分三个维度，分别为宽度，高度，通道数。这里取[0][0][0]，其实就是取出第一个通道的第一个像素
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 参数：均值，方差。因为图片为RGB三色，传输有三个信道，所以要写一个三维向量
img_norm = trans_norm(img_tensor)   # 注意：这里的参数只能是tensor类图片文件
# print(img_norm[0][0][0])    #通过输出归一化前后的值，辅助理解归一化的意义
# 对于每个信道，归一化公式为：output[channel] = (input[channel] - mean[channel]) / std[channel]
# mean为均值，std为方差

# Resize类  (对图片进行缩放处理)
trans_resize = transforms.Resize((512, 512))    # 参数为图片需要变成的大小[长、宽]
img_resize = trans_resize(img)  # 注意：这里的参数只能为PIL类图片文件，返回的缩放后的图片类型也依然是PIL

# Compose - Resize - 2  (resize单参数情况的缩放)
trans_resize2 = transforms.Resize(512)  # 将图片[长、宽]中的较小值拉伸为参数值，同时保证图形的长宽比不变
trans_compose = transforms.Compose([trans_resize2, trans_tensor])   # compose参数需要为一个transforms类的列表
# compose作用，将前一一个类的输出作为后一个类的输入使用。
img_resize2 = trans_compose(img)

# RandomCrop    (随机裁剪)
trans_random = transforms.RandomCrop(20)   # 单参数(x):随机裁剪的大小为[x,x];双参数(x,y):随机裁剪的大小为[x,y]
# 注意：randomcrop也只接受PIL类图片文件
trans_compose2 = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_compose2(img)


