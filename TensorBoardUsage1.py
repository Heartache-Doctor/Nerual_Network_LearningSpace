from torch.utils.tensorboard import SummaryWriter   #导入SummaryWriter类

# SummaryWriter类的使用
writer = SummaryWriter("logs")  #创建SummaryWriter类对象，logs参数代表将生成的事件文件保存到当前路径下的logs文件夹

# writer.add_scalar()函数的使用(常用以绘制train/val loss)
for i in range(100):
    writer.add_scalar("Graph", 2*i, i)  #参数对应：图像的title；y轴对应值(value)；x轴对应值(step)
# 注意：图像title一致时，会被放到同一张图像中进行展示拟合。如果想要做两个不同的图，就要更改图像title

writer.close()



"""
如何调用事件文件：
打开conda终端，switch到当前环境下。
输入指令：tensorboard --logdir=E:\python_Vsc\logs(事件文件所在文件夹的绝对路径) --port=6007(自行指定显示tensorboard的本机端口，默认为6006)
访问localhost对应端口即可看到效果。
"""