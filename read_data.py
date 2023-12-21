from torch.utils.data import Dataset    #Dataset为一个可继承的抽象类
from PIL import Image   #Image类可以加载图片文件
import os

class MyData(Dataset):
    
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir) #拼接数据集根路径和标签路径
        self.img_path = os.listdir(self.path)   #从拼接好的路径里面取出数据名，并将数据名组成数组返回给img_path变量
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]   #从img_path数组中取出第idx个图片文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)   #把图片文件名和路径拼接，生成文件的完整路径
        img = Image.open(img_item_path) #从刚刚拼接好的图片路径中取出图片文件，把它存在变量img中
        label = self.label_dir  #因为数据集的格式为数据所在文件夹名为数据的label，所以label就是label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)   #img_path数组的大小就是图片文件的数量，返回数组大小即可
    
root_dir = "dataset\\beeVSant\\train"  #数据集根路径
ants_label_dir = "ants" #标签路径
ants_dataset = MyData(root_dir, ants_label_dir) #生成对象ants_dataset
img, label = ants_dataset[0]    #自动调用__getitem__，返回的img和label对应赋值给变量img和label
img.show()  #展示img文件所代表的图片

bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset #合并数据集，数据集中编号按照加法顺序排列
img, label = train_dataset[0]
img.show()  #理论上这时候展示的图片和上一次的一致