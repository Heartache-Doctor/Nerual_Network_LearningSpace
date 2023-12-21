import torchvision

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

from torch.utils.data import DataLoader # 加载已处理好的数据集

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# batch_size:一次训练使用数据集中的几个数据；shuffle:每个取数据阶段是否随机；num_workers:并发数；drop_last:data数量除不尽batch_size时是否去尾
# 返回的是以batch_size打包后的的img, target集合

for data in test_loader:
    img, target = data
    print(img)
    print(target)

# img此时就可以作为神经网络的输入了

