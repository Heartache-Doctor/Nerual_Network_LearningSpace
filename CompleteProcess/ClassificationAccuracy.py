"""
假设：二分类model
https://www.bilibili.com/video/BV1hE411t7RN?p=28&spm_id_from=pageDriver&vd_source=da8e0f1d30e350fad5266bf5afc123d3
18分钟
"""
import torch

test_data_size = 2
outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])
targets = torch.tensor([0, 1])

preds = outputs.argmax(1)   # argmax函数可以将向量中最大值对应的idx返回，参数1/0是用来判断读取向量方向的。[横向读取，竖向读取]

accuracy = ((preds == targets).sum()) / test_data_size
print(accuracy)