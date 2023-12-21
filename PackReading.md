# 遇到新包怎么处理呢

## 1 按住control，左键不懂的参数，看底层代码

## 2 看官方文档

## 3 注意输入输出的数据类型[如果输出的数据类型官方未说明，可以把输出的变量print一下，看是什么类型的变量]

## 4 关注函数中需要自定义的参数名用途，在Args里面查找对应的解释；关注输入和输出对应的变量需要的shape格式

## 5 package中的函数调用方法一般都是先用一个变量将抽象函数实例化，同时为函数添加参数值；然后再调用实例化后的funtion变量，再加入输入的数据，将输出的return值赋予给result变量
```python
import torch
from torch.nn import L1Loss

loss_l1 = L1Loss(reduction='sum')   # 将函数添加参数后实例化
result_l1 = loss_l1(input, targets) # 将实例化的函数加入输入参数，将输出结果传递给result变量
```



