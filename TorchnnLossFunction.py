"""
反向传播：
根据LossFunction, 可以计算出实际输出与预期target的差距, 再根据这个差距提供更新网络的依据、
"""
import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)


loss_l1 = L1Loss(reduction='sum')   # 函数的参数详情，请见官方文档的解释[如果为默认的求mean，需要将input和target的参数类型设置为浮点]
result_l1 = loss_l1(input, targets)

loss_mse = MSELoss()
result_mse = loss_mse(input, targets)

"""
对于交叉熵损失函数:
loss(x,class) = -x[class] + log(sum(exp(x[i])))
其中: x代表卷积层最后对每个分类类别的概率预测向量, class代表target对应的类别在预测向量中的idx.
"""
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))    # 将输入的tensor格式reshape为适合CrossEntropyLoss函数的格式
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)

# print(result_cross)
# print(result_l1)
# print(result_mse)