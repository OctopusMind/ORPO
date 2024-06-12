import torch
import torch.nn as nn

# 创建一个损失函数实例
criterion = nn.CrossEntropyLoss()

# 模拟logits（模型未归一化的输出）和目标标签
logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
labels = torch.tensor([2, 1])

# 计算损失
loss = criterion(logits, labels)
print(loss)

# 反向传播
loss.backward()
