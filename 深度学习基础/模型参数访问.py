from Model import Model
import torch.nn as nn

# %%
model = Model()
for name, param in model.named_parameters():
    """
    通过输出可学习参数名称，可以发现模型中的池化操作参数没有被输出， 故池化不参与反向传播
    conv1.0.weight
    conv1.0.bias
    conv1.3.weight
    conv1.3.bias
    """
    print(name)
    """
    权重（核值）：torch.Size([6, 3, 5, 5])
    偏置：torch.Size([6])
    两者是成对出现的。

    可以看到第一层的参数shape=[6, 3, 5, 5]因为我们输入的是3通道的图片故卷积核也应该是3通道。
    我们有6个[3,5,5]这样的核，每次拿出一个[3,5,5]的核参与计算得到[3,n,n]的feature map。然后将这个3通道的feature map
    每个通道的图片对应位置相加融合成一张1通道的图片[n,n]。
    而我们有6个[3,5,5]的核。最后得到6张[n,n]的feature map，将这6张feature map叠加获得[6,n,n]的feature map。

    因为输出是[6,n,n]的feature map，故有6个偏置
    """
    print(param.shape)
    print(param.data)
    #  储存参数
    a = param.storage()
print('shuchu:', a)  # 输出参数

name2 = []
param = []
name2, param = model.named_parameters()
