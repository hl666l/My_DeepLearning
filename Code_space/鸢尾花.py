import torch.nn as nn
import torch
import torch.utils.data as Data
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def getdata():
    train_data = load_iris()  # 加载鸢尾花数据集 特征是四个
    data = train_data['data']  # 取出特征
    labels = train_data['target'].reshape(-1, 1)  # 取出标签
    total_data = np.hstack((data, labels))  # 连接成完整的数据集
    np.random.shuffle(total_data)  # 随机打乱
    train = total_data[0:80, :-1]  # 划分数据集核测试集
    test = total_data[80:, :-1]
    train_label = total_data[0:80, -1].reshape(-1, 1)  # 划分，训练标签，测试标签
    test_label = total_data[80:, -1].reshape(-1, 1)
    return data, labels, train, test, train_label, test_label


# 网络类
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.fc = nn.Sequential(  # 添加神经元以及激活函数
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 3)
        )
        self.mse = nn.CrossEntropyLoss()  # 损失函数
        self.optim = torch.optim.Adam(params=self.parameters(), lr=0.1)  # 优化函数

    def forward(self, inputs):  # 正向传播函数
        outputs = self.fc(inputs)
        return outputs

    def train(self, x, label):
        out = self.forward(x)  # 正向传播
        loss = self.mse(out, label)  # 根据正向传播计算损失
        self.optim.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        self.optim.step()  # 应用梯度更新参数

    def test(self, test_):  # 测试函数
        return self.fc(test_)


if __name__ == '__main__':
    data, labels, train, test, train_label, test_label = getdata()  # 依次获取数据
    mynet = mynet()  # 实例化模型
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(), torch.from_numpy(train_label).long())  # 生成Tensor数据集
    BATCH_SIZE = 10
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 声称加载数据集
    for epoch in range(100):  # 训练轮
        for step, (x, y) in enumerate(train_loader):  # 迭代器
            y = torch.reshape(y, [BATCH_SIZE])
            mynet.train(x, y)  # 训练
            if epoch % 20 == 0:  # 每20轮输出一次一轮的训练内容
                print('Epoch: ', epoch, '| Step: ', step, '| batch y: ', y.numpy())
    out = mynet.test(torch.from_numpy(data).float())  # 测试
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值  torch.max(tensor, dim),dim=1是返回行的[maxdata,index]。
    pred_y = prediction.data.numpy()  # 转化数据类型便于后面计算
    test_y = labels.reshape(1, -1)
    target_y = torch.from_numpy(test_y).long().data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    print("莺尾花预测准确率", accuracy)
