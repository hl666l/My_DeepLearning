import glob
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def get_species(path):
    all_species_path = glob.glob(path)
    species = []
    for name in all_species_path:
        s = name.split('/')[-1]
        species.append(s)
    return species


def All_Img_Label(all_imgs_path, species):
    """
    制作所有图片的标签
    :param all_imgs_path: 所有图片的地址
    :param species: 图片的种类名称（就是每个类表所在文件夹的名称）
    :return: 返回所有图片的标签
    """
    all_Img_label = []
    for img in all_imgs_path:
        for i, c in enumerate(species):
            if c in img:
                all_Img_label.append(i)
    return all_Img_label


def my_transform(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
        ]
    )
    return transform


class myDataset_class(data.Dataset):
    """
    data.Dataset 是抽象类，子类继承后里面有三个函数需要我们具体实现

    为什么要写这样一个类？
    答：后面我们要将我们的数据交给data.DataLoader()，
    他能帮我们把数据按照一个batchsize的大小分好组，
    通过训练时的迭代器扔进训练模型中。
    而这个函数需要的是一个data.Dataset类型的数据
    """

    def __init__(self, img_path, label, transform):
        """
        :param img_path: 所有图片的路径list
        :param label: 所有图片的标签list
        :param transform: 对图片转换的操作
        """
        self.data = img_path
        self.label = label
        self.transform = transform

    def __getitem__(self, index):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        """
        :param index: 默认
        :return: 返回制作好的数据，标签
        """
        img = self.data[index]
        label = self.label[index]
        pill_img = Image.open(img)
        data = self.transform(pill_img)
        return data, label

    def __len__(self):
        """
        :return: 返回数据的长度
        """
        return len(self.data)


class FaceModel(torch.nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=64,
                            kernel_size=7,
                            padding=3,
                            stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=6,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=64,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=32,
                            kernel_size=4,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

        )
        self.linear1 = torch.nn.Linear(8192, 2048)
        self.linear2 = torch.nn.Linear(2048, 512)
        self.linear3 = torch.nn.Linear(512, 150)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


def Partition_Dataset(all_imgs_path, all_imgs_label, scale):
    """

    :param all_imgs_path: 所有图片的路径 list
    :param all_imgs_label: list
    :param scale: 训练集和测试集的比例
    :return: 返回一个元组
    """
    index = np.random.permutation(len(all_imgs_path))
    all_img_path = np.array(all_imgs_path)[index]
    all_img_label = np.array(all_imgs_label)[index]
    s = int(len(all_img_path) * scale)
    st = len(all_img_path) - s

    train_img = all_img_path[:s]
    train_label = all_img_label[:s]

    test_img = all_img_path[s:]
    test_label = all_img_label[s:]

    return train_img, train_label, test_img, test_label, st


def View_correct(X_axis, Y_axis, correct_save_path, X_label='Epoch', Y_label='Accuracy'):
    plt.cla()
    plt.plot(X_axis, Y_axis, 'r-', lw=1)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.savefig(correct_save_path + str(X_axis[-1]) + '正确率' + '.png')
    plt.pause(0.2)


def View_loss(X_axis, Y_axis, X_label, Y_label, title, save_path):
    """

    :param X_axis: x轴
    :param Y_axis: y轴
    :param X_label: x 标签
    :param Y_label: y 标签
    :param title: 图标题
    :return: 无
    """
    """
    
    plt.cla()  # 清除轴， 即当前图形中当前活动的轴，使其它轴保持不变
    plt.plot(X_axis, Y_axis, 'r-', lw=2)  # 画表函数 'r-' r是红色，‘-’是线条的形状 lw是线条的宽度
    plt.ylabel('loss')  # x轴表示的含义
    plt.xlabel('rate of progress ')  # y轴的含义
    plt.title('epoch=%d step=%d loss=%.4f ' % (epoch, step, loss.cpu()))  # 图表的含义（标题）
    plt.pause(0.1)  # 用于暂停0.1秒
    
    """
    plt.cla()
    plt.plot(X_axis, Y_axis, 'r-', lw=1)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)
    plt.savefig(save_path + title + '.png')
    plt.pause(0.2)


def test_accuracy_img(epoch_list, test_data, number_data, model, lossFunction, correct_axis, save_path):
    model = model
    sum_loss = 0.0
    sum_correct = 0.0
    for step, (data, label) in enumerate(test_data):
        data = Data_To_Cuda(data)
        label = Data_To_Cuda(label)
        y = model(data)
        loss = lossFunction(y, label)
        _, pred = torch.max(y.data, dim=1)
        correct = pred.eq(label.data).sum()
        sum_loss += loss.item()
        sum_correct += correct.item()
    test_correct = sum_correct * 1.0 / number_data
    correct_axis.append(test_correct)
    print('epoch:', epoch_list[-1], 'Accuracy:%.2f' % test_correct)
    View_correct(epoch_list, correct_axis, save_path)


def Data_To_Cuda(data):
    """

    :param data: 要传入cuda中的数据
    :return: 返回传入后的数据
    """
    my_data = data
    if torch.cuda.is_available():
        my_data = my_data.cuda()
    else:
        print('No device use')
    return my_data


def Train_Tets_Function(Epoch, model, Train_Data, lossFunction, optimizer, model_path, model_save_name, Test_data,
                        number_test_data, save_path, correct_save_path):
    """
    训练函数
    :param save_path:
    :param model_path: 模型保存地址
    :param Epoch: 训练轮数
    :param model: 模型
    :param Train_Data:训练数据
    :param lossFunction:损失函数
    :param optimizer:优化器
    :param Step_number:一轮中多少步输出一次
    :param model_save_name: 模型保存时的名称
    :return: 无返回
    """
    X_axis = [0]
    Y_axis = [0]
    X_label = 'rate of progress'
    Y_label = 'loss'
    epoch_list = []
    correct_list = []
    for epoch in range(Epoch):
        epoch_list.append(epoch)
        for step, (train_data, train_labels) in enumerate(Train_Data):
            # 数据扔到cuda中
            train_data = Data_To_Cuda(train_data)
            train_labels = Data_To_Cuda(train_labels)

            y = model(train_data)
            loss = lossFunction(y, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Y_axis.append(loss.cpu().tolist())
        X_axis.append(X_axis[len(X_axis) - 1] + 1)
        title = ('epoch=%d  loss=%.4f ' % (epoch, loss.cpu()))
        View_loss(X_axis, Y_axis, X_label, Y_label, title, save_path)
        test_accuracy_img(epoch_list, Test_data, number_test_data, model, lossFunction, correct_list, correct_save_path)
    torch.save(model.state_dict(), model_path + '/' + model_save_name)


"""
scale: 训练集和测试集比例
epoch： 训练多少轮
step_number： 训练步数
model_path: 模型保存的地址
model_name: 模型保存时的名称
img_size: 训练时需要的图片尺寸
img_path:匹配路径
BATCH_SIZE：每个batch的大小
"""
species_path = '/kaggle/input/pubfig-dataset-256x256-jpg/CelebDataProcessed/*'
img_path = '/kaggle/input/pubfig-dataset-256x256-jpg/CelebDataProcessed/*/*.jpg'
scale = 0.8
epoch = 35
step_number = 200
model_name = 'FC.pk'
img_size = 256
BATCH_SIZE = 40
model_path = '/kaggle/working/Model_space'
save_path = '/kaggle/working/img/'
correct_path = '/kaggle/working/correct_img/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(correct_path):
    os.mkdir(correct_path)
# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(img_path)

species = get_species(species_path)

all_labels = All_Img_Label(all_imgs_path, species)

# 对数据进行转换处理
transform = my_transform(img_size)
# 划分测试集和训练集
train_imgs, train_labels, test_imgs, test_labels, s = Partition_Dataset(all_imgs_path, all_labels, scale)

train_ds = myDataset_class(train_imgs, train_labels, transform)
test_ds = myDataset_class(test_imgs, test_labels, transform)

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
model = FaceModel()
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    Train_Tets_Function(epoch, model, train_dl, loss, optimizer, model_path,
                        model_name, test_dl, s, save_path, correct_path)
