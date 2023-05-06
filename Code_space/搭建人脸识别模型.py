import glob
import torch
import torch.nn as nn
from torch.utils import data
import myFunction
from myFunction import myDataset_class as MC
import torchvision
from Model import FaceModel as FC

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
species_path = '/home/helei/PycharmProjects/My_DeepLearning/Data_space/CelebDataProcessed/*'
scale = 0.8
epoch = 35
step_number = 200
model_path = '/home/helei/PycharmProjects/My_DeepLearning/Model_space'
model_name = 'FC.pk'
img_size = 256
img_path = '/home/helei/PycharmProjects/My_DeepLearning/Data_space/CelebDataProcessed/*/*.jpg'
BATCH_SIZE = 20
save_path = '/home/helei/PycharmProjects/My_DeepLearning/Code_space/img/'
correct_path = '/home/helei/PycharmProjects/My_DeepLearning/Code_space/correct_img/'
# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(img_path)

species = myFunction.get_species(species_path)

all_labels = myFunction.All_Img_Label(all_imgs_path, species)

# 对数据进行转换处理
transform = myFunction.my_transform(img_size)
# 划分测试集和训练集
train_imgs, train_labels, test_imgs, test_labels, s = myFunction.Partition_Dataset(all_imgs_path, all_labels, scale)

train_ds = MC(train_imgs, train_labels, transform)
test_ds = MC(test_imgs, test_labels, transform)

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
model = FC()
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    myFunction.Train_Tets_Function(epoch, model, train_dl, loss, optimizer, model_path,
                                   model_name, test_dl, s, save_path, correct_path)
"""
epoch: 0 Accuracy:0.06
epoch: 1 Accuracy:0.15
epoch: 2 Accuracy:0.25
epoch: 3 Accuracy:0.38
epoch: 4 Accuracy:0.44
epoch: 5 Accuracy:0.53
epoch: 6 Accuracy:0.55
epoch: 7 Accuracy:0.59
epoch: 8 Accuracy:0.56
epoch: 9 Accuracy:0.67
epoch: 10 Accuracy:0.69
epoch: 11 Accuracy:0.69
epoch: 12 Accuracy:0.71
epoch: 13 Accuracy:0.71
epoch: 14 Accuracy:0.71
epoch: 15 Accuracy:0.71
epoch: 16 Accuracy:0.72
epoch: 17 Accuracy:0.72
epoch: 18 Accuracy:0.71
epoch: 19 Accuracy:0.72
epoch: 20 Accuracy:0.72
epoch: 21 Accuracy:0.72
epoch: 22 Accuracy:0.72
epoch: 23 Accuracy:0.72
epoch: 24 Accuracy:0.72
epoch: 25 Accuracy:0.71
epoch: 26 Accuracy:0.71
epoch: 27 Accuracy:0.72
epoch: 28 Accuracy:0.72
epoch: 29 Accuracy:0.72
epoch: 30 Accuracy:0.72
epoch: 31 Accuracy:0.72
epoch: 32 Accuracy:0.73
epoch: 33 Accuracy:0.72
epoch: 34 Accuracy:0.73
"""