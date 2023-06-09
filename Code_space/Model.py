import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=8,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # 8x26x26
        )
        self.linear1 = torch.nn.Linear(5408, 1352)
        self.linear2 = torch.nn.Linear(1352, 338)
        self.linear3 = torch.nn.Linear(338, 120)
        self.linear4 = torch.nn.Linear(120, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x


class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=8,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # 8x26x26
        )
        self.linear1 = torch.nn.Linear(5408, 1352)
        self.linear2 = torch.nn.Linear(1352, 150)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# 人脸识别模型
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
