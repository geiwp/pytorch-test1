import torch
import torch.nn as nn


class MyNet(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类nn.Module是所有神经网络的基类
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # 卷积层,默认padding=0,stride=1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 最大池化层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)  # 卷积层
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #stride 默认值为kernel_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)   #激活处理，把一个预测值转换为0-1的概率
        
    # 主要会用到的就是torch.nn 和torch.nn.funtional这
    def forward(self, x):  # 前向传播，反向传播涉及到torch.autograd模块 
        x = self.maxpool1(self.conv1(x))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# x = torch.randn(1,3,32,32)
# myNet = MyNet()
# out = myNet(x)
# # print(out)
# Softmax 是用于多类分类问题的激活函数，在多类分类问题中，超过两个类标签则需要类成员关系。
# 对于长度为 K 的任意实向量，Softmax 可以将其压缩为长度为 K，值在（0，1）范围内，并且向量中元素的总和为 1 的实向量。