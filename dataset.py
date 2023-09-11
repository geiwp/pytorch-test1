from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# transforms模块中的Compose( )可以把多个变换组合在一起，需要写成列表的形式
# 将PIL图片或者numpy.ndarray(HxWxC) (范围在0-255) 转成torch.FloatTensor (CxHxW) (范围为0.0-1.0)
# 归一化，RGB三个通道上的均值(0.5，0.5，0.5)，三个通道的标准差(0.5, 0.5, 0.5)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #平均值和标准差

# 下载数据集，并将其保存在当前文件夹下的CIFAR文件夹中，并对其进行transform变换
trainset = datasets.CIFAR10(root='./CIFAR', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./CIFAR', train=False, download=True, transform=transform)

# 打印出数据和标签的维度, 标签是list，需要将其转换成numpy数组才能查看其维度
# print("train:", trainset.data.shape)
# print("train_label:", np.array(trainset.targets).shape)
# print("test:", testset.data.shape)
# print("test_label:", np.array(testset.targets).shape)

# 数据加载，
trainloader = DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2) #shuffle是否把数据打乱
testloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=2) #num_workers用多少个子进程加载数据

# 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')




