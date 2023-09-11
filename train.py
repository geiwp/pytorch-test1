import torch
import gradio as gr
from torch.autograd import Variable
import matplotlib.pyplot as plt
import altair as alt
# 导入
from dataset import trainloader,testloader,trainset,testset
from net import MyNet
from noResNet import ResNet,ResidualBlock
# from dataset1 import traindata,trainlabels,testdata,testlabels
# 定义超参数
learning_rate = 0.01
momentum = 0.9
num_epoches = 20
batch_size = 128

# model = ResNet(ResidualBlock)
model = MyNet()
#判断是否有GPU, 如果使用GPU就将模型放到GPU中进行训练
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()


# 交叉熵损失函数刻画了实际输出概率与期望输出概率之间的相似度，也就是交叉熵的值越小，两个概率分布就越接近  它可以有效避免梯度消散
criterion = torch.nn.CrossEntropyLoss()     
# SGD梯度下降方法  带动量的随机梯度下降  momentum = 0.9时效果最好
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
trloss = []
teloss = []
tracc = []
teacc = []
# traindatas = [traindata,trainlabels]
if __name__ == '__main__':
    
    for epoch in range(num_epoches):
        print("*"*10)
        print("epoch{}".format(epoch+1))
        train_loss = 0.0  
        train_acc =0.0
        test_loss = 0.0
        test_acc = 0.0
        for data in trainloader: # enumerate是python的内置函数，既获得索引也获得数据
        #for image, label in trainloader:
            image,label = data
             # data包含数据和标签信息
            # 将数据转换成Variable,Variable就是 变量 的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
            if use_gpu:
                image = image.cuda()
                label = label.cuda()
            else:
                image = Variable(image)
                label = Variable(label)

            # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()

            output = model(image) #输入到模型中

            loss = criterion(output, label) #计算损失，实际输出 label实际标签
            # train_loss += loss.item()*label.size(0) # 此处是什么意思？
            # 得到预测值最大的值和索引

            train_loss += loss.item() #也可以
            _, pred = torch.max(output,1) # 返回每行的最大值的索引值，也就是预测出的可能性最大的类别的相应的标签
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()

            loss.backward() # 反向传播
            optimizer.step() # 梯度更新,canshu
        
    # 每训练完一个周期打印一次平均损失和平均精度
        

    # 测试
        with torch.no_grad():
            for data in testloader: # enumerate是python的内置函数，既获得索引也获得数据
    
                image, label = data # data包含数据和标签信息
                # 将数据转换成Variable,Variable就是 变量 的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
                if use_gpu:
                    image = Variable(image).cuda()
                    label = Variable(label).cuda()
                else:
                    image = Variable(image)
                    label = Variable(label) 
                output = model(image) #输入到模型中

                loss = criterion(output, label) #计算损失，实际输出 label实际标签
                

                test_loss += loss.item() #也可以
                _, pred = torch.max(output,1) # 返回每行的最大值的索引值，也就是预测出的可能性最大的类别的相应的标签
                test_correct = (pred == label).sum()
                test_acc += test_correct.item()
        print('Train {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, train_loss / (len(trainset)), train_acc / (len(trainset))
        ))
        print("Test Loss: {:.6f}, Acc: {:.6f}".format(test_loss/len(testset), test_acc/len(testset)))
        trloss.append(train_loss / len(trainset))
        tracc.append(train_acc / len(trainset))
        # teloss.append(test_loss/len(testset))
        # teacc.append(test_acc/len(testset))
    
    plt.plot(num_epoches, tracc, c='red', label="acc")
    plt.plot(num_epoches, trloss,c='green', linestyle='--', label="loss")
    plt.show()

    # model.eval()
    # eval_loss = 0.0
    # eval_acc = 0.0
    # for image, label in testloader:
    #     if use_gpu:
    #         image = Variable(image).cuda()
    #         label = Variable(label).cuda()
    #     else:
    #         image = Variable(image)
    #         label = Variable(label)

    #     output = model(image)
    #     loss = criterion(output, label)
    #     eval_loss += loss.item()*label.size(0) # 此处是什么意思？
    #     _, predicted = torch.max(output, 1)
    #     eval_correct = (predicted == label).sum()
    #     eval_acc += eval_correct.item()
    # print("Test Loss: {:.6f}, Acc: {:.6f}".format(
    #     eval_loss/len(testset), eval_acc/len(testset)
    # ))
