# cnn_vit
融合传统的卷积神经网络与最新的视觉Transformer模块，发挥两者的优势，构建更为强大的图像分类模型
# 实验环境 #
- 操作系统：Linux
- cuda版本：11.8
- pytorch版本：2.0.0
- 需要安装的包：pytorch，torchvision，einops

安装方式：
> pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
# 数据集下载 #
CIFAR-10（Canadian Institute For Advanced Research - 10）是一个常用的图像分类数据集，由加拿大高级研究院创建

- 下载链接：
[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html "The CIFAR-10 dataset")
- 图像类别：飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车
- 图像尺寸：32x32像素
- 图像数量：60000张（每个类别6000张）
- 选择理由：CIFAR-10是计算机视觉领域中最常用的基准数据集之一，包含多个类别的图像，涵盖了日常生活中常见的物体和动物，适用于图像分类任务，同时由于图像尺寸相对较小，训练和评估模型的速度相对较快，有助于在较短时间内进行快速迭代和实验

# 运行方式 #
1. 替换数据集保存路径
> trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)<br/>
> testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

将代码中root的地址换成自己设定的地址，注意这里的data是数据集文件夹的父级目录

2. 运行

> python mycnn_vit.py > result.xlsx

运行该命令需要切换到代码所在目录，运行结果会保存到result.xlsx文件中

# 实验结果 #

Epoch 1/10, Loss: 1.783, Accuracy: 30.64%  
Epoch 2/10, Loss: 1.266, Accuracy: 53.91%  
Epoch 3/10, Loss: 0.943, Accuracy: 67.39%  
Epoch 4/10, Loss: 0.714, Accuracy: 75.88%  
Epoch 5/10, Loss: 0.515, Accuracy: 82.78%  
Epoch 6/10, Loss: 0.339, Accuracy: 88.74%  
Epoch 7/10, Loss: 0.220, Accuracy: 92.96%  
Epoch 8/10, Loss: 0.153, Accuracy: 95.18%  
Epoch 9/10, Loss: 0.117, Accuracy: 96.34%  
Epoch 10/10, Loss: 0.094, Accuracy: 97.05%  
Finished Training  
Accuracy on the test images: 73 %  
