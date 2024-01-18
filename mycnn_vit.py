import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
import torch.nn.functional as F

# 设置随机种子以保证实验的可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 视觉Transformer模块
class VisionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 解决维度问题
        x = rearrange(x, 'b n d -> n b d')  # 调整输入形状以适应Transformer
        x = self.transformer(x)
        x = rearrange(x, 'n b d -> b n d')
        x = x.mean(dim=1)  # 取平均以获得整个序列的表示
        x = self.fc(x)
        return x

# !!!???!!!   参数：滤波器数量
# 滤波器数量
num_filters_conv1 = 256
num_filters_conv2 = 512
# !!!???!!!   参数：滤波器大小
# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters_conv1, kernel_size=3, padding=1)
        # 归一化  BN LN GN
        # self.bn1 = nn.BatchNorm2d(num_filters_conv1)
        # self.ln1 = nn.LayerNorm((num_filters_conv1, 32, 32))
        # self.gn1 = nn.GroupNorm(num_groups=8, num_channels=num_filters_conv1)
        # 激活函数 
        self.relu = nn.ReLU()
        # self.gelu = F.gelu
        # self.tanh = nn.Tanh()
        # self.elu = nn.ELU()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters_conv1, num_filters_conv2, kernel_size=3, padding=1)

        # 归一化  BN LN GN
        # self.bn2 = nn.BatchNorm2d(num_filters_conv2)
        # self.ln2 = nn.LayerNorm((num_filters_conv2, 16, 16))
        # self.gn2 = nn.GroupNorm(num_groups=8, num_channels=num_filters_conv2)

        self.fc1 = nn.Linear(num_filters_conv2 * 8 * 8, 512)

    def forward(self, x):
        # 无归一化
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # 归一化  BN
        # x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # 归一化  LN
        # x = self.pool(self.relu(self.ln1(self.conv1(x))))
        # x = self.pool(self.relu(self.ln2(self.conv2(x))))

        # 归一化  GN
        # x = self.pool(self.relu(self.gn1(self.conv1(x))))
        # x = self.pool(self.relu(self.gn2(self.conv2(x))))

        # 激活函数 gelu
        # x = self.pool(self.gelu(self.conv1(x)))
        # x = self.pool(self.gelu(self.conv2(x)))
        # 激活函数 tanh
        # x = self.pool(self.tanh(self.conv1(x)))
        # x = self.pool(self.tanh(self.conv2(x)))
        # 激活函数 elu
        # x = self.pool(self.elu(self.conv1(x)))
        # x = self.pool(self.elu(self.conv2(x)))

        x = x.view(-1, num_filters_conv2 * 8 * 8)
        x = self.relu(self.fc1(x))
        return x
    
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# !!!???!!!  参数：批次大小
# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# !!!???!!!   参数：优化函数
# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNN().to(device)
transformer_model = VisionTransformer(input_dim=512, hidden_dim=256, num_heads=8, num_layers=4, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(cnn_model.parameters()) + list(transformer_model.parameters()), lr=0.001)
# optimizer = optim.Adadelta(list(cnn_model.parameters()) + list(transformer_model.parameters()), lr=1.0)
# optimizer = optim.SGD(list(cnn_model.parameters()) + list(transformer_model.parameters()), lr=0.001, momentum=0.9)

# 训练网络   
# !!!???!!!   参数：训练轮次
num_epochs = 10
for epoch in range(num_epochs):
    cnn_model.train()
    transformer_model.train()
    running_loss = 0.0

    correct_predictions = 0
    total_samples = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用卷积神经网络提取特征
        cnn_outputs = cnn_model(inputs)
        # 使用视觉Transformer处理提取的特征
        transformer_outputs = transformer_model(cnn_outputs)
        # 计算损失
        loss = criterion(transformer_outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 统计准确预测的样本数
        _, predicted = torch.max(transformer_outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # 计算并输出每轮的训练结果
    epoch_loss = running_loss / len(trainloader)
    accuracy = correct_predictions / total_samples * 100
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {accuracy:.2f}%')

print('Finished Training')

# 测试网络
cnn_model.eval()
transformer_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # 使用卷积神经网络提取特征
        cnn_outputs = cnn_model(images)
        # 使用视觉Transformer处理提取的特征
        transformer_outputs = transformer_model(cnn_outputs)

        _, predicted = torch.max(transformer_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test images: %d %%' % (100 * correct / total))