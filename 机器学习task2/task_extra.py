import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 画图
# 创建图表
fig, ax = plt.subplots()
# 初始化一条线
train_line, = ax.plot([], [], 'b-', label='Train Loss')
ax.set_title('loss')
ax.set_xlabel('Batch')
ax.set_ylabel('loss')
ax.legend()
# 打开交互模式
plt.ion()

# TODO:解释参数含义，在?处填入合适的参数
# 批量大小
batch_size =64
# 学习率
learning_rate = 1e-3
# 训练轮数
num_epochs = 10
# 对图片进行的改变
transform = transforms.Compose([
    # 水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机裁剪
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self,input_features):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.max_pool2d = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(128*16*16, 1024)
        self.relu1 = nn.ReLU()
        # 增加dropout，防止过拟合
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 10)


    def forward(self, x):
        output1 = self.max_pool2d(self.conv2d_1(x))
        output2 = self.conv2d_2(output1)
        output3 = self.conv2d_3(output2)
        # 展平
        output3 = output3.view(output3.size(0), -1)
        output4 = self.dropout1(self.relu1(self.linear1(output3)))
        output5 = self.dropout2(self.relu2(self.linear2(output4)))
        output6 = self.relu3(self.linear3(output5))
        output7 = self.linear5(self.linear4(output6))
        return output7

# TODO:补全

model = Network(3)
# 如果有训练过的模型
if os.path.isfile("best.pth"):
    model = model.load_state_dict(torch.load("best.pth"))
# 除了自己的模型还可以使用Resnet
# model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)



def train():
    # 开启训练模式
    losses = []
    corrects = []
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        # 可视化训练过程
        for data in tqdm(trainloader):
            inputs, labels = data
            # 假如GPU能用，用GPU
            inputs, labels = inputs.to(device), labels.to(device)
            #  清空梯度
            optimizer.zero_grad()
            # 输出结果
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            # 反向传播+优化
            loss.backward()
            optimizer.step()
            # 计算总的loss
            running_loss += loss.item()
            # 计算预测
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            corrects.append(correct)
            # 显示图形
            train_line.set_data(range(len(losses)), losses)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
        # 计算准确率
        accuracy = 100 * correct / total
        # 保存最佳模型
        if running_loss / len(trainloader) < best_loss:
            best_loss = running_loss / len(trainloader)
            torch.save(model.state_dict(), 'best.pth')
            print('Best model saved')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
    plt.ioff()
    plt.show()


def test():
    # 开启测试模式，关掉一些层
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()
