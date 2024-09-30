import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
# 批量大小
batch_size =64
# 学习率
learning_rate = 1e-3
# 训练轮数
num_epochs = 20
# 对图片进行的改变
transform = transforms.Compose([
    transforms.ToTensor()
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
        self.max_pool2d_1 = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool2d_2 = nn.MaxPool2d(2)
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max_pool2d_3 = nn.MaxPool2d(2)
        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(256*4*4, 1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 10)


    def forward(self, x):
        output1 = self.max_pool2d_1(self.conv2d_1(x))
        output2 = self.max_pool2d_2(self.conv2d_2(output1))
        output3 = self.max_pool2d_3(self.conv2d_3(output2))
        output4 = self.conv2d_4(output3)

        # 展平
        output4 = output4.view(output4.size(0), -1)
        output5 = self.relu1(self.linear1(output4))
        output6 = self.relu2(self.linear2(output5))
        output7 = self.relu3(self.linear3(output6))
        output8 = self.linear5(self.linear4(output7))
        return output8

# TODO:补全
model = Network(3).to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    # 开启训练模式
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # 假如GPU能用，用GPU
            inputs, labels = inputs.to(device), labels.to(device)
            #  清空梯度
            optimizer.zero_grad()
            # 输出结果
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播+优化
            loss.backward()
            optimizer.step()
            # 计算总的loss
            running_loss += loss.item()
            # 计算预测
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算准确率
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

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
