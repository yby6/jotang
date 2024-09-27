import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

# 读取数据
train_data  = pd.read_csv('./data/Boston Housing/train.csv')
test_data = pd.read_csv('./data/Boston Housing/test.csv')
sample_submission = pd.read_csv('./data/Boston Housing/submission_example.csv')

# 查看分析数据(没有缺失值)
print(train_data.info())
print(test_data.info())


# 转换为numpy
train_df = train_data.values
test_df = test_data.values

# 训练集
x = train_df[:, :-1]
y = train_df[:, -1]

# 转换为torch向量
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
test_df = torch.tensor(test_df, dtype=torch.float32)

# 划分数据集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=5)

in_features = x.shape[1]

class my_model(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features,32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32,64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64,32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32,1)

    def forward(self,x):
        output1 = self.relu1(self.fc1(x))
        output2 = self.relu2(self.fc2(output1))
        output3 = self.relu3(self.fc3(output2))
        output4 = self.fc4(output3)
        return output4

model = my_model(in_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1000


model.train()
# 训练过程
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)

    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    print(f"loss: {loss}")


model.eval()
y_val_predicted = model(x_val)
y_val_loss = criterion(y_val_predicted, y_val).item()
print(f"y_val_loss: {y_val_loss}")


test_predicted = model(test_df).flatten()
res = pd.DataFrame(
    {
        'ID': sample_submission['ID'],
        'medv':test_predicted.detach().numpy()
    }
)
res.to_csv('submission.csv', index=False)
print('successful')



