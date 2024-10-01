import torch
import torchvision
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)

test_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8)


pretrained_cfg = timm.models.create_model('vit_base_patch32_224').default_cfg
pretrained_cfg['file'] = r"B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224 (2).npz"

model = timm.models.create_model('vit_base_patch32_224', pretrained=True, pretrained_cfg=pretrained_cfg)

model.head = nn.Linear(model.head.in_features, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)


epochs = 5
def train():
    model.train()
    print('>>>>>>>>>>>>>> Training Time <<<<<<<<<<<<<<')

    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        print(f'[Epoch{epoch}/{epochs}]')

        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        acc = 100*correct / total
        print(f"Training loss:{epoch_loss:.4f}, Training accuracy:{acc:.2f}%")

def test():
    model.eval()
    print('>>>>>>>>>>>>>> Testing Time <<<<<<<<<<<<<<')

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        acc = 100 * test_correct / test_total
        print(f"Test ing accuracy:{acc:.2f}%")

if __name__ == "__main__":
    train()
    test()
