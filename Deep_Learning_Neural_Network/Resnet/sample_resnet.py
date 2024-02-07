# mnistによるサンプル
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from resnet import Net

# initialize
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
model = Net(28*28, 10).to(device)
batch_size = 256
epochs = 30
# CrossEntropyLoss：交差エントロピー誤差関数
loss_fn = nn.CrossEntropyLoss()

# 最適化法の指定　optimizer：最適化
# SGD：確率的勾配降下法
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(28*28))
])
print("initialzation is done")
#訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download = True)
#検証データ
valid_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transform,
                                           download = True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
print("start learning")
for epoch in range(epochs):
    model.train()
    train_loss = []
    valid_loss = []
    for x_train, t_train in train_loader:
        x_train = x_train.to(device)
        t_train = t_train.to(device)

        pred = model(x_train)

        # 誤差を得る
        loss = loss_fn(pred, t_train)

        # 勾配の初期化
        optimizer.zero_grad()

        # 誤差逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()
        train_loss.append(loss.to("cpu").detach().numpy())

    model.eval()
    for x_val, t_val in valid_loader:
        x_val = x_val.to(device)
        t_val = t_val.to(device)

        loss_val = loss_fn(model(x_val), t_val)
        valid_loss.append(loss_val.to("cpu").detach().numpy())
    if epoch % 10 == 9 or epoch == 0:
        print('EPOCH: {}, Train Loss: {:.3f} ,Validation Loss: {:.3f}'.format(
            epoch+1,
            np.mean(train_loss),
            np.mean(valid_loss)
        ))



    