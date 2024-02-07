import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    """
    H(x) = F(x) + x
    """
    def __init__(self, in_dim):
        super().__init__()
        # この線形和は畳み込み演算でも可
        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, in_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, 256)
        self.resblock1 = ResnetBlock(256)
        self.linear2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        # resblock and skip connection
        x = self.resblock1(x) + x
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=1)

        return x
if __name__ == "__main__":
    # test
    in_dim = 28*28
    x = torch.randn(256, in_dim).unsqueeze(dim=0)
    model = Net(28*28, 10)
    print(model(x).size())