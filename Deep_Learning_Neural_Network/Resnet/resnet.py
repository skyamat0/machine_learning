import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlockLinear(nn.Module):
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

class BuildingBlock(nn.Module):
    """
    H(x) = BuildingBlock(x) + x
    """
    def __init__(self, in_channels, med_channels, out_channels, is_downsample=False):
        super().__init__()
        if is_downsample == True:
            stride = 2
        else:
            stride = 1
        self.m_1 = nn.Conv2d(in_channels, med_channels, kernel_size=3, stride=stride, padding=1)
        self.m_2 = nn.Conv2d(med_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.m_1(x)
        out = F.relu(out)
        out = self.m_2(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        self.num_classes = num_classes
        super().__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)

        # conv2_x(maxpool)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # conv2_x
        self.resblock2_1 = BuildingBlock(in_channels=64, med_channels=64, out_channels=64)
        self.resblock2_2 = BuildingBlock(in_channels=64, med_channels=64, out_channels=64)

        # conv3_x（一つ目はダウンサンプリングのため，ストライドを2とする）
        self.resblock3_1 = BuildingBlock(in_channels=64, med_channels=128, out_channels=128, is_downsample=True)
        self.resblock3_2 = BuildingBlock(in_channels=128, med_channels=128, out_channels=128)

        # conv4_x（一つ目はダウンサンプリングのため，ストライドを2とする）
        self.resblock4_1 = BuildingBlock(in_channels=128, med_channels=256, out_channels=256, is_downsample=True)
        self.resblock4_2 = BuildingBlock(in_channels=256, med_channels=256, out_channels=256)

        # conv5_x（一つ目はダウンサンプリングのため，ストライドを2とする）
        self.resblock5_1 = BuildingBlock(in_channels=256, med_channels=512, out_channels=512, is_downsample=True)
        self.resblock5_2 = BuildingBlock(in_channels=512, med_channels=512, out_channels=512)

        # 出力のサイズを指定してAvgPooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fully connectionによって変換
        self.fc = nn.Linear(512, self.num_classes)
    def conv11(self, in_channels, out_channels):
        """
        if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.
        だそうなので，画像のサイズが1/2になり，フィルタの数が２倍になっていく．
        そのため，画像のサイズを調整し，チャンネル数も調整する必要がある．
        画像のサイズはstrideを２にすることで，1/2に，チャンネル数は1*1のカーネルをin_channelsの２倍用いて，畳み込みをすればよい．
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
    
    def forward(self, x):
        # conv
        out = self.conv1(x)
        out = F.relu(out)
        # maxpool
        out = self.maxpool(out)
        out = F.relu(out)

        # conv2_x
        out = self.resblock2_1(out) + out
        out = F.relu(out)
        out = self.resblock2_2(out) + out
        out = F.relu(out)

        # conv3_x
        out = self.resblock3_1(out) + self.conv11(64, 128)(out)
        out = F.relu(out)
        out = self.resblock3_2(out) + out
        out = F.relu(out)

        # conv4_x
        out = self.resblock4_1(out) + self.conv11(128, 256)(out)
        out = F.relu(out)
        out = self.resblock4_2(out) + out
        out = F.relu(out)

        # conv5_x
        out = self.resblock5_1(out) + self.conv11(256, 512)(out)
        out = F.relu(out)
        out = self.resblock5_2(out) + out
        out = F.relu(out)

        # avgpool
        out = self.avgpool(out)
        out = F.relu(out)

        # fully connection
        out = self.fc(out.flatten())

        return out
if __name__ == "__main__":
    # test
    x = torch.randn(3, 224, 224)
    model = ResNet18(in_channels=3, num_classes=1000)
    model(x)
    
    # in_dim = 224
    # in_channels = 3
    # x = torch.randn(in_channels, in_dim, in_dim)
    # model = ResNet18(in_dim=in_dim, out_dim=1000, in_channels=3)
    # print(model(x).size())