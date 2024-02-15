import torch
import torch.nn as nn

m_3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
m_1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)

class BottleNeck:
    def __init__(self, in_channels, out_channels) -> None:
        self.in_c = in_channels
        self.out_c = out_channels

    def __call__(self, x):
        out = nn.Conv2d(in_channels=self.in_c, out_channels=64, kernel_size=1, stride=1, padding=0)(x)
        print("==1 by 1 kernel==")
        print(out.size())
        out = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)(out)
        print("==3 by 3 kernel==")
        print(out.size())
        out = nn.Conv2d(in_channels=64, out_channels=self.out_c, kernel_size=1, stride=1, padding=0)(out)
        print("==1 by 1 kernel==")
        print(out.size())
        return x
    
x_in = torch.randn(256, 28, 28)
print("==input dim==")
print(x_in.size())
print("== 3 by 3 kernel ==")
print(m_3(x_in).size())
print("== 1 by 1 kernel ==")
print(m_1(x_in).size())
print("==bottle neck==")
bn = BottleNeck(in_channels=256, out_channels=256)
bn(x_in)
