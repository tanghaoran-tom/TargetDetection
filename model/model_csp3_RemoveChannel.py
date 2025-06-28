import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile  # 用于计算FLOPs和参数量

'''
去除“渐进式通道增加”后，每层通道变得均匀，导致浅层计算量暴涨（因为特征图大）；

但 参数量变少，是因为浅层的卷积核数量（即通道数）固定不再增长。
'''


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


class CBL(nn.Module):
    def __init__(self, c_in, c_out, size, stride, padding):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out


class CSP(nn.Module):
    def __init__(self, c_in):
        super(CSP, self).__init__()
        self.cbl = CBL(c_in, c_in, 3, 1, 1)
        self.res = ResBlock(c_in)
        self.conv = nn.Conv2d(c_in, int(c_in / 2), 3, 1, 1)
        self.batchnorm = nn.BatchNorm2d(c_in)

    def forward(self, x):
        b1 = self.cbl(x)
        b1 = self.res(b1)
        b1 = self.conv(b1)
        b2 = self.conv(x)
        b = torch.cat((b1, b2), dim=1)
        b = self.batchnorm(b)
        b = F.leaky_relu(b)
        b = self.cbl(b)
        return b


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        self.cbl1 = CBL(3, 32, 2, 2, 0)      # 640 -> 320
        self.csp1 = CSP(c_in=32)
        self.cbl2 = CBL(32, 32, 2, 2, 0)     # 320 -> 160
        self.cbl3 = CBL(32, 32, 2, 2, 0)     # 160 -> 80
        self.csp2 = CSP(c_in=32)
        self.cbl4 = CBL(32, 32, 2, 2, 0)     # 80 -> 40
        self.cbl5 = CBL(32, 32, 2, 2, 0)     # 40 -> 20
        self.csp3 = CSP(c_in=32)
        self.cbl6 = CBL(32, 32, 2, 2, 0)     # 20 -> 10

        # neck
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')  # 10 -> 40
        self.cbl7 = CBL(64, 32, 3, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # 40 -> 80

        # head
        self.cbl8 = CBL(64, 32, 3, 1, 1)
        self.cbl9 = CBL(32, 16, 4, 4, 0)  # 80 -> 20
        self.fc1 = nn.Linear(16 * 20 * 20, 25)  # 6400
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(25, 10)
        self.fc3 = nn.Linear(10, 4)

    def forward(self, x):
        # backbone
        layer1 = self.csp1(self.cbl1(x))       # (32, 320, 320)
        layer2 = self.cbl2(layer1)             # (32, 160, 160)
        layer3 = self.csp2(self.cbl3(layer2))  # (32, 80, 80)
        layer4 = self.cbl4(layer3)             # (32, 40, 40)
        layer5 = self.csp3(self.cbl5(layer4))  # (32, 20, 20)
        layer6 = self.cbl6(layer5)             # (32, 10, 10)

        # neck
        layer7 = self.upsample1(layer6)        # (32, 40, 40)
        layer8 = torch.cat((layer4, layer7), dim=1)  # (64, 40, 40)
        layer9 = self.cbl7(layer8)             # (32, 40, 40)
        layer10 = self.upsample2(layer9)       # (32, 80, 80)
        layer11 = torch.cat((layer3, layer10), dim=1)  # (64, 80, 80)

        # head
        layer12 = self.cbl8(layer11)           # (32, 80, 80)
        layer13 = self.cbl9(layer12)           # (16, 20, 20)
        layer13_flat = layer13.view(layer13.size(0), -1)  # (B, 6400)
        layer14 = self.dropout(self.fc1(layer13_flat))
        layer14 = torch.sigmoid(layer14)
        layer15 = self.fc2(layer14)
        layer15 = torch.sigmoid(layer15)
        layer16 = self.fc3(layer15)
        layer16 = torch.sigmoid(layer16)
        return layer16
    
def calculate_flops_and_params(model, input_size=(1, 3, 640, 640)):
    """计算模型的FLOPs和参数量（输入尺寸改为640x640）"""
    input = torch.randn(*input_size)
    flops, params = profile(model, inputs=(input,))
    
    flops_b = flops / 1e9
    params_m = params / 1e6
    return flops_b, params_m


if __name__ == "__main__":
    # 创建模型实例
    model = Net()
    
    # 计算FLOPs和参数量（输入尺寸640x640）
    flops_b, params_m = calculate_flops_and_params(model)
    print(f"模型计算量: {flops_b:.4f} GFLOPs")
    print(f"模型参数量: {params_m:.4f} M parameters")
    
    # 生成640x640的输入数据
    x = torch.randn(1, 3, 640, 640)
    
    # 前向传播
    try:
        output = model(x)
        print("模型输出形状:", output.shape)
        print("输入输出测试通过!")
    except Exception as e:
        print("输入输出测试失败:", e)