import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile  # 用于计算FLOPs和参数量


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
        self.leakyrelu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.leakyrelu(out)
        return out


class ConvModule(nn.Module):
    def __init__(self, c_in):
        super(ConvModule, self).__init__()
        self.cbl1 = CBL(c_in, c_in, 3, 1, 1)
        self.cbl2 = CBL(c_in, c_in, 3, 1, 1)
        self.conv = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.batchnorm = nn.BatchNorm2d(c_in)

    def forward(self, x):
        b = self.cbl1(x)
        b = self.cbl2(b)
        b = self.conv(b)
        b = self.batchnorm(b)
        b = F.leaky_relu(b)
        b = self.cbl1(b)
        return b


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        self.cbl1 = CBL(3, 16, 2, 2, 0)
        self.conv1 = ConvModule(c_in=16)
        self.cbl2 = CBL(16, 24, 2, 2, 0)
        self.cbl3 = CBL(24, 32, 2, 2, 0)
        self.conv2 = ConvModule(c_in=32)
        self.cbl4 = CBL(32, 64, 2, 2, 0)

        self.cbl5 = CBL(64, 128, 2, 2, 0)
        self.conv3 = ConvModule(c_in=128)
        self.cbl6 = CBL(128, 64, 2, 2, 0)

        # neck
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.cbl7 = CBL(128, 32, 3, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # head - 调整全连接层输入维度为50（640x640输入时）
        self.cbl8 = CBL(64, 16, 4, 4, 0)
        self.cbl9 = CBL(16, 2, 4, 4, 0)
        self.fc1 = nn.Linear(50, 25)  # 640x640输入时展平维度为5*5*2=50
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(25, 10)
        self.fc3 = nn.Linear(10, 4)

    def forward(self, x):
        print(f"输入尺寸: {x.shape[2:]}")  # 打印输入尺寸
        
        # backbone
        layer1 = self.conv1(self.cbl1(x))
        
        layer2 = self.cbl2(layer1)
        
        layer3 = self.conv2(self.cbl3(layer2))
        
        layer4 = self.cbl4(layer3)
        
        layer5 = self.conv3(self.cbl5(layer4))
        
        # neck
        layer6 = self.cbl6(layer5)
        
        layer7 = self.upsample1(layer6)
        
        layer8 = torch.cat((layer4, layer7), 1)
        
        layer9 = self.cbl7(layer8)
        
        layer10 = self.upsample2(layer9)
        
        layer11 = torch.cat((layer3, layer10), 1)
        
        # head
        layer12 = self.cbl8(layer11)
        
        layer13 = self.cbl9(layer12)
        
        layer13_ = layer13.view(layer13.shape[0], -1)
        
        layer14 = self.dropout(self.fc1(layer13_))
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