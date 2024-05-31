import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SwitchableNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(SwitchableNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # 权重参数，用于学习在BN, IN和LN之间的最佳组合
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

        self.bn = nn.BatchNorm2d(num_features, momentum=momentum)
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))

    def forward(self, x):
        # 计算BN, IN, LN的均值和方差
        bn_mean, bn_var = self.bn.running_mean.view(1, -1, 1, 1), self.bn.running_var.view(1, -1, 1, 1)
        in_mean, in_var = torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True)
        ln_mean, ln_var = torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)

        # 使用expand_as确保mean和var与x形状一致
        mean = bn_mean * self.mean_weight[0] + in_mean * self.mean_weight[1] + ln_mean * self.mean_weight[2]
        var = bn_var * self.var_weight[0] + in_var * self.var_weight[1] + ln_var * self.var_weight[2]

        # 确保mean和var的形状与x兼容
        mean = mean.expand_as(x)
        var = var.expand_as(x)

        # 归一化
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias



class CnnNet(nn.Module):
    def __init__(self, classes=10, normalization='bn'):
        super(CnnNet, self).__init__()
        self.normalization = normalization
        self.classes = classes

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.norm1 = self._get_normalization_layer(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.norm2 = self._get_normalization_layer(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.norm3 = self._get_normalization_layer(64)

        # 自适应池化
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(64, classes)

    def _get_normalization_layer(self, num_features):
        if self.normalization == 'bn':
            return nn.BatchNorm2d(num_features)
        elif self.normalization == 'ln':
            # Layer Norm
            return nn.GroupNorm(1, num_features)
        elif self.normalization == 'in':
            # Instance Norm
            return nn.InstanceNorm2d(num_features)
        elif self.normalization == 'gn':
            # Group Norm
            return nn.GroupNorm(8, num_features)  # 假设分组数量为8
        elif self.normalization == 'sn':
            # Switchable Norm (需要自定义实现或者查找实现)
            return SwitchableNormalization(num_features)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.advpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    cnn = CnnNet(classes=10, normalization='bn').to(device)  # 使用SwitchableNormalization作为示例

    # 打印模型架构
    print(cnn)

    # 使用torchsummary提供模型的概览
    summary(cnn, input_size=(3, 28, 28), device=device.type)

    # 生成一个随机输入，以验证模型是否能够正常运行
    x = torch.rand((2, 3, 28, 28)).to(device)
    out = cnn(x)
    print(out.shape)  # 应该输出与类别数匹配的尺寸，例如torch.Size([2, 10])对于10个类别

