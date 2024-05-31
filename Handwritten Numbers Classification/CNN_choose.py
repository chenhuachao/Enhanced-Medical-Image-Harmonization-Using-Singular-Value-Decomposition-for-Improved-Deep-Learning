import torch
import torch.nn as nn
from torchsummary import summary

class CnnNet(nn.Module):
    def __init__(self, classes=10, normalization_type='bn'):
        super(CnnNet, self).__init__()
        self.classes = classes
        self.normalization_type = normalization_type

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            self.get_normalization_layer(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            self.get_normalization_layer(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.get_normalization_layer(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.classes)

    def get_normalization_layer(self, channels):
        if self.normalization_type == 'bn':
            return nn.BatchNorm2d(channels)
        elif self.normalization_type == 'ln':
            return nn.LayerNorm([channels, 1, 1])
        elif self.normalization_type == 'in':
            return nn.InstanceNorm2d(channels)
        elif self.normalization_type == 'gn':
            return nn.GroupNorm(4, channels)  # You can adjust the number of groups
        elif self.normalization_type == 'sn':
            return nn.utils.spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.advpool(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    x = torch.rand((2, 3, 28, 28))
    cnn = CnnNet(classes=10, normalization_type='bn')  # You can change 'bn' to other types
    print(cnn)
    out = cnn(x)
    print(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.to(device)
    summary(cnn, (3, 28, 28))
