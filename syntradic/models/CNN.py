class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层
        self.fc = nn.Linear(64 * 7 * 7, 10) # 假设输入图像大小为28x28

    def forward(self, x):
        # 通过第一个卷积层
        x = F.relu(self.conv1(x))
        # 通过最大池化层
        x = self.pool(x)
        # 通过第二个卷积层
        x = F.relu(self.conv2(x))
        # 再次通过最大池化层
        x = self.pool(x)
        # 展平操作
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fc(x)
        return x
