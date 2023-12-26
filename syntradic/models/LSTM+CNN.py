class CNNLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size):
        super(CNNLSTM, self).__init__()
        # CNN 层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=64 * 7 * 7, hidden_size=lstm_hidden_size, batch_first=True)
        # 线性层
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # 应用CNN层
        batch_size, seq_length, C, H, W = x.size()
        c_in = x.view(batch_size * seq_length, C, H, W)
        c_out = F.relu(self.conv1(c_in))
        c_out = self.pool(c_out)
        c_out = F.relu(self.conv2(c_out))
        c_out = self.pool(c_out)
        c_out = c_out.view(batch_size, seq_length, -1) # 重新整理为序列形式

        # 通过 LSTM 层
        _, (hidden, _) = self.lstm(c_out)
        # 取最后一个时间步的隐藏状态
        x = hidden[-1, :, :]
        # 通过线性层
        x = self.fc(x)
        return x
