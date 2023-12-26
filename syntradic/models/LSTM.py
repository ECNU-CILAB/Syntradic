class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleLSTM, self).__init__()
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 线性层，将LSTM的输出转换为类别输出
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 通过LSTM层
        _, (hidden, _) = self.lstm(x)
        # 取最后一个时间步的隐藏状态
        x = hidden[-1, :, :]
        # 通过线性层
        x = self.fc(x)
        return x
