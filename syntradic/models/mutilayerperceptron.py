class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        # 第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size) 
        # 非线性激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        # 通过第一个全连接层
        x = self.fc1(x)
        # 应用ReLU激活函数
        x = self.relu(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x
