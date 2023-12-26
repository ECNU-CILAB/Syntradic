class SelfAttentionModule(nn.Module):
    def __init__(self, size):
        super(SelfAttentionModule, self).__init__()
        self.size = size
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.size**0.5
        attention_weights = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, V)

class SelfAttentionLSTM(nn.Module):
    def __init__(self):
        super(SelfAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=40, batch_first=True)
        self.self_attention = SelfAttentionModule(size=40)
        self.fc = nn.Linear(40, 3)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        attention_out = self.self_attention(lstm_out)
        logits = self.fc(attention_out[:, -1, :]) 
        return F.softmax(logits, dim=1)
