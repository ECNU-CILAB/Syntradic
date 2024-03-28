import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from model import LSTM
import seaborn as sns
import math, time
from sklearn.metrics import mean_squared_error

def split_data(stock, lookback):
    data_raw = stock.to_numpy() 
    data = []    
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:] 
    y_train = data[:train_set_size,-1,0:1]
    
    x_test = data[train_set_size:,:-1,:] 
    y_test = data[train_set_size:,-1,0:1] 
    
    return [torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test),torch.Tensor(y_test)]


filepath = 'dataset/csi300_20210311_20231229.csv'
data = pd.read_csv(filepath)

price = data[['close', 'volume']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['close'] = scaler.fit_transform(price['close'].values.reshape(-1,1))
price['volume'] = scaler.fit_transform(price['volume'].values.reshape(-1,1))

lookback = 5
x_train, y_train, x_test, y_test = split_data(price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

input_dim = 2
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

log = np.zeros(num_epochs)
lstm = []

for i in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train)
    log[i] = loss.item()
    print("Epoch ", i, "MSE: ", loss.item())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
predict.to_csv('predict_data.csv')
original = pd.DataFrame(scaler.inverse_transform(y_train.detach().numpy()))

y_test_pred = model(x_test)

#反归一
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

#平均数方差
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)

