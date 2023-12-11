# Syntradic

Syntradic is a framework that can help you construct your own trading strategies.

Here is a demo (only a framework, not implemented now):

```python
from syntradic import RandomData, RandomTrader, RandomBacktester


data = RandomData(data_path="stock_data.csv")
trader = RandomTrader(model_path="an_interesting_model.pth")
backtester = RandomBacktester()
results = backtester.simulate_trade(trader, data)
print(results)
```
