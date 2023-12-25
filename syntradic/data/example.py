import numpy as np
import pandas as pd


class RandomData:
    def __init__(self, data_path):
        self.data = {
            ("114514.SZ", "20231211"): self.generate_random_data(),
            ("233333.SH", "20231211"): self.generate_random_data(),
        }
        self.col = ['trade_date','stock_code','stock_name','volume','topen','tclose','turnover_rate','high','low']
        self.orignal_data = pd.read_csv(r'stk_daily.csv').set_index('trade_date',inplace = True)
        self.stock_keys = set(list(self.orignal_data['stick_name']))

    def generate_random_data(self):
        data = pd.DataFrame()
        for column_name in ['trade_date','stock_code','stock_name','volume','topen','tclose','turnover_rate','high','low']:
            values = np.random.uniform(5, 10, 240)
            data[column_name] = values
        return data
    
    def keys(self):
        return [key for key in self.data]

    def query(self, key):
        return self.data[key]
    
