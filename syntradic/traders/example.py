import numpy as np

class RandomTrader:
    def __init__(self, model_path):
        pass

    def __call__(self, data):
        length = len(data)
        signal = np.random.randn(length).tolist()
        return signal
