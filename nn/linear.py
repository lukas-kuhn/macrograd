from core.tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.zeros(out_features))

    def __call__(self, x):
        out = (x @ self.weights) + self.bias
        return out
    
    def parameters(self):
        return [self.weights, self.bias]
    