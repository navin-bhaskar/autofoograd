import numpy as np

from tensor import Tensor
from module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.1)
        self.b = Tensor(np.zeros(out_features))

    def __call__(self, x):
        return x.matmul(self.W) + self.b
    
    def parameters(self):
        return [self.W, self.b]
    