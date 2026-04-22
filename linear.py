import numpy as np

from tensor import Tensor
from module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.1, label="W")
        self.b = Tensor(np.zeros(out_features), label="b")

    def __call__(self, x):
        return x.matmul(self.W) + self.b
    
    def parameters(self):
        return [self.W, self.b]
    

if __name__ == "__main__":
    x = Tensor([[1.0, 2.0]])
    layer = Linear(2, 3)

    out = layer(x)
    print(out.data.shape)  # should be (1, 3)