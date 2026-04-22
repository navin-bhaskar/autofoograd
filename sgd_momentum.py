import numpy as np

from optimizer import Optimizer

class SGDWithMomentum(Optimizer):

    def __init__(self, parameters, lr=0.01, beta=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            self.v[i] = self.v[i] * self.beta + p.grad
            p.data -= self.lr * self.v[i] 