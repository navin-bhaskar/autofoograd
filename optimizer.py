import numpy as np

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplemented("Implement the step function in derived class")
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)
