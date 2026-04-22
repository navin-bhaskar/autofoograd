from optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, paramters, lr=0.1):
        super().__init__(paramters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

