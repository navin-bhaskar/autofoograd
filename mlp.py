import numpy as np

from linear import Linear
from tensor import Tensor
from sgd import SGD
from sgd_momentum import SGDWithMomentum

class MLP:

    def __init__(self):
        self.l1 = Linear(2, 3)
        self.l2 = Linear(3, 1)

    def __call__(self, x):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)
    

def mse(pred, target):
    diff = pred - target
    return (diff * diff).sum()


def main():
    model_with_sgd = MLP()
    model_with_sgd_momentunm = MLP()

    x = Tensor([[1.0, 2.0]])
    y = Tensor([[10.0]])

    optimizer_sgd = SGD(model_with_sgd.parameters())
    optimizer_sgd_momentum = SGDWithMomentum(model_with_sgd_momentunm.parameters(), lr=0.001, beta=0.8)

    for step in range(50):
        pred_with_sgd = model_with_sgd(x)
        loss_with_sgd = mse(pred_with_sgd, y)

        loss_with_sgd.backward()
        optimizer_sgd.step()
        
        pred_with_sgd_momentum = model_with_sgd_momentunm(x)
        loss_with_sgd_momentum = mse(pred_with_sgd_momentum, y)

        loss_with_sgd_momentum.backward()
        optimizer_sgd_momentum.step()

        print(f"step {step}: SGD loss: {loss_with_sgd.data} SGD momentum loss: {loss_with_sgd_momentum.data}")


if __name__ == "__main__":
    main()