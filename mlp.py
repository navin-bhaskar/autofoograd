import numpy as np

from linear import Linear
from tensor import Tensor
from sgd import SGD
from sgd_momentum import SGDWithMomentum

class MLP:

    def __init__(self, num_classes=2, hidden=4):
        self.l1 = Linear(2, hidden)
        # num_classes -> number of classes to classify in output
        # hidden -> hidden layers out
        self.l2 = Linear(hidden, num_classes)

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


def softmax(x):
    shifted = x - x.data.max(axis=1, keepdims=True)
    exps = shifted.exp()
    return exps / exps.sum(axis=1, keepdims=True)

def cross_entropy(pred, target):
    log_preds = pred.log()
    loss = -(target * log_preds).sum(axis=1)
    return loss.mean()

def main():
    model = MLP()
   
    x = Tensor([[1.0, 2.0, 3.0]])
    y = x.sum()
    y.backward()

    print(x.grad)  # should be all 1s

    x = Tensor([[1.0, 2.0]])
    y = Tensor([[0.0, 1.0]])  # class 1 (one-hot)

    optimizer = SGD(model.parameters())

    for step in range(100):
        logits = model(x)
        probs = softmax(logits)

        loss = cross_entropy(probs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(step, loss.data)
    print(probs.data)


if __name__ == "__main__":
    main()