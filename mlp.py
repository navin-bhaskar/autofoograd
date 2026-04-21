from linear import Linear
from tensor import Tensor

class MLP:

    def __init__(self):
        self.l1 = Linear(2, 3)
        self.l2 = Linear(3, 1)

    def __call__(self, x):
        x = self.l1(x)
        print("l1 output:", x.data)
        x = x.relu()
        print("after relu:", x.data)
        x = self.l2(x)
        return x
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()
    

def mse(pred, target):
    diff = pred - target
    return (diff * diff).sum()


def main():
    model = MLP()

    x = Tensor([[1.0, 2.0]])
    y = Tensor([[1.0]])

    # forward
    pred = model(x)

    # loss
    loss = mse(pred, y)

    # backward
    loss.backward()

    lr = 0.01

    for p in model.parameters():
        p.data -= lr * p.grad

    for _ in range(100):
        pred = model(x)
        loss = mse(pred, y)

        loss.backward()

        for p in model.parameters():
            p.data -= 0.01 * p.grad
            p.grad = 0 * p.grad

    x = Tensor([[1.0, 2.0]])
    y = Tensor([[10.0]])

    model = MLP()

    for _ in range(50):
        pred = model(x)
        loss = mse(pred, y)

        loss.backward()

        print("The grad on model")
        for p in model.parameters():
            print(p.grad)

        for p in model.parameters():
            p.data -= 0.01 * p.grad
            p.grad = 0 * p.grad

        print(loss.data)


if __name__ == "__main__":
    main()