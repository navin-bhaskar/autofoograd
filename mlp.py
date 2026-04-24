import numpy as np
import matplotlib.pyplot as plt

from linear import Linear
from tensor import Tensor
from sgd import SGD
from sgd_momentum import SGDWithMomentum
from adam import ADAM

class MLP:

    def __init__(self, inp=2, num_classes=2, hidden=4):
        self.l1 = Linear(inp, hidden)
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
    model = MLP(10)
    initial_params = [p.data.copy() for p in model.parameters()]

    x = Tensor(np.random.randn(100, 10))
    y = Tensor([[0.0, 1.0]])  # class 1 (one-hot)

    optimizers = [
        ('SGD', SGD(model.parameters(), lr=0.1)),
        ('SGDWithMomentum', SGDWithMomentum(model.parameters(), lr=0.01, beta=0.9)),
        ('ADAM', ADAM(model.parameters(), lr=0.01))
    ]

    results = {}
    for name, optimizer in optimizers:
        # reset params
        for i, p in enumerate(model.parameters()):
            p.data = initial_params[i].copy()
        # reset grads
        model.zero_grad()

        losses = []
        for step in range(100):
            optimizer.zero_grad()
            logits = model(x)
            probs = softmax(logits)
            loss = cross_entropy(probs, y)
            losses.append(loss.data)
            loss.backward()
            optimizer.step()

        results[name] = losses
        print(f"{name} final loss: {losses[-1]:.4f}")

    # Print losses at every 10 steps for comparison
    print("\nLoss convergence comparison:")
    print("Step\tSGD\t\tMomentum\tADAM")
    for step in range(0, 100, 10):
        sgd_loss = results['SGD'][step]
        mom_loss = results['SGDWithMomentum'][step]
        adam_loss = results['ADAM'][step]
        print(f"{step}\t{sgd_loss:.4f}\t{mom_loss:.4f}\t{adam_loss:.4f}")

    # Plot the loss convergence
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Optimizer Loss Convergence Comparison')
    plt.legend()
    plt.savefig('optimizer_comparison.png')
    print("\nPlot saved as 'optimizer_comparison.png'")


if __name__ == "__main__":
    main()