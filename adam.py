import numpy as np

from optimizer import Optimizer

class ADAM(Optimizer):
    """ADAM optimizier.
    This optimizer maintains two moving avergaes:
    1. First Momentum (m)
    2. Seconnd Momentum (Variance, v)
    m -> direction
    v -> confidence in the direction

    This is a stateful optimizer
    """

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

        self.t = 0

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            grad = p.grad

            # Calculate momentum
            # m = β1 * m + (1 - β1) * grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # v = β2 * v + (1 - β2) * grad²
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # update the grad; note eps is used for numerical stablity so that we do not end up with NaN
            # when v_hat is very small and sqrt -> 0
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        


