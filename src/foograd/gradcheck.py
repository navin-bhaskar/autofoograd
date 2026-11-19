from collections.abc import Callable

import numpy as np

from foograd.tensor import Tensor
from foograd.value import Value


def numerical_grads(
    f: Callable[[list[Value]], Value], xs: list[float], eps: float = 1e-6
) -> list[float]:
    """Compute numerical gradients of f at xs using finite differences.
    df/dx = approx((f(x + eps) - f(x - eps)) / (2 * eps))

    params:
        f: function that takes a list of Values and returns a Value
        xs: list of floats to compute gradients at
        eps: small value to use for finite differences
    """
    grads: list[float] = []

    for i in range(len(xs)):
        # nudge the i-th input by eps in both directions
        plus = list(xs)
        plus[i] += eps

        minus = list(xs)
        minus[i] -= eps

        # Evaluate the function at the perturbed points
        f_plus = f([Value(x) for x in plus])
        f_minus = f([Value(x) for x in minus])

        # Compute the numerical gradient using central difference
        grad = (f_plus.data - f_minus.data) / (2 * eps)
        grads.append(grad)

    return grads


def gradcheck(
    f: Callable[[list[Value]], Value],
    xs: list[float],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Check the gradients of f at xs using numerical gradients."""
    # Compute analytical gradients using backpropagation
    inputs = [Value(x) for x in xs]
    output = f(inputs)
    output.backward()
    analytical_grads = [x.grad for x in inputs]

    # Compute numerical gradients
    numerical_grads_list = numerical_grads(f, xs, eps)

    # Compare the two sets of gradients
    return all(
        abs(analytical - numerical) <= atol + rtol * abs(numerical)
        for analytical, numerical in zip(analytical_grads, numerical_grads_list, strict=True)
    )


def tensor_numerical_grads(
    f: Callable[[list[Tensor]], Tensor], xs: list[np.ndarray], eps: float = 1e-6
) -> list[np.ndarray]:
    """Do numerical gradient on Tensor. The idea is same as numerical_grads() above.
    Nudge the inputs slightly in both directions and measure the change.
    In backward(), out.grad is set to ones as seed, which is the gradient of out.sum().
    So we measure f(...).data.sum() as well to match what our autograd engine would do"""
    base = [np.array(x, dtype=np.float64) for x in xs]
    grads: list[np.ndarray] = []

    for k, x in enumerate(base):
        grad = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            plus = [a.copy() for a in base]
            plus[k][idx] += eps

            minus = [a.copy() for a in base]
            minus[k][idx] -= eps

            # Add up all elements to get the total deviation
            f_plus = f([Tensor(val) for val in plus]).data.sum()
            f_minus = f([Tensor(val) for val in minus]).data.sum()

            grad[idx] = (f_plus - f_minus) / (2 * eps)
        grads.append(grad)
    return grads
