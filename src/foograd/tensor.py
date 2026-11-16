import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduces a gradient back to the shape of the input that was broadcast.
    Follows two rules in this order
    1. While grad has more dims than shape, sum axis 0 (keepdims=False):
       these dims were padded on the left and never existed in the input.
    2. For each axis i where shape[i] == 1 but grad.shape[i] != 1,
       sum axis i with keepdims=True: the input had size 1 there and was copied."""

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0, keepdims=False)

    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
