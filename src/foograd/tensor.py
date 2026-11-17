from __future__ import annotations

from types import NotImplementedType

import numpy as np
from numpy.typing import ArrayLike


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


class Tensor:
    def __init__(self, data: ArrayLike, children: tuple[Tensor, ...] = (), op: str = "") -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._children = children
        self._op = op
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self) -> None:
        """Backprop the gradients on the tensor"""
        visited: set[Tensor] = set()
        topo_sorted_nodes: list[Tensor] = []

        self.grad.fill(1.0)
        Tensor._topo_sort(self, visited, topo_sorted_nodes)

        for cur_node in reversed(topo_sorted_nodes):
            cur_node._backward()

    @staticmethod
    def _topo_sort(cur_node: Tensor, visited: set[Tensor], topo_sorted_nodes: list[Tensor]) -> None:
        """Does DFS for building topological sorting of nodes"""
        if cur_node not in visited:
            visited.add(cur_node)
            for child in cur_node._children:
                Tensor._topo_sort(child, visited, topo_sorted_nodes)
            topo_sorted_nodes.append(cur_node)

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            """Rule for self + other
            d(out)/d(self) = 1
            d(out)/d(other) = 1
            So just pass the out.grad to both inputs
            Note that the shape of out grad has to be mapped to original inputs.
            So use the rules as specified in unbroadcast
            """
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike | Tensor) -> Tensor:
        return self + other


def _coerce(other: float | int | np.ndarray) -> Tensor | NotImplementedType:
    if isinstance(other, Tensor):
        return other
    if isinstance(other, (float, int, np.ndarray)):
        return Tensor(other)
    return NotImplemented
