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
    __array_ufunc__ = None  # Stop numpy from taking over our operations on np arrays

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

    def __add__(self, other: Tensor | ArrayLike) -> Tensor | NotImplementedType:
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

    def __radd__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        return self + other

    def __mul__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            """Chain rule when doing self*other:
            d(out)/d(self) = other
            d(out)/d(other) = self
            """
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        return self * other

    def __pow__(self, exponent: int | float) -> Tensor | NotImplementedType:
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            """Chain rule when doing self**exponent:
            d(out)/d(self) = exponent * self**(exponent-1) or
            d(x^n)/dx = n*x^(n-1)
            """
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor | NotImplementedType:
        out = -1 * self
        out._op = "neg"
        return out

    def __sub__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        out = self + -other
        out._op = "-"
        return out

    def __rsub__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other - self

    def __truediv__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        out = self * (other**-1)
        out._op = "/"
        return out

    def __rtruediv__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return other
        return other / self

    def __matmul__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(
                f"Unsupported dimensions for Tensor matmul got {self.shape} @ {other.shape}"
            )

        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            """Matmul is not commutative arrange the 'out.grad' so that
            matrix rule of multiplication that only matrix of shape MxN and NxK
            can be multiplied.
            Let self -> MxN and other -> NxK then out.grad -> MxK
            self.T -> NxM; other.T -> KxN

            out.grad @ other.data.T -> MxK @ KxN -> MxN
            self.data.T @ out.grad -> NxM @ MxK -> NxK
            """
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __rmatmul__(self, other: ArrayLike | Tensor) -> Tensor | NotImplementedType:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other @ self

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """d(out)/d(x_i) = 1 for every element, so copy(broadcast) the out.grad into
        every slot of self.grad while properly handling the dims and axes
        """
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)

            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """In mean we divide sum by number of elements the contibuted to that sum, n.
        When axis is not given, then n is simply number of elements.
        When axis is given we need to consider N as number of elements that contibuted
        to sum of that particular element.
        Example 1: a = [1, 2, 3], [4, 5, 6] axis=0
        sum = [1+4, 2+5, 3+6] N = 2
        Example 2:a = [1, 2, 3], [4, 5, 6] axis=1
        sum = [1+2+3, 4+5+6] N = 3
        Formula then is: N = size of a/size of out
        Example 1      : N = 6        / 3          2
        Example 2      : N = 6        / 2          3
        """
        out = self.sum(axis, keepdims)
        n = self.data.size / out.data.size
        out = out / n
        out._op = "mean"
        return out

    def reshape(self, *newshape: int) -> Tensor:
        """Reshapes the tensor. The gradient is also a reshped transfer to the node above
        out"""
        out = Tensor(self.data.reshape(newshape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out


def _coerce(other: float | int | np.ndarray | Tensor) -> Tensor | NotImplementedType:
    if isinstance(other, Tensor):
        return other
    if isinstance(other, (float, int, np.ndarray)):
        return Tensor(other)
    return NotImplemented
