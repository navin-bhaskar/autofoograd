"""Scalar value class for autograd."""

from __future__ import annotations

import math
from types import NotImplementedType


def _coerce(other: Value | float | int) -> Value | NotImplementedType:
    if isinstance(other, Value):
        return other
    if isinstance(other, (float, int)):
        return Value(other)

    return NotImplemented


class Value:
    def __init__(self, data: float, children: tuple[Value, ...] = (), op: str = "") -> None:
        self.data = float(data)
        self._children = children
        self._op = op
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            """Rule for self + other
            d(out)/d(self) = 1
            d(out)/d(other) = 1
            So just pass the out.grad to both inputs
            """
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | float | int) -> Value:
        return self + other

    def __sub__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        out = Value(self.data - other.data, (self, other), "-")

        def _backward() -> None:
            """Rule for         self - other
            d(out)/d(self) = 1 ( 1   -   0)
            d(out)/d(other) = -1(0   -   1)
            So just pass the out.grad to self and -out.grad to other
            """
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return other - self

    def __mul__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented

        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            """Chain rule when doing self*other:
            d(out)/d(self) = other
            d(out)/d(other) = self
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float | int) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1

    def __pow__(self, exponent: float | int) -> Value:
        if not isinstance(exponent, (float, int)):
            return NotImplemented

        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            """Chain rule when doing self**exponent:
            d(out)/d(self) = exponent * self**(exponent-1) or
            d(x^n)/dx = n*x^(n-1)
            """
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        out = self * (other**-1)  # a/b = a * b^-1 (b^-1 = 1/b)
        out._op = "/"
        return out

    def __rtruediv__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        out = other * (self**-1)  # b/a = b * a^-1 (a^-1 = 1/a)
        out._op = "/"
        return out

    def exp(self) -> Value:
        """Compute the exponential of this value."""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            """Chain rule when doing exp(self):
            d(out)/d(self) = exp(self) or d(e^x)/dx = e^x
            """
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Compute natural logarithm of this value."""
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            """Chain rule when doing log(self)
            d(out)/d(self) = 1 / self or d(ln(x))/dx = 1/x
            """
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Compute the ReLU of this value."""
        out = Value(max(0, self.data), (self,), "relu")

        def _backward() -> None:
            """Chain rule when doing ReLU(self)
            d(out)/d(self) = 1 if self > 0 else 0
            """
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Compute the hyperbolic tangent of this value."""
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            """Chain rule when doing tanh(self)
            d(out)/d(self) = 1 - tanh(self)^2 or d(tanh(x))/dx = 1 - tanh(x)^2
            """
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagate the gradients through the computational graph."""
        visited: set[Value] = set()
        topo_sorted_nodes: list[Value] = []

        # Starting from this node, build sorted topological order
        Value._topo_sort(self, visited, topo_sorted_nodes)
        # d(self)/d(self) = 1; seed this
        self.grad = 1.0
        # Reversed, so each node has its full gradient before passing it on
        for cur_node in reversed(topo_sorted_nodes):
            cur_node._backward()

    @staticmethod
    def _topo_sort(node: Value, visited: set[Value], topo_order: list[Value]) -> None:
        """Topological sort of the computational graph. Do a DFS and
        build the topological order of the nodes in the graph.
        This is used to ensure that we compute gradients in the correct
        order during backpropagation."""
        if node not in visited:
            visited.add(node)
            for child in node._children:
                Value._topo_sort(child, visited, topo_order)
            topo_order.append(node)
