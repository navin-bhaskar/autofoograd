"""Scalar value class for autograd."""

from __future__ import annotations


def _coerce(other: Value | float | int) -> Value:
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
