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

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other: Value | float | int) -> Value:
        other = _coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Value(self.data * other.data, (self, other), "*")
