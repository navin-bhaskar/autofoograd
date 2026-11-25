import numpy as np
import pytest

from foograd.gradcheck import tensor_gradcheck, tensor_numerical_grads
from foograd.tensor import Tensor


def test_tensor_numerical_grads_pow_x_2():
    def func(xs):
        return xs[0] ** 2

    x = np.array([[1, 2, 3], [3, 4, 3]], dtype=np.float64)
    grads = tensor_numerical_grads(func, [x])
    # d(x^2)/dx = 2x or 2*data
    np.testing.assert_allclose(grads[0], 2 * x, atol=1e-5)


def test_tensor_numerical_grads_do_not_mutate_data():
    def func(xs):
        return xs[0] ** 3

    x = np.array([[1, 2, 3], [3, 4, 3]], dtype=np.float64)
    x_before = x.copy()
    tensor_numerical_grads(func, [x])
    np.testing.assert_array_equal(x, x_before)


def test_tensor_grad_check():
    def test_func(xs):
        return xs[0] * xs[1]

    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    b = np.array([7, 10])

    assert tensor_gradcheck(test_func, [a, b])


def test_tensor_grad_check_fails_on_broken_backward(monkeypatch):
    def broken_backward_mul(this, other):
        out = Tensor(this.data * other.data, (this, other), "*")

        def _backward() -> None:
            "Forget *out.grad, also keep same shape will not work other wise"
            this.grad += other.data
            other.grad += this.data

        out._backward = _backward
        return out

    monkeypatch.setattr(Tensor, "__mul__", broken_backward_mul)

    def test_func(xs):
        # a*a*b -> a^2b
        # let t -> (a*b)
        # a*a*b = ta
        # t*a is last node so gradient is all 1 (seed)
        # now t = a*b should get the gradient from the outer node correctly but does
        # not due to missing *grad
        return (xs[0] * xs[1]) * xs[0]

    # Make both same shape we do not have unbroadcast here
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    b = np.array([[4, 5], [6, 7], [8, 8]], dtype=np.float64)

    assert not tensor_gradcheck(test_func, [a, b])


BINARY_OPS = {
    "add": lambda xs: xs[0] + xs[1],
    "sub": lambda xs: xs[0] - xs[1],
    "mul": lambda xs: xs[0] * xs[1],
    "div": lambda xs: xs[0] / xs[1],
}

SHAPES = [
    ((3, 2), (3, 2)),  # Same shape
    ((3, 2), (2,)),  # Left padded
    ((3, 4), (3, 1)),  # size 1 axis
    ((2, 3, 4), (3, 1)),  # both rules at once
    ((3, 2), ()),  # scalar
]

ALL_SHAPES = SHAPES + [(s1, s0) for s0, s1 in SHAPES if s0 != s1]


@pytest.mark.parametrize("op_name", BINARY_OPS)
@pytest.mark.parametrize("shape_a, shape_b", ALL_SHAPES, ids=[f"{a}--{b}" for a, b in ALL_SHAPES])
def test_binary_ops_gradcheck(op_name, shape_a, shape_b):
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 2.0, size=shape_a)
    b = rng.uniform(0.5, 2.0, size=shape_b)
    assert tensor_gradcheck(BINARY_OPS[op_name], [a, b])


UNARY_OPS = {
    "neg": lambda xs: -xs[0],
    "pow": lambda xs: xs[0] ** 2,
    "pow3": lambda xs: xs[0] ** 3,
    "pow_neg1": lambda xs: xs[0] ** -1,
    "sqrt": lambda xs: xs[0] ** 0.5,
}


UNARY_OPS_SHAPES = [(), (3,), (2, 3)]


@pytest.mark.parametrize("op_name", UNARY_OPS)
@pytest.mark.parametrize("shape", UNARY_OPS_SHAPES, ids=[f"{a}" for a in UNARY_OPS_SHAPES])
def test_unary_ops_gradcheck(op_name, shape):
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 2.0, size=shape)
    assert tensor_gradcheck(UNARY_OPS[op_name], [a])


def test_composite_ops_grad_check():
    def test_func(xs):
        # (a*b - a/b) ** 2
        return (xs[0] * xs[1] - xs[0] / xs[1]) ** 2

    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([8, 9])

    assert tensor_gradcheck(test_func, [a, b])


def test_matmul_op_grad_check():
    def matmul_func(xs):
        # a @ b
        return xs[0] @ xs[1]

    rng = np.random.default_rng(0)
    a = rng.integers(1, 100, size=(2, 3))
    b = rng.integers(1, 100, size=(3, 4))

    assert tensor_gradcheck(matmul_func, [a, b])


def test_sum():
    def tensor_sum(xs):
        return xs[0] * xs[0].sum()

    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 2.0, size=(2, 3))

    assert tensor_gradcheck(tensor_sum, [a])


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_axis_gradcheck(axis, keepdims):
    def summ_sqr(xs):
        return xs[0].sum(axis, keepdims) ** 2

    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 5.0, size=(2, 3))

    assert tensor_gradcheck(summ_sqr, [a])


@pytest.mark.parametrize("method", ["sum", "mean"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reduction_gradcheck(method, axis, keepdims):
    def mean_sum(xs):
        return getattr(xs[0], method)(axis, keepdims) ** 2

    rng = np.random.default_rng(0)
    a = rng.uniform(0.55, 5.55, size=(2, 3))
    assert tensor_gradcheck(mean_sum, [a])
