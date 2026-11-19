import numpy as np

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
