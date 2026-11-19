import numpy as np

from foograd.gradcheck import tensor_numerical_grads


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
