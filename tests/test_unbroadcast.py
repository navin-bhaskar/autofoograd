import numpy as np

from foograd.tensor import unbroadcast


def test_unbroadcast_on_3_2_to_2():
    grad = np.array([[1, 0], [2, 0], [3, 1]])
    out = unbroadcast(grad, (2,))
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, [6, 1])


def test_unbroadcast_on_3_4_to_3_1():
    # shape = 3, 4
    grad = np.array([[1, 0, 2, 3], [2, 1, 3, 4], [4, 5, 6, 7]])
    out = unbroadcast(grad, (3, 1))
    assert out.shape == (3, 1)
    np.testing.assert_array_equal(out, [[6], [10], [22]])


def test_unbroadcast_on_3_4_to_1_4():
    # shape = 3,4
    grad = np.array([[1, 0, 2, 3], [2, 1, 3, 4], [4, 5, 6, 7]])
    out = unbroadcast(grad, (1, 4))
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out, [[7, 6, 11, 14]])


def test_unbroadcast_on_2_3_4_to_3_1():
    # shape 2, 3, 4
    grad = np.ones((2, 3, 4))
    out = unbroadcast(grad, (3, 1))
    assert out.shape == (3, 1)
    np.testing.assert_array_equal(out, np.full((3, 1), 8))


def test_unbroadcast_on_3_2_to_3_2_no_op():
    # shape 3, 2
    grad = np.array([[1, 0], [2, 0], [3, 1]])
    out = unbroadcast(grad, (3, 2))
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, np.array([[1, 0], [2, 0], [3, 1]]))


def test_unbroadcast_scalar():
    grad = np.array([[1, 0], [2, 0], [3, 1]])
    out = unbroadcast(grad, ())
    assert out.shape == ()
    assert out == 7
