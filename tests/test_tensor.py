import numpy as np

from foograd import Tensor


def test_data_type_on_tensor():
    temp = Tensor([1, 2, 3])
    assert temp.shape == (3,)
    assert temp.data.dtype == np.float64


def test_zero_d_tensor():
    assert Tensor(5).shape == ()


def test_np_array_tensor():
    assert Tensor(np.ones((2, 3))).shape == (2, 3)


def test_grad_is_zero():
    t = Tensor(np.ones((2, 3)))
    np.testing.assert_array_equal(t.grad, np.zeros((2, 3)))


def test_backward():
    ten = Tensor([1, 2, 3])
    ten.backward()

    np.testing.assert_array_equal(ten.grad, np.ones((3,)))


def test_repr():
    assert str(Tensor([2, 3])).find("Tensor") != -1


def test_tensor_copies_input():
    arr = np.array([1.0, 2.0])
    t = Tensor(arr)
    arr[0] = 1.5
    assert t.data[0] == 1.0
