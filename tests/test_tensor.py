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


def test_tensor_add_on_same_shape():
    a = Tensor([1, 2])
    b = Tensor([3, 4])

    out = a + b
    np.testing.assert_array_equal(out.data, [4, 6])

    out.backward()

    np.testing.assert_array_equal(out.grad, [1, 1])
    np.testing.assert_array_equal(a.grad, [1, 1])
    np.testing.assert_array_equal(b.grad, [1, 1])


def test_tensor_add_with_broadcast():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor(np.ones((2,)))

    out = a + b
    np.testing.assert_array_equal(out.data, [[2, 3], [4, 5], [6, 7]])

    out.backward()
    np.testing.assert_array_equal(b.grad, [3, 3])
    np.testing.assert_array_equal(a.grad, [[1, 1], [1, 1], [1, 1]])


def test_tensor_radd():
    a = Tensor([[1, 2], [2, 2]])

    out = 2 + a
    np.testing.assert_array_equal(out.data, [[3, 4], [4, 4]])

    out.backward()
    np.testing.assert_array_equal(a.grad, [[1, 1], [1, 1]])


def test_tensor_add_3_4_and_3_1():
    a = Tensor(np.ones((3, 4)))
    b = Tensor(np.ones((3, 1)))

    c = a + b
    c.backward()
    np.testing.assert_array_equal(a.grad, np.ones((3, 4)))
    np.testing.assert_array_equal(b.grad, 4 * np.ones((3, 1)))


def test_tensor_add_with_number():
    a = Tensor([1, 1])
    t = a + 2

    t.backward()
    np.testing.assert_array_equal(t.data, [3.0, 3.0])
    np.testing.assert_array_equal(a.grad, [1, 1])


def test_tensor_sum_twice_contribution():
    a = Tensor([1, 2])
    c = a + a

    c.backward()
    np.testing.assert_array_equal(a.grad, [2, 2])


def test_tensor_mul_with_broadcast():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor([10, 20])

    out = a * b
    np.testing.assert_array_equal(out.data, [[10, 40], [30, 80], [50, 120]])

    out.backward()
    np.testing.assert_array_equal(a.grad, [[10, 20], [10, 20], [10, 20]])

    np.testing.assert_array_equal(b.grad, [9, 12])


def test_tensor_mul_with_same_shape():
    a_data = [[1, 2], [3, 4]]
    b_data = [[4, 5], [6, 7]]

    a = Tensor(a_data)
    b = Tensor(b_data)

    c = a * b
    np.testing.assert_array_equal(c.data, [[4, 10], [18, 28]])

    c.backward()
    # The data of the 'other' is what we are expecting gradients to be
    np.testing.assert_array_equal(a.grad, b_data)
    np.testing.assert_array_equal(b.grad, a_data)


def test_tensor_rmul():
    a = Tensor([1, 2])

    c = 3 * a
    np.testing.assert_array_equal(c.data, [3, 6])

    c.backward()
    np.testing.assert_array_equal(a.grad, [3, 3])


def test_tensor_mul_self():
    a = Tensor([2, 3])
    c = a * a

    np.testing.assert_array_equal(c.data, [4, 9])

    c.backward()
    np.testing.assert_array_equal(a.grad, [4, 6])
