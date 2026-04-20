from unittest import TestCase

from tensor import Tensor

class TestTensor(TestCase):

    def test_tensor_backward_add(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        c = a + b
        c.backward()

        assert (a.grad == [1.0, 1.0, 1.0]).all()
        assert (b.grad == [1.0, 1.0, 1.0]).all()

    def test_add_vector_scalar(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor(2.0)

        c = a + b
        c.backward()

        assert (a.grad == [1.0, 1.0, 1.0]).all()
        assert b.grad == 3.0

    def test_add_scalar_vector(self):
        a = Tensor(2.0)
        b = Tensor([1.0, 2.0, 3.0])

        c = a + b
        c.backward()

        assert a.grad == 3.0
        assert (b.grad == [1.0, 1.0, 1.0]).all()

    def test_add_chain(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        c = a + b + a
        c.backward()

        assert (a.grad == [2.0, 2.0, 2.0]).all()
        assert (b.grad == [1.0, 1.0, 1.0]).all()

    def test_add_mixed_graph(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor(2.0)
        c = Tensor([1.0, 1.0, 1.0])

        d = (a + b) + c
        d.backward()

        assert (a.grad == [1.0, 1.0, 1.0]).all()
        assert b.grad == 3.0
        assert (c.grad == [1.0, 1.0, 1.0]).all()

    def test_add_reuse(self):
        a = Tensor([1.0, 2.0, 3.0])

        c = a + a
        c.backward()

        assert (a.grad == [2.0, 2.0, 2.0]).all()

    def test_add_multiple_backward_calls(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        c = a + b
        c.backward()

        # reset
        a.grad = 0 * a.grad
        b.grad = 0 * b.grad

        c = a + b
        c.backward()

        assert (a.grad == [1.0, 1.0, 1.0]).all()
        assert (b.grad == [1.0, 1.0, 1.0]).all()

    def test_add_single_element_vector(self):
        a = Tensor([1.0])
        b = Tensor(2.0)

        c = a + b
        c.backward()

        assert (a.grad == [1.0]).all()
        assert b.grad == 1.0

    