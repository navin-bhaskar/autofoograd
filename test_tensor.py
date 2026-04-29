from unittest import TestCase

import numpy as np

from tensor import Tensor, constant_fold, fuse_linear_relu
from mlp import MLP

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

    def test_constant_fold(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = Tensor(4.0)

        d = a * b + c  # (2*3) + 4 = 10

        optimized = constant_fold(d)

        assert optimized.data == 10

    def test_fused_linear_relu(self):
        x = Tensor(np.random.randn(1, 2))
        model = MLP()

        out = model(x)

        opt_out = fuse_linear_relu(out)

        # The top-level node remains '+' (output of l2), but first layer's relu(matmul+add) should be fused
        # Check that fusion happened in the graph by verifying some child was fused
        def has_fused_child(node, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return False
            visited.add(id(node))
            
            if 'fused_linear_relu' in node._op:
                return True
            for child in node._prev:
                if has_fused_child(child, visited):
                    return True
            return False
        
        self.assertTrue(has_fused_child(opt_out))

    def test_fused_relu_at_top(self):
        x = Tensor(np.random.randn(1, 2))
        W = Tensor(np.random.randn(2, 3))
        b = Tensor(np.random.randn(3))

        y = (x.matmul(W) + b).relu()

        opt = fuse_linear_relu(y)

        self.assertEqual(opt._op, "fused_linear_relu")