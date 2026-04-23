import numpy as np

from utils import unbroadcast

class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(data)
        self._op = _op
        self._label = label
        self._prev = set(_children)

        self._backward = lambda: None

    @staticmethod
    def _match_shape(grad, shape):
        if grad.shape == shape:
            return grad
        else:
            return grad.sum()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data + other.data, (self, other), _op="+", label="+")

        def _backward():
            # TODO: Revisitt for proper broadcast
            self.grad += Tensor._match_shape(out.grad, self.data.shape)
            other.grad += Tensor._match_shape(out.grad, other.data.shape)
            

        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += Tensor._match_shape(out.grad, self.data.shape)
            other.grad -= Tensor._match_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data * other.data, (self, other), _op="*", label="*")

        def _backward():
            # TODO: Revisitt for proper broadcast
            self_contrib = other.data * out.grad
            other_contib = self.data * out.grad

            self.grad += Tensor._match_shape(self_contrib, self.data.shape)
            other.grad += Tensor._match_shape(other_contib, other.data.shape)

        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), _op="relu")

        def _backward():
            self.grad += (out.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out
    
    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), _op="matmul", label="matmul")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def mean(self):
        out = Tensor(np.mean(self.data), (self,), 'mean')

        def _backward():
            self.grad += (1/self.data.size) * out.grad

        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad

            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)

            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward():
            grad_self = (1 / other.data) * out.grad
            grad_other = (-self.data / (other.data ** 2)) * out.grad

            self.grad += unbroadcast(grad_self, self.data.shape)
            other.grad += unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')

        def _backward():
            self.grad += -1 * out.grad

        out._backward = _backward
        return out
    
    def _topo_sort(self):
        topo_sorted = []
        visited = set()

        def _dfs(cur_node):
            if cur_node in visited:
                return
            visited.add(cur_node)
            for child in cur_node._prev:
                _dfs(child)
            topo_sorted.append(cur_node)

        _dfs(self)
        return topo_sorted
    
    def backward(self):
        topo_sorted = self._topo_sort()

        # clear out the gradients on all concerned nodes
        for node in topo_sorted:
            node.grad = np.zeros_like(node.data)

        # seed the current node grad with 1
        self.grad = np.ones_like(self.data)

        # compute backward gradient starting from the current node
        # hence we need to reverse the topo sorted node
        for node in reversed(topo_sorted):
            node._backward()


def main():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])

    c = a * b
    c.backward()

    print(a.grad)  # expect: [4, 5]
    print(b.grad)  # expect: [2, 3]

    # broad cast test
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor(2.0)

    c = a * b
    c.backward()

    print(a.grad)  # expect: [2, 2, 2]
    print(b.grad)  # expect: 1+2+3 = 6

if __name__ == "__main__":
    main()