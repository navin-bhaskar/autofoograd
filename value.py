class Value:
    def __init__(self, data, label="", _children=(), _op = ""):
        self.data = data
        self.grad = 0.0
        self.children = _children 
        self._prev = set(_children)
        self._op = _op
        self.set_label(label)
        self._backward = lambda: None

    def set_label(self, label):
        if not label:
            self._label = self._op
        else:
            self._label = label

    def get_label(self):
        return self._label
    
    def get_prev(self):
        return self._prev

    def __add__(self, other):
        """Performs addition, returns new node representing this operation that was done"""
        out = Value(self.data + other.data, '+', (self, other), '+')

        def _backward():
            """
            Backpropagation rule for addition.

            General chain rule:
                input.grad += (∂out/∂input) * out.grad

            Where:
                out.grad = upstream gradient = dL/d(out)
                input.grad = gradient we are accumulating = dL/d(input)

            For addition:
                out = self + other

            Local derivatives:
                ∂out/∂self  = 1 
                ∂out/∂other = 1

            Applying chain rule:
                dL/d(self)  = dL/d(out) * ∂out/∂self  = out.grad * 1
                dL/d(other) = dL/d(out) * ∂out/∂other = out.grad * 1

            So we accumulate:
                self.grad  += 1 * out.grad
                other.grad += 1 * out.grad

            Note:
                We use += because a node can contribute to the output
                through multiple paths (gradient contributions must sum).
            """
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, '*', (self, other), '*')

        # Apply chain rule
        def _backward():
            """
            Backpropagation rule for multiplication.

            General chain rule:
                input.grad += (∂out/∂input) * out.grad

            For multiplication:
                out = self * other

            Local derivatives:
                ∂out/∂self  = other
                ∂out/∂other = self

            Applying chain rule:
                dL/d(self)  = out.grad * other.data
                dL/d(other) = out.grad * self.data

            So we accumulate:
                self.grad  += other.data * out.grad
                other.grad += self.data * out.grad

                Note:
                This is NOT the full product rule (f'g + g'f). 
                That emerges automatically when backpropagating
                through the graph via the chain rule.
                I was mixing this up, keep this in mind
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (Value(-1) * other)
    
    def __power__(self, power):
        out = Value(self.data ** power, f"self.data**{power}", (self,), "power")

        def _backward():
            """Local differenation is f'(x^n) = n*x^(n-1)"""
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(self.data, 0), f"relu({self.data})", (self,), "relu")

        def _backward():
            if out.data > 0:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    
    def build_topology(self):
        """Builds a topolgical graph, starting from self.
        In simple terms, we visit the parent before children (solving dependency)
        """
        topo = []
        visited = set()

        def dfs(cur_node):
            if cur_node not in visited:
                visited.add(cur_node)
                for child in cur_node._prev:
                    dfs(child)
                topo.append(cur_node)

        dfs(self)
        return topo
    
    def backward(self):
        """Builds topology and then computes the backwards gradient"""
        topo = self.build_topology()

        self.grad = 1.0 # seed gradient

        # start from this node which would appear at the very last of the topo array
        # due to the nature of dfs visit, hence reversed
        for node in reversed(topo):
            node._backward()

    
    def __repr__(self):
        return f"op: {self._op}, val: {self.data}, grad: {self.grad}"
    

    
def trace(root):
    cur = root
    stk = []
    level = 0
    
    node_info = (cur, level)
    stk.append(node_info)

    while stk:
        cur_node, cur_level = stk.pop()
        if cur_level > 0:
            print("|", end="")
            print("-"*cur_level*5, end="")
        print(cur_node.get_label())
        
        for child in cur_node.get_prev():
            stk.append((child, cur_level+1))

    
def main():
    a = Value(2.0, label="a")
    b = Value(3.0, label="b")
    c = a * b

    print(c)
    
    d = a * b + a

    trace(d)
    d.backward()
    print(a.grad)
    print(b.grad)

    f = (a * b) + (b * a)
    f.backward()

    print(a.grad)  # expect: 2b = 6
    print(b.grad)  # expect: 2a = 4


if __name__ == "__main__":
    main()


        