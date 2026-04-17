class Value:
    def __init__(self, data, _children=(), _op = "", label=""):
        self.data = data
        self.grad = 0.0
        self.children = _children 
        self._prev = set(_children)
        self._op = _op
        self.set_label(label)

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
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-')
        return out
    
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


if __name__ == "__main__":
    main()


        