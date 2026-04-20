import unittest

from value import Value

class TestValue(unittest.TestCase):
    def test_label(self):
        val = Value(1, label='a')
        self.assertEqual(val.data, 1)
        self.assertEqual(val.get_label(), 'a')

    def test_children(self):
        a = Value(1, label="a")
        b = Value(2, label="b")

        c = a + b

        #     
        #   a---- \
        #          +
        #   b-----/
        #
        # a and b are previous of '+'

        self.assertEqual(a in c.get_prev(), True) 
        self.assertEqual(b in c.get_prev(), True)

    def test_gradient(self):
        a = Value(2.0, label="a")
        b = Value(3.0, label="b")
    
        d = a * b + a

        d.backward()
        self.assertEqual(a.grad, 4)
        self.assertEqual(b.grad, 2)

    def test_gradient_contribution(self):
        a = Value(2.0, label="a")
        b = Value(3.0, label="b")

        f = (a * b) + (b * a)
        f.backward()

        self.assertEqual(a.grad, 6)  # expect: 2b = 6
        self.assertEqual(b.grad, 4)  # expect: 2a = 4

    def test_topo_sort(self):
        a = Value(2.0, label="a")
        b = Value(3.0, label="b")

        c = a * a + b

        # a ----\       
        #        *----- + == c
        # a ----/       |
        #               |
        # b ------------
        topo_sorted = c.build_topology()

        # total nodes, a, b, * and +
        self.assertEqual(len(topo_sorted), 4)
        # The '+' output at the very end
        self.assertEqual(topo_sorted[-1], c)
        self.assertEqual(topo_sorted[-1].get_label(), '+')
        

    def test_sub(self):
        a = Value(3, "a")
        b = Value(2, "b")

        c = a - b
        self.assertEqual(c.data, 1)

        c.backward()

        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, -1.0)

    def test_self_interaction(self):
        a = Value(3.0)
        
        c = a * a
        c.backward()

        self.assertEqual(a.grad, 6)




