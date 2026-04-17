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
