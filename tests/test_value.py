import pytest

from foograd import Value


def test_stores_value():
    val = Value(2.0)
    assert val.data == 2.0
    assert isinstance(val.data, float)


def test_grad_starts_at_zero():
    assert Value(2.0).grad == 0.0


def test_add():
    assert (Value(2.0) + Value(3.0)).data == 5.0


def test_mul():
    assert (Value(2.0) * Value(3.0)).data == 6.0


def test_add_records_graph():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c._children == (a, b)
    assert c._op == "+"


def test_add_number():
    a = Value(2.0)
    b = 3.0
    c = a + b
    assert c.data == 5.0


def test_mul_number():
    a = Value(2.0)
    b = 3.0
    c = a * b
    assert c.data == 6.0


def test_add_rejects_string():
    a = Value(2.0)
    b = "3.0"
    with pytest.raises(TypeError):
        _ = a + b


def test_reused_input_kept_twice():
    a = Value(2.0)
    b = a + a
    assert b._children == (a, a)
    assert b._op == "+"


def test_repr():
    a = Value(2.0)
    assert repr(a) == "Value(data=2.0, grad=0.0)"


def test_backward_seeds_output():
    y = Value(5.0)
    y.backward()
    assert y.grad == 1


def test_backward_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_backward_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_backward_chain():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    y = (a + b) * c
    y.backward()
    # a -> 2, b->3 and c->4
    # y = a*c + b*c
    # d(y)/da = c = 4 (no change in b*c as not term of a in it)
    # d(y)/db = c = 4
    # d(y)/dc = a + b = 3 + 2 = 5
    assert a.grad == 4.0
    assert b.grad == 4.0
    assert c.grad == 5.0


def test_backward_reused_add():
    a = Value(2.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0  # 1 from each addition


def test_backward_reused_mul():
    a = Value(3.0)
    b = a * a
    b.backward()
    assert a.grad == 6.0  # 3 from each multiplication


def test_backward_shared_node():
    a = Value(1.0)
    s = a * 3
    m = s * 2
    y = s + m  # y = (a * 3) + ((a * 3) * 2) = 3a + 6a = 9a
    y.backward()
    assert a.grad == 9.0


def test_backward_number():
    a = Value(3.0)
    y = a * 2 + 1
    y.backward()
    assert a.grad == 2.0


def test_subtract():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_subtract_same_object():
    a = Value(5.0)
    b = a - a
    assert b.data == 0.0
    b.backward()
    assert a.grad == 0.0  # The gradients cancel each other out


def test_subtract_number():
    a = Value(5.0)
    b = 2.0
    c = a - b
    assert c.data == 3.0
    c.backward()
    assert c.grad == 1.0
    assert a.grad == 1.0


def test_subtract_string():
    a = Value(5.0)
    b = "2.0"
    with pytest.raises(TypeError):
        _ = a - b


def test_radd():
    a = Value(2.0)
    b = 3.0
    c = b + a  # This will call a.__radd__(b)
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0


def test_rsub():
    a = Value(5.0)
    b = 2.0
    c = b - a  # This will call a.__rsub__(b)
    assert c.data == -3.0
    c.backward()
    assert a.grad == -1.0


def test_rmul():
    a = Value(5.0)
    c = 2 * a
    assert c.data == 10.0
    c.backward()
    assert a.grad == 2.0
    assert c.grad == 1.0  
