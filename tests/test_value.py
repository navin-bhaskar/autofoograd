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
