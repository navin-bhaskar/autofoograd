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


def test_pow():
    a = Value(3.0)
    c = a**2
    assert c.data == 9.0
    c.backward()
    assert a.grad == 6.0  # d(a^2)/da = 2*a = 2*3 = 6


def test_pow_float():
    a = Value(4.0)
    c = a**0.5  # Square root
    assert c.data == 2.0
    c.backward()
    assert a.grad == 0.25  # d(a^0.5)/da = 0.5 * a^(-0.5) = 0.5 * (1/2) = 0.25


def test_pow_negative_exponent():
    a = Value(2.0)
    c = a**-1  # Reciprocal
    assert c.data == 0.5
    c.backward()
    assert a.grad == -0.25  # d(a^-1)/da = -1 * a^(-2) = -1 * (1/4) = -0.25


def test_pow_zero_exponent():
    a = Value(5.0)
    c = a**0  # Any number to the power of 0 is 1
    assert c.data == 1.0
    c.backward()
    assert a.grad == 0.0  # d(a^0)/da = 0


def test_pow_one_exponent():
    a = Value(5.0)
    c = a**1  # Any number to the power of 1 is itself
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0  # d(a^1)/da = 1


def test_pow_fractional_exponent():
    a = Value(8.0)
    c = a ** (1 / 3)  # Cube root
    assert c.data == pytest.approx(2.0)
    c.backward()
    assert a.grad == pytest.approx(
        1 / 12
    )  # d(a^(1/3))/da = (1/3) * a^(-2/3) = (1/3) * (1/4) = 1/12


def test_pow_with_value():
    a = Value(2.0)
    b = Value(3.0)

    with pytest.raises(TypeError):
        _ = a**b  # This should raise an error since exponent must be float or int


def test_true_division():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    assert c.data == 3.0
    c.backward()
    assert c._op == "/"
    assert a.grad == pytest.approx(1 / 2)  # d(a/b)/da = 1/b = 1/2
    assert b.grad == pytest.approx(-3 / 2)  # d(a/b)/db = -a/b^2 = -6/4 = -3/2


def test_rtrue_division():
    a = Value(2.0)
    b = 6.0
    c = b / a  # This will call a.__rtruediv__(b)
    assert c.data == 3.0
    c.backward()
    assert a.grad == pytest.approx(-3 / 2)  # d(b/a)/da = -b/a^2 = -6/4 = -3/2


def test_true_division_with_number():
    a = Value(6.0)
    b = 2.0
    c = a / b
    assert c.data == 3.0
    c.backward()
    assert a.grad == pytest.approx(1 / 2)  # d(a/b)/da = 1/b = 1/2


def test_true_division_with_string():
    a = Value(6.0)
    b = "2.0"
    with pytest.raises(TypeError):
        _ = a / b


def test_exp():
    a = Value(1.0)
    c = a.exp()
    assert c.data == pytest.approx(2.718281828459045)  # e^1
    c.backward()
    assert a.grad == pytest.approx(2.718281828459045)  # d(e^a)/da = e^a = e^1
    assert c._op == "exp"


def test_exp_of_zero():
    a = Value(0.0)
    c = a.exp()
    assert c.data == pytest.approx(1.0)  # e^0 = 1
    c.backward()
    assert a.grad == pytest.approx(1.0)  # d(e^a)/da = e^a = e^0 = 1


def test_log():
    a = Value(5)
    c = a.log()
    assert c.data == pytest.approx(1.60943791243)
    c.backward()
    assert a.grad == pytest.approx(0.2)  # d(log(a))/da = 1/a = 1/5 = 0.2


def test_log_of_one():
    a = Value(1.0)
    c = a.log()
    assert c.data == pytest.approx(0.0)  # log(1) = 0
    c.backward()
    assert a.grad == pytest.approx(1.0)  # d(log(a))/da = 1/a = 1/1 = 1


def test_exp_log_chain():
    a = Value(2.0)
    c = a.exp().log()  # log(exp(a)) = a
    assert c.data == pytest.approx(2.0)
    c.backward()
    assert a.grad == pytest.approx(1.0)  # d(log(exp(a)))/da = 1


@pytest.mark.parametrize("x", [0.0, -1.0])
def test_log_non_positive_raises(x):
    with pytest.raises(ValueError):
        Value(x).log()


def test_relu_positive():
    a = Value(3.0)
    c = a.relu()
    assert c.data == 3.0
    c.backward()
    assert a.grad == 1.0  # d(relu(a))/da = 1 for a > 0


def test_relu_negative():
    a = Value(-2.0)
    c = a.relu()
    assert c.data == 0.0
    c.backward()
    assert c._op == "relu"
    assert a.grad == 0.0  # d(relu(a))/da = 0 for a < 0


def test_relu_zero():
    a = Value(0.0)
    c = a.relu()
    assert c.data == 0.0
    c.backward()
    assert a.grad == 0.0  # d(relu(a))/da = 0 for a = 0


def test_tanh_at_zero():
    a = Value(0.0)
    c = a.tanh()
    assert c.data == 0.0
    assert c._op == "tanh"
    c.backward()
    assert a.grad == 1.0  # d(tanh(a))/da = 1 for a = 0


def test_tanh_at_one():
    a = Value(1.0)
    c = a.tanh()
    assert c.data == pytest.approx(0.7615941559557649)  # tanh(1)
    c.backward()
    assert a.grad == pytest.approx(0.41997434161402614)  # d(tanh(a))/da = 1 - tanh^2(a)


def test_relu_chain():
    a = Value(0.25)

    b = (a * 2 - 1).relu()
    b.backward()
    assert a.grad == 0.0
