import pytest

from foograd import Value
from foograd.gradcheck import gradcheck, numerical_grads


@pytest.mark.parametrize(
    "f, xs, expected_grads",
    [
        (lambda xs: xs[0].exp(), [0.5], [1.6487212707001282]),
        (lambda xs: xs[0] * xs[1], [2, -3], [-3, 2]),
        (lambda xs: xs[0] * xs[0], [2, 3], [4, 0]),
        (
            lambda xs: (xs[0] * xs[1] + xs[2]).tanh(),
            [0.5, -1.0, 0.3],
            [-0.9610429829661166, 0.4805214914830583, 0.9610429829661166],
        ),
    ],
    ids=["exp", "mul", "square_unused_input", "tanh"],
)
def test_numerical_grads(f, xs, expected_grads):
    actual_grads = numerical_grads(f, xs)
    assert actual_grads == pytest.approx(expected_grads, abs=1e-6)


def test_numerical_grads_does_not_mutate_inputs():
    xs = [2.0, -3.0]
    numerical_grads(lambda v: v[0] * v[1], xs)
    assert xs == [2.0, -3.0]


# Inputs are chosen away from kinks (relu at 0) and outside invalid
# domains (log / fractional powers need x > 0, division needs x != 0).
@pytest.mark.parametrize(
    "f, xs",
    [
        (lambda v: v[0] + v[1], [2.0, -3.0]),
        (lambda v: v[0] - v[1], [2.0, -3.0]),
        (lambda v: v[0] * v[1], [2.0, -3.0]),
        (lambda v: v[0] / v[1], [2.0, -3.0]),
        (lambda v: v[0] + 3, [2.0]),
        (lambda v: 3 + v[0], [2.0]),
        (lambda v: v[0] - 3, [2.0]),
        (lambda v: 3 - v[0], [2.0]),
        (lambda v: 3 * v[0], [2.0]),
        (lambda v: v[0] / 4, [2.0]),
        (lambda v: 2 / v[0], [2.0]),
        (lambda v: -v[0], [2.0]),
        (lambda v: v[0] ** 3, [1.5]),
        (lambda v: v[0] ** -2, [1.5]),
        (lambda v: v[0] ** 0.5, [1.5]),
        (lambda v: v[0].exp(), [0.5]),
        (lambda v: v[0].log(), [1.5]),
        (lambda v: v[0].relu(), [1.5]),
        (lambda v: v[0].relu(), [-1.5]),
        (lambda v: v[0].tanh(), [0.5]),
    ],
    ids=[
        "add",
        "sub",
        "mul",
        "div",
        "add_number",
        "radd",
        "sub_number",
        "rsub",
        "rmul",
        "div_number",
        "rdiv",
        "neg",
        "pow_int",
        "pow_negative",
        "pow_fractional",
        "exp",
        "log",
        "relu_positive",
        "relu_negative",
        "tanh",
    ],
)
def test_ops_pass_gradcheck(f, xs):
    assert gradcheck(f, xs)


def test_neuron_passes_gradcheck():
    # tanh(w1*x1 + w2*x2 + b), inputs ordered [x1, x2, w1, w2, b]
    def neuron(v):
        x1, x2, w1, w2, b = v
        return (w1 * x1 + w2 * x2 + b).tanh()

    assert gradcheck(neuron, [2.0, 0.0, -3.0, 1.0, 6.8813735870195432])


def test_reused_node_passes_gradcheck():
    # x appears three times; backward must accumulate with +=
    assert gradcheck(lambda v: v[0] * v[0] + v[0], [3.0])


def test_shared_subexpression_passes_gradcheck():
    # a is used by two downstream branches that rejoin
    def f(v):
        a = v[0] * v[1]
        return a.tanh() + a * v[0]

    assert gradcheck(f, [0.5, -0.8])


def test_large_gradient_passes_gradcheck():
    # grad ~ 3.3 million; absolute error is ~1e-3, so this needs rtol
    assert gradcheck(lambda v: v[0].exp(), [15.0])


def test_gradcheck_catches_broken_backward():
    def broken_square(v):
        x = v[0]
        out = x * x

        def wrong_backward():
            x.grad += 3 * x.data * out.grad  # correct rule is 2 * x.data

        out._backward = wrong_backward
        return out

    assert not gradcheck(broken_square, [2.0])


def test_gradcheck_catches_missing_backward():
    # Forgetting to set _backward leaves the input grad at 0
    def no_backward(v):
        return Value(v[0].data * 2, (v[0],), "*2")

    assert not gradcheck(no_backward, [2.0])
