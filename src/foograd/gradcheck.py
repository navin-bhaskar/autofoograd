from collections.abc import Callable

from foograd.value import Value


def numerical_grads(
    f: Callable[[list[Value]], Value], xs: list[float], eps: float = 1e-6
) -> list[float]:
    """Compute numerical gradients of f at xs using finite differences.
    df/dx = approx((f(x + eps) - f(x - eps)) / (2 * eps))

    params:
        f: function that takes a list of Values and returns a Value
        xs: list of floats to compute gradients at
        eps: small value to use for finite differences
    """
    grads: list[float] = []

    for i in range(len(xs)):
        # nudge the i-th input by eps in both directions
        plus = list(xs)
        plus[i] += eps

        minus = list(xs)
        minus[i] -= eps

        # Evaluate the function at the perturbed points
        f_plus = f([Value(x) for x in plus])
        f_minus = f([Value(x) for x in minus])

        # Compute the numerical gradient using central difference
        grad = (f_plus.data - f_minus.data) / (2 * eps)
        grads.append(grad)

    return grads


def gradcheck(
    f: Callable[[list[Value]], Value],
    xs: list[float],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Check the gradients of f at xs using numerical gradients."""
    # Compute analytical gradients using backpropagation
    inputs = [Value(x) for x in xs]
    output = f(inputs)
    output.backward()
    analytical_grads = [x.grad for x in inputs]

    # Compute numerical gradients
    numerical_grads_list = numerical_grads(f, xs, eps)

    # Compare the two sets of gradients
    return all(
        abs(analytical - numerical) <= atol + rtol * abs(numerical)
        for analytical, numerical in zip(analytical_grads, numerical_grads_list, strict=True)
    )


