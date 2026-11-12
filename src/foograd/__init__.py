"""foograd: a tiny autograd engine built from scratch."""

__version__ = "0.1.0"

from .gradcheck import gradcheck as gradcheck
from .gradcheck import numerical_grads as numerical_grads
from .value import Value as Value
