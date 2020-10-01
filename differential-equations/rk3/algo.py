from typing import Callable
from numpy import linspace, array, copy
from math import ceil


def rk3(dxdy: Callable, h: float, x0: float, x1: float, y: array) -> float:
    """
    Solves numerically differential equation **dxdy** and returns y(**x1**)

    :param dxdy: system of differential equations y', y'', ...
    :param h: step size
    :param x0: start value of argument x
    :param x1: target value of argument x
    :param y: value of function y(x0)
    :return: value of function y(x1)
    """

    steps = ceil((x1 - x0) / h) + 1
    x_space = linspace(x0, x1, steps)
    yi = copy(y)

    for xi in x_space[1:]:
        k1i = h * dxdy(yi, xi)
        k2i = h * dxdy(yi + h * 0.5 * k1i, xi + h * 0.5)
        k3i = h * dxdy(yi - h * k1i + h * 2.0 * k2i, xi + h * 0.5)

        yi += (k1i + 4.0 * k2i + k3i) / 6.0

    return yi[0]
