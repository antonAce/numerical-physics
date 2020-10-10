from typing import Callable
from numpy import linspace, array, copy
from math import ceil


def rk4(dxdy: Callable, h: float, x0: float, x1: float, y: array) -> float:
    """
    Solves numerically differential equation **dxdy** using
    Runge Kutta 4th order method and returns y(**x1**)

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
        k1i = h * dxdy(xi, yi)
        k2i = h * dxdy(xi + h * 0.5, yi + h * 0.5 * k1i)
        k3i = h * dxdy(xi + h * 0.5, yi + h * 0.5 * k2i)
        k4i = h * dxdy(xi + h, yi + h * k3i)

        yi += (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0

    return yi[0]
