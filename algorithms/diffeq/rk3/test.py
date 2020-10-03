from math import pow, fabs
from numpy import array, exp

from algorithms.diffeq.rk3.algo import rk3

import pytest


def get_test_error(step: float) -> float:
    return pow(step, 1.0 / 3.0)


def dx_dy_first_order(x: float, y: array) -> array:
    """
    y' = x + y ^ 2
    """
    return array([x + y[0] ** 2])


def dx_dy_second_order(x: float, y: array) -> array:
    """
    y'' + 2y' + 2y = x * exp(x)
    """
    return array([y[1], -2.0 * y[1] - 2.0 * y[0] + x * exp(x)])  # z = y' => z', -2z - 2y + x * exp(x)


@pytest.fixture(scope="function", params=[
    (-2.0, 0.0, 0.5, -0.66),
    (1.0, 2.0, 0.1, 3.56)])
def first_order_equation_param(request):
    return request.param


@pytest.fixture(scope="function", params=[
    (-2.0, 0.0, -0.28994),
    (1.0, 4.0, 34.93017),
    (-2.0, 2.0, 1.78479)])
def second_order_equation_param(request):
    return request.param


@pytest.mark.parametrize("step", [0.1, 0.01, 0.0001])
def test_should_solve_second_order_equation_for_different_step_size(step):
    # arrange
    test_x0 = 0.0
    test_x1 = 1.5
    test_y0 = array([0.0, 0.0])  # y(0) = 0, y'(0) = 0
    expected_y = 0.65667

    # act
    actual_y = rk3(dx_dy_second_order, step, test_x0, test_x1, test_y0)

    # assert
    assert fabs(expected_y - actual_y) < get_test_error(step)


def test_should_solve_second_order_equation_for_different_arguments(second_order_equation_param):
    # arrange
    (x0, x1, y) = second_order_equation_param
    test_step = 0.001
    y0 = [0.1, -1.2]

    # act
    actual_y = rk3(dx_dy_second_order, test_step, x0, x1, y0)

    # assert
    assert fabs(y - actual_y) < get_test_error(test_step)


@pytest.mark.parametrize("step", [0.1, 0.01, 0.0001])
def test_should_solve_first_order_equation_for_different_step_size(step):
    # arrange
    test_x0 = 0.0
    test_x1 = 1.0
    test_y0 = array([0.5])  # y(0) = 0, y'(0) = 0
    expected_y = 2.128

    # act
    actual_y = rk3(dx_dy_first_order, step, test_x0, test_x1, test_y0)

    # assert
    assert fabs(expected_y - actual_y) < get_test_error(step)


def test_should_solve_first_order_equation_for_different_arguments(first_order_equation_param):
    # arrange
    (x0, x1, y0, y) = first_order_equation_param
    test_step = 0.001

    # act
    actual_y = rk3(dx_dy_first_order, test_step, x0, x1, array([y0]))

    # assert
    assert fabs(y - actual_y) < get_test_error(test_step)
