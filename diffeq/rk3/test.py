from math import pow
from numpy import array, exp

from diffeq.rk3.algo import rk3

import unittest


def get_test_error(step: float) -> float:
    return pow(step, 1.0 / 4.0)


def dxdy(x: float, y: array) -> array:
    return array([y[1], -2.0 * y[1] - 2.0 * y[0] + x * exp(x)])


class RungeKutta3(unittest.TestCase):
    def positive_case_second_order_equation(self):
        # arrange
        test_step = 0.1
        test_x0 = 0.0
        test_x1 = 1.5
        test_y0 = array([0.0, 0.0])
        expected_y = 0.65667

        # act
        actual_y = rk3(dxdy, test_step, test_x0, test_x1, test_y0)

        self.assertAlmostEquals(expected_y, actual_y, delta=get_test_error(test_step))


if __name__ == '__main__':
    unittest.main()
