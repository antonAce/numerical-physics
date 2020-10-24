from numpy import sin, cos, power
from p5 import *


class Rod:
    def __init__(self,
                 mass: float,
                 radian: float,
                 velocity: float,
                 acceleration: float):
        self.mass = mass
        self.radian = radian
        self.velocity = velocity
        self.acceleration = acceleration


def calculate_acceleration(
        upper_rod: Rod,
        lower_rod: Rod,
        g: float):
    """
    Differential equation that calculates acceleration (second derivative of
    radians function) value for both upper and lower rod of the pendulum.

    :param upper_rod: Mass, radians, velocity and acceleration of the upper pendulum
    :param lower_rod: Mass, radians, velocity and acceleration of the lower pendulum
    :param g: gravity constant
    :return: acceleration diffs for both upper and lower rod of the pendulum
    """
    return (
        (-g * (2 * upper_rod.mass + lower_rod.mass) * sin(upper_rod.radian) - lower_rod.mass * g *
         sin(upper_rod.radian - 2 * lower_rod.radian) -
         2 * sin(upper_rod.radian - lower_rod.radian) * lower_rod.mass
         * (power(lower_rod.velocity, 2) * lower_rod.radian + power(upper_rod.velocity, 2) * upper_rod.radian *
            cos(upper_rod.radian - lower_rod.radian))) /
        (upper_rod.radian * (2 * upper_rod.mass + lower_rod.mass - lower_rod.mass * cos(
            2 * upper_rod.radian - 2 * lower_rod.radian))),
        (2 * sin(upper_rod.radian - lower_rod.radian) * (power(upper_rod.velocity, 2) * upper_rod.radian *
                                                           (upper_rod.mass + lower_rod.mass) + g * (
                                                                   upper_rod.mass + lower_rod.mass) * cos(
                    upper_rod.radian) + power(lower_rod.velocity, 2) * lower_rod.radian * lower_rod.mass * cos(
                    upper_rod.radian - lower_rod.radian))) / (lower_rod.radian *
                                                                (2 * upper_rod.mass + lower_rod.mass - lower_rod.mass *
                                                                 cos(2 * upper_rod.radian - 2 * lower_rod.radian)))
    )


def setup():
    title('Double pendulum')
    size(600, 600)


def draw():
    background(255)
    stroke(0)
    stroke_weight(2)
    translate(300, 50)


run()
