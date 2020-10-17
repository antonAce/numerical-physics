from numpy import sin, cos, power
from PyQt5 import QtGui, QtWidgets


class Rod:
    def __init__(self,
                 mass: float,
                 radians: float,
                 velocity: float,
                 acceleration: float):
        self.mass = mass
        self.radians = radians
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
        (-g * (2 * upper_rod.mass + lower_rod.mass) * sin(upper_rod.radians) - lower_rod.mass * g *
         sin(upper_rod.radians - 2 * lower_rod.radians) -
         2 * sin(upper_rod.radians - lower_rod.radians) * lower_rod.mass
         * (power(lower_rod.velocity, 2) * lower_rod.radians + power(upper_rod.velocity, 2) * upper_rod.radians *
            cos(upper_rod.radians - lower_rod.radians))) /
        (upper_rod.radians * (2 * upper_rod.mass + lower_rod.mass - lower_rod.mass * cos(
            2 * upper_rod.radians - 2 * lower_rod.radians))),
        (2 * sin(upper_rod.radians - lower_rod.radians) * (power(upper_rod.velocity, 2) * upper_rod.radians *
                                                           (upper_rod.mass + lower_rod.mass) + g * (
                                                                   upper_rod.mass + lower_rod.mass) * cos(
                    upper_rod.radians) + power(lower_rod.velocity, 2) * lower_rod.radians * lower_rod.mass * cos(
                    upper_rod.radians - lower_rod.radians))) / (lower_rod.radians *
                                                                (2 * upper_rod.mass + lower_rod.mass - lower_rod.mass *
                                                                 cos(2 * upper_rod.radians - 2 * lower_rod.radians)))
    )


class DoublePendulum(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(400, 300)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.draw_something()

    def draw_something(self):
        painter = QtGui.QPainter(self.label.pixmap())
        painter.drawLine(10, 10, 300, 200)
        painter.end()
