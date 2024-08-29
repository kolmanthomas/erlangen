import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

import matplotlib.animation as animation

class Geometry:
    def generate(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class Circle(Geometry):
    def __init__(self, center: float, radius: float):
        self.c = center
        self.r = radius

    def generate(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 2 * np.pi, step)
        np_x, np_y = np.vectorize(lambda t : (self.r * np.cos(t), self.r * np.sin(t)))(t)
        print(np_x)
        print(len(np_y))
        return (np_x, np_y)

class Line(Geometry):
    def __init__(self, slope: float, offset: float):
        self.m = slope
        self.b = offset

    def generate(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        np_x = np.linspace(-5, 5, step)
        np_y = np.vectorize(lambda x : self.m * x + self.b)(np_x)
        return (np_x, np_y)

class Ellipse(Geometry):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def generate(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        np_t = np.linspace(-np.pi, np.pi, step)
        np_x, np_y = np.vectorize(lambda t : (self.a * np.cos(t), self.b * np.sin(t)))(np_t)
        return (np_x, np_y)

    def plot_directrix(self, step: int):
        e = -(self.b**2 - self.a**2)/self.a**2
        directrix = self.a/e
        return (np.full((step), directrix), np.linspace(-3, 3, step))

class Parabola(Geometry):
    def __init__(self, a: float):
        self.a = a

    def generate(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        np_x = np.linspace(-3, 3, step)
        np_y = np.vectorize(lambda x : (self.a * x**2))(np_x)
        return (np_x, np_y)

def construct_parabola_from_focus_and_directrix(a: float):
    """
        Constructs a parabola

        e = 1

    """
    x, y = Parabola(a).generate(100)
    directrix = Line(0, -a)
    # Distance between 
    p = a * 3**2
    plt.plot(x, y, color='g')
    plt.plot(3, p, color='r')
    


def present():
    fig = plt.figure()
    ax = plt.axes(xlim=(-3, 3), ylim=(-5, 5))

    np_x, np_y = Parabola(1).generate(100)

    line, = ax.plot([], [], lw=2, color='r')

    #plt.gca().set_aspect('equal')
    #plt.show()

    def update(frame): 
        line.set_data(np_x[:frame], np_y[:frame])
        return line,

    ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=30)
    plt.show()
        


