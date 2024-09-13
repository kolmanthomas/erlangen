import numpy as np
import matplotlib.pyplot as plt

from conics import *
from mobius import *
from draw import *
from ex import *
from util import *


#plt.ion()
fig, ax = plt.subplots()
ax.set_aspect('equal')

def show():
    plt.show()

"""
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
        

"""

