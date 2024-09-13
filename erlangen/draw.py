"""
    Drawing module
"""

import shapely as shp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import PathPatch 
from matplotlib.path import Path 
from matplotlib.collections import PatchCollection
from math import floor
import threading
from typing import List

import conics 

"""
    Handles 2D and 3D plotting.

"""
class Interval():
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),

            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

"""
def plot_circle(ax, circle: Circle, color: str = 'r'):
    ax.plot(circle.geometry.exterior.xy[0], circle.geometry.exterior.xy[1], color)
"""

def plot_point(ax, x: np.ndarray, color: str = 'ro'):
    ax.plot(x[0], x[1], color)

def plot_line(ax, line: conics.Line, color):
    ax.axline(xy1=line.a, xy2=line.b, color=color)

def plot_line_segment(ax, line: conics.Line):
    ax.plot(np.array([line.a[0], line.b[0]]), np.array([line.a[1], line.b[1]]), 'r')

def plot_conic(ax, conic: conics.Conic, color='r'):
    t = np.linspace(-np.pi, np.pi, 100)
    x, y = conic.x(t), conic.y(t)
    ax.plot(x, y, color)


def draw(ax, shape, color='r'):
    if isinstance(shape, conics.Line):
        plot_line(ax, shape, color)
    elif isinstance(shape, conics.Conic):
        plot_conic(ax, shape, color)
    else:
        raise ValueError("Shape not recognized.")


def animate(fig, ax, shapes, color='r'):
    th = threading.Thread(target=present, args=(fig, ax, shapes, color)) 



def present(fig, ax, shapes: List[List], color='r'):
    """
        Animates.

        Args:
            shapes: List of a list of shapes to emulate. Each inner list should contain shapes that are to be rendered at the same time.

        Each image should display 60 frames
    """
    #fig = plt.figure()
    ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))

    # We render at 30 fps
    interval = 30
    frames_per_interval = 30
    pixels_per_frame = 60
    frames = 60 * len(shapes)

    t = np.linspace(-np.pi, np.pi, frames_per_interval * pixels_per_frame)

    x_list, y_list, artists = [], [], []
    for shape_group in shapes:
        x_list.append([shape.x(t) for shape in shape_group])
        y_list.append([shape.y(t) for shape in shape_group])
        artists.append([ax.plot([], [], color='r')[0] for _ in shape_group])

    print(type(artists))
    print(artists)
    """
    artists = []
    for i in range(len(x_list)):
        artist, = ax.plot([], y_list[i], color='r')
        artists.append(artist)

    [[ax.plot(x_list[i][j], y_list[i][j], 'r') for j in range(len(x_list[i]))] for i in range(len(x_list))]

    print(type(artists))
    print(type(artists[0]))

    for artist in artists:
        artist.set_data([], [])

    """
    def update(frame): 
        i = floor(frame/60)
        frame_mod = frame % 60
        for j in range(len(artists[i])):
            artists[i][j].set_data(x_list[i][j][:(frame_mod + 1)*frames_per_interval], y_list[i][j][:(frame_mod + 1)*frames_per_interval])

        print(*artists[i])
        return *artists[i],

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, repeat=False)

    ax.set_aspect('equal')
    plt.show()
    return ani
    
 
