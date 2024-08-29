from shapely import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch 
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import erlangen as erl
import mobius 
import matplotlib.animation as animation

from draw import plot_polygon, plot_circle, plot_line
import shapes

fig, ax = plt.subplots()

circle_1 = shapes.GeneralizedCircle(0 + 0j, 1)
circle_2 = shapes.GeneralizedCircle(0 + 3j, 1)
plot_circle(ax, circle_1)
plot_circle(ax, circle_2)

#plot_line(ax, shapes.Line(np.array([0, 0]), np.array([1, 1])))

def inversion(gen_circle: shapes.GeneralizedCircle, circle: shapes.GeneralizedCircle):
    """
        Inverts a circle with respect to a mobius transformation.
    """
    a = gen_circle.c
    b = gen_circle.r**2 - gen_circle.c * gen_circle.c.conjugate()
    c = 1
    d = -gen_circle.c.conjugate()
    m = mobius.MobiusTransform(a, b, c, d)

    points = circle.get_three_points()
    inv_points = np.array([m(p) for p in points])
    print(len(inv_points))
    return inv_points

points = inversion(circle_1, circle_2)
plot_circle(ax, shapes.GeneralizedCircle.from_three_points(*points))
# Invert line
plt.gca().set_aspect('equal')
plt.show()

'''
a = 1
x, y = erl.Parabola(a).generate(100)
directrix_x, directrix_y = erl.Line(0, -a).generate(100)
# Distance between 
p = a * 3**2
plt.plot(x, y, color='g')
plt.plot(3, p, 'ro')
plt.plot([3, 3], [p, -a], 'r')
plt.plot([0, 3], [a, p], 'r')
plt.plot(directrix_x, directrix_y)

fig = plt.figure()
ax = plt.axes(xlim=(-3, 3), ylim=(-5, 5))

np_x, np_y = erl.arabola(1).generate(100)

line, = ax.plot([], [], lw=2, color='r')

#plt.gca().set_aspect('equal')
#plt.show()

def update(frame): 
    line.set_data(np_x[:frame], np_y[:frame])
    return line,

ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=30)
plt.show()
'''  

"""
verts = [(0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.),]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,]

path = Path(verts, codes)

#patch = Point(0.0, 0.0).buffer(10.0, 0)
fig, ax = plt.subplots()
fig

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),

            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

polygon = Polygon(((0, 0), (10, 0), (10, 10), (0, 10)), 
                  (((1, 3), (5, 3), (5, 1), (1, 1)), ((9, 9), (9, 8), (8, 8), (8, 9))))
plot_polygon(ax, polygon, facecolor='lightblue', edgecolor='red')
plt.show()
"""











