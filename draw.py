"""
    Drawing module
"""

import shapely as shp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch 
from matplotlib.path import Path 
from matplotlib.collections import PatchCollection

from shapes import Circle, Line, GeneralizedCircle

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),

            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def plot_circle(ax, circle: GeneralizedCircle, **kwargs):
    ax.plot(circle.geometry.exterior.xy[0], circle.geometry.exterior.xy[1])

def plot_line(ax, line: Line, **kwargs):
    ax.axline(xy1=line.point1, xy2=line.point2)

