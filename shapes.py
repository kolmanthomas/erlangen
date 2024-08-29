import numpy as np
import shapely as shp
import cmath

class Circle:
    def __init__(self, center: np.ndarray, radius: float):
        self.c = center
        self.r = radius
        self.geometry = shp.geometry.Point(center).buffer(radius)

    def three_points(self) -> np.ndarray:
        """
            Returns three points on the circle.
        """
        north = self.c + complex(0, self.r)
        west = self.c + complex(self.r, 0)
        east = self.c + complex(-self.r, 0)
        return np.array([
            [north.real, north.imag], 
            [west.real, west.imag], 
            [east.real, east.imag]])

class Line:
    """
        Representation of a line.

        The line is defined by passing between two points.
    """
    def __init__(self, point1: shp.Point, point2: shp.Point):
        self.point1 = point1
        self.point2 = point2

class GeneralizedCircle:
    def __init__(self, center: complex, radius: float):
        self.c = center
        self.r = radius
        self.geometry = shp.geometry.Point(np.array([center.real, center.imag])).buffer(radius)

    @classmethod
    def from_three_points(cls, z1: complex, z2: complex, z3: complex):
        w = (z3 - z1)/(z2 - z1)
        c = (z2 - z1) * (w - abs(w)**2)/(2j * w.imag) + z1
        r = abs(z1 - c)
        return cls(c, r)


    def get_three_points(self) -> np.ndarray:
        """
            Returns three points on the circle.
        """
        north = self.c + complex(0, self.r)
        west = self.c + complex(self.r, 0)
        east = self.c + complex(-self.r, 0)
        return np.array([north, west, east])
            #[north.real, north.imag], 
            #[west.real, west.imag], 
            #[east.real, east.imag]])

