import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
import cmath
import geometry

"""
    Class that represents fundamental geometric shapes representable by algebraic equations.
"""

class Point:
    """
        An n-dimensional point.
    """

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

class Line(geometry.Geometry):
    """
        Representation of a line in arbitrary dimensions.

        \\[ f(t) = t(x - y) + y \\]

        The line is defined by passing between two points.
    """
    def __init__(self, a: np.ndarray, b: np.ndarray):
        if a.ndim != 1:
            raise ValueError("Points a and b must be one-dimensional arrays.")
        elif a.shape != b.shape:
            raise ValueError("Points a and b must have the same shape.")

        self.a = a
        self.b = b
        self.f = lambda t : t * (a - b) + b
        super().__init__(a.shape[0])



    @classmethod
    def point_slope_form(cls, p: np.ndarray, m: float):
        """
            Creates a line from a point and a slope.

            \\[ y = mx + b \\]
        """
        b = p[1] - m*p[0]
        print(type(b))
        
        x0 = p[0] + 1
        print(type(x0))
        y0 = m*x0 + b
        print(type(y0))

        p0 = np.array([x0, y0])

        return cls(p, p0)


"""
class ExtendedComplex(complex):
    def __init__(self, z: complex):
        self.z = z

    def __add__(self, other: 'ExtendedComplex') -> 'ExtendedComplex':
        return ExtendedComplex(self.z + other.z)

    def __sub__(self, other: 'ExtendedComplex') -> 'ExtendedComplex':
        return ExtendedComplex(self.z - other.z)

    def __mul__(self, other: 'ExtendedComplex') -> 'ExtendedComplex':
        return ExtendedComplex(self.z * other.z)

    def __truediv__(self, other: 'ExtendedComplex') -> 'ExtendedComplex':
        return ExtendedComplex(self.z / other.z)

    def conjugate(self) -> 'ExtendedComplex':
        return ExtendedComplex(self.z.conjugate())

    def __repr__(self):
        return str(self.z)

    def __str__(self):
        return str(self.z)

    def __eq__(self, other: 'ExtendedComplex'):
        return self.z == other.z

    def __ne__(self, other: 'ExtendedComplex'):
        return self.z != other.z

    def __lt__(self, other: 'ExtendedComplex'):
        return self.z < other.z

    def __le__(self, other: 'ExtendedComplex'):
        return self.z <= other.z

    def __gt__(self, other: 'ExtendedComplex'):
        return self.z > other.z

    def __ge__(self, other: 'ExtendedComplex'):
        return self.z >= other.z

    def __abs__(self):
        return abs(self.z)

    def __pow__(self, other: 'ExtendedComplex'):
        return ExtendedComplex(self.z**other.z)

    def __radd__(self, other: 'ExtendedComplex'):
        return ExtendedComplex(other.z + self.z)

    def __rsub__(self, other: 'ExtendedComplex'):
        return ExtendedComplex(other.z - self.z)

    def __rmul__(self, other: 'ExtendedComplex'):
        return ExtendedComplex(other.z * self.z)

    def __rtruediv__(self, other: 'ExtendedComplex'):
        return ExtendedComplex(other.z / self.z)

    def __rpow__(self, other: 'ExtendedComplex'):
        return ExtendedComplex(other.z**self.z)

    def __neg__(self):
        return ExtendedComplex(-self.z)

    def __int__(self):
        return int(self.z)

    def __float__(self):
        return
"""

class GeneralizedCircle:
    def __init__(self, center: complex, radius: float):
        self.c = center
        self.r = radius
        self.geometry = shp.geometry.Point(np.array([center.real, center.imag])).buffer(radius)

    @classmethod
    def from_three_points(cls, z1: complex, z2: complex, z3: complex):
        """

        """
        if (z1 == z2) or (z2 == z3) or (z1 == z3):
            raise ValueError("Points are not distinct.")

        # If three points are collinear, then radius is infinite
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

    """
    def inversion(self, circle: 'GeneralizedCircle'):
        a = self.c
        b = self.r**2 - self.c * self.c.conjugate()
        c = 1
        d = -self.c.conjugate()
        m = mobius.MobiusTransform(a, b, c, d)

        points = circle.get_three_points()
        inv_points = np.array([m(p.conjugate()) for p in points])
        inverted_circle = GeneralizedCircle.from_three_points(*inv_points)
        return inverted_circle
    """



