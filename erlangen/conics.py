import numpy as np
import matplotlib.pyplot as plt
import shapely as shp
import numpy.typing as npt

class Conic():
    """
    General class for the conic sections.
    Implements conics of the form:

    \\[ ax^2 + bxy + cy^2 + dx + ey + f = 0 \\]
   
    The class



    """
    def __init__(self, c1, c2, c3, c4, c5, c6, pos): 
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.pos = pos

    def domain(self):
        raise NotImplementedError

    def sample_points(self, min, max, num):
        t = np.linspace(min, max, num)
        x = np.vectorize(self.x)(t)
        y = np.vectorize(self.y)(t)
        return np.array([x, y])

    def x(self, t):
        raise NotImplementedError

    def y(self, t):
        raise NotImplementedError

    def __eq__(self, other):
        return np.allclose(self.sample_points(-1, 1, 3), other.sample_points(-1, 1, 3))
"""
    def __new__(cls, a, b, c, d, e, f):
        M = np.array([[a, 1/2 * b], [1/2 * b, c]])
        d = np.linalg.det(M)

        if d == 0:
            pass

        if cls == Conic:
            for cls in Conic.__subclasses__():
                return cls
"""

class Line(Conic):
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
        self.xpos = 0
        self.ypos = 0

    def x(self, t):
        #return self.f(t)[0]
        return np.vectorize(lambda t : (t * (self.a - self.b) + self.b)[0])(t)

    def y(self, t):
        return np.vectorize(lambda t : (t * (self.a - self.b) + self.b)[0])(t)
        return self.f(t)[1]


    @classmethod
    def point_slope_form(cls, p: np.ndarray, m: float):
        """
            Creates a line from a point and a slope.

            \\[ y = mx + b \\]
        """
        b = p[1] - m*p[0]
        
        x0 = p[0] + 1
        y0 = m*x0 + b

        p0 = np.array([x0, y0])

        return cls(p, p0)


        
class Ellipse2(Conic):
    def __init__(self, a: float, b: float, xpos, ypos):
        self.c1 = 1/a**2
        self.c2 = 0
        self.c3 = 1/b**2
        self.c4 = -2 * xpos / a**2
        self.c5 = -2 * ypos / b**2
        self.c6 = xpos**2 / a**2 + ypos**2 / b**2 - 1
        
        self.a = a
        self.b = b
        self.e = np.sqrt(1 - (self.b**2 / self.a**2))
        self.f1 = np.array([self.a * self.e, 0])
        self.f2 = np.array([-self.a * self.e, 0])

    def __call__(self, t) -> np.ndarray:
        x = self.a * np.cos(t) 
        y = self.b * np.sin(t)
        return np.array([x, y])

    def x(self, t: np.ndarray) -> np.ndarray:
        return self.a * np.cos(t)

    def y(self, t: np.ndarray) -> np.ndarray:
        return self.b * np.sin(t)


class Ellipse(Conic):
    def __init__(self, a: float, b: float, xpos, ypos):
        self.a = a
        self.b = b
        self.xpos = xpos
        self.ypos = ypos
        #self.e = np.sqrt(1 - (self.b**2 / self.a**2))
        #self.f1 = np.array([self.a * self.e, 0])
        #self.f2 = np.array([-self.a * self.e, 0])

    def __call__(self, t) -> np.ndarray:
        x = self.a * np.cos(t) + self.xpos
        y = self.b * np.sin(t) + self.ypos
        return np.array([x, y])

    def x(self, t: np.ndarray) -> np.ndarray:
        return self.a * np.cos(t) + self.xpos

    def y(self, t: np.ndarray) -> np.ndarray:
        return self.b * np.sin(t) + self.ypos

    def intersection(self, line: Line) -> np.ndarray:
        """
            Calculates the intersection of an ellipse and a line

            Formulas are
            \\[ \\alpha = \\frac{-2a^2cm}{2(b^2 + a^2m^2)} \\]

            \\[ \\beta = a \\sqrt{\\frac{b^2 - c^2}{b^2 + a^2m^2} + \\frac{a^2c^2m^2}{b^2 + a^2m^2}} \\]

        """
        # Checks if line is 2D
        #if line.dim != 2:
        #    raise ValueError("Line must be 2D.")

        # Convert line to point-slope form
        m = (line.b[1] - line.a[1]) / (line.b[0] - line.a[0])
        # Find the y-intercept 
        c = line.a[1] - m * line.a[0]

        alpha = -(2*self.a**2*c*m) / (2*(self.b**2 + self.a**2*m**2))
        beta = self.a*np.sqrt((self.b**2 - c**2)/(self.b**2 + self.a**2*m**2) + self.a**2*c**2*m**2/(self.b**2 + self.a**2*m**2))

        p1 = np.array([alpha + beta + c, m*(alpha + beta) + c])
        p2 = np.array([alpha - beta + c, m*(alpha - beta) + c])

        # Checks if the points lie on the ellipse
        points = np.array([np.zeros(2), np.zeros(2)])
        if self.verify_point(p1):
            points[0] = p1
        elif self.verify_point(p2):
            points[1] = p2

        return np.array([p1, p2])

    def x_prime(self, t) -> float:
        return self.b * np.cos(t)

    def y_prime(self, t) -> float:
        return -self.a * np.sin(t)

    def verify_point(self, p: np.ndarray) -> bool:
        return np.isclose(p[0]**2 / self.a**2 + p[1]**2 / self.b**2, 1)

    def __eq__(self, other):
        return super().__eq__(other)

    def parametric_from_cartesian(self, x: float, y: float) -> np.ndarray:
        """
            Converts a point on the ellipse to a point on the unit circle.
        """
        return np.array([x / self.a, y / self.b])
       
        
    """
    def eccentricity(self):
        return np.sqrt(1 - (self.b**2 / self.a**2))

    def foci(self):
        return np.array([(0, self.a * e), -(0, self.a * self.e)])
    """

class Hyperbola(Conic):
    """
        Constructs a hyperbola

        By default, evaluating the hyperbola only evaluates the right branch. Use the left and right 

    """
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __call__(self, t):
        return self.right(t)

    def x(self, t):
        return self.a * np.cosh(t)

    def y(self, t):
        return self.b * np.sinh(t)

    def right(self, t):
        x = self.a * np.cosh(t)
        y = self.b * np.sinh(t)
        return (x, y)

    def left(self, t):
        x = -self.a * np.cosh(t)
        y = self.b * np.sinh(t)
        return (x, y)

class Parabola(Conic):
    def __init__(self, a: float):
        self.a = a

    def __call__(self, t):
        x = t
        y = self.a * t**2
        return (x, y)

    def x(self, t):
        return t

    def y(self, t):
        return self.a * t**2


class Circle(Ellipse):
    def __init__(self, xpos, ypos, r):
        self.r = r
        super().__init__(r, r, xpos, ypos)

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
        return cls(c.real, c.imag, r)

    @classmethod
    def from_two_points(cls, x1: np.ndarray, x2: np.ndarray):
        mid = (x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2
        r = np.linalg.norm(x1 - x2, ord=2)/2
        return cls(mid[0], mid[1], r)

    
    def __eq__(self, other):
        return super().__eq__(other)


"""
def map_ellipse_to_circle(ellipse: Ellipse):
    M = np.array([[1/ellipse.a, 0], [0, 1/ellipse.b]])
    a = at.AffineTransform(np.array([[1/ellipse.a, 0.0], [0.0, 1/ellipse.b]]), 0.0)
    t = np.linspace(-np.pi, np.pi, 100)
    x, y = ellipse.x(t), ellipse.y(t)
    x_0, y_0 = a(np.array([x, y]))

    plt.plot(x, y, 'r')
    plt.plot(x_0, y_0, 'g')
    plt.gca().set_aspect('equal')
    plt.show()

def map_hyperbola_to_rectangular_hyperbola(hyperbola: Hyperbola):
    M = np.array([[1/hyperbola.a, 0], [0, 1/hyperbola.b]])
    a = at.AffineTransform(np.array([[1., -1.], [1., 1.0]]), 0.0)
    t = np.linspace(-np.pi, np.pi, 100)
    x, y = hyperbola.right(t)
    x_0, y_0 = a(np.array([x, y]))

    plt.plot(x, y, 'r')
    plt.plot(x_0, y_0, 'g')
    plt.gca().set_aspect('equal')
    plt.show()

def map_parabola_to_standard_parabola(parabola: Parabola):
    M = np.array([[1/parabola.a, 0], [0, 1/parabola.a]])
    a = at.AffineTransform(np.array([[1/parabola.a, 0.0], [0.0, 1/parabola.a]]), 0.0)
    t = np.linspace(-np.pi, np.pi, 100)
    x, y = parabola.x(t), parabola.y(t)
    x_0, y_0 = a(np.array([x, y]))

    plt.plot(x, y, 'r')
    plt.plot(x_0, y_0, 'g')
    plt.gca().set_aspect('equal')
    plt.show()

def first_optical_property_of_ellipse(ellipse: Ellipse):
    # Plot the ellipse
    t = np.linspace(-np.pi, np.pi, 100)
    x, y = ellipse.x(t), ellipse.y(t)
    plt.plot(x, y, 'r')
    # Plot the foci
    plt.plot(ellipse.f1[0], ellipse.f1[1], 'ro')
    plt.plot(ellipse.f2[0], ellipse.f1[1], 'ro')
    # Pick a point p_0 on the ellipse
    p0 = ellipse(0.5)
    # Create a line from p_0 to one of the foci
    shapes.Line(p0, ellipse.f1)
    # Plot line
    plt.plot([p0[0], ellipse.f1[0]], [p0[1], ellipse.f1[1]], 'g')
    # Plot reflection line

    plt.show()
    

map_ellipse_to_circle(Ellipse(2, 1))
map_hyperbola_to_rectangular_hyperbola(Hyperbola(2, 1))
map_parabola_to_standard_parabola(Parabola(3))
a = 2
b = 1 
circle = Ellipse(a, b)
T = at.AffineTransform(np.array([[1/a, 0.0], [0.0, 1/b]]), 0.0)
t = np.linspace(-np.pi, np.pi, 100)

x, y = circle.x(t), circle.y(t)
print(x)
print(y)

x_0, y_0 = T(np.array([x, y]))

#y = circle.generate(t)
plt.plot(x_0, y_0)
plt.gca().set_aspect('equal')
plt.show()
#x = np.linspace(-np.pi, np.pi, 100)
#plt.plot(x, circle.generate(x))
"""


