import numpy as np
from projective_transform import ProjectiveTransform
from conics import Circle, Line

import numpy.typing as npt
from typing import List

import util


class MobiusTransform(ProjectiveTransform):
    """
        
        \\[ \\begin{pmatrix} 1/\\sqrt{2} & 1/\\sqrt{2} \\\\ -1/\\sqrt{2} & 1/\\sqrt{2}\\end{pmatrix} \\]

    """
    def __init__(self, a: complex, b: complex, c: complex, d: complex):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.A = np.array([[a, b], [c, d]])
        super().__init__(self.A)

    def __call__(self, z: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]: 
        """
        
        \\[ M = \\begin{pmatrix} 1/\\sqrt{2} & 1/\\sqrt{2} \\\\ -1/\\sqrt{2} & 1/\\sqrt{2}\\end{pmatrix} \\]

        Args: 
            z

        """
        z = np.atleast_1d(z)
        z_quotient = np.stack((z, np.ones(z.size, dtype=np.complex_)), axis=1)
        z_arr = super().__call__(z_quotient)
        z_arr = z_arr[:, 0] / z_arr[:,1]
        return z_arr

class InversiveTransform(MobiusTransform):
    """


    """

    def __init__(self, center: complex, radius: float):
        a = center
        b = radius**2 - center * center.conjugate()
        c = 1
        d = -center.conjugate()
        super().__init__(a, b, c, d)

    @classmethod
    def from_circle(cls, circle: Circle):
        return cls(complex(circle.xpos, circle.ypos), circle.r)

    def __call__(self, z: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
        return super().__call__(np.conjugate(z))

    """
    def __call__(self, z: complex | np.ndarray | Line | Circle) -> complex | Line | Circle:
        if isinstance(z, np.ndarray):
            return super().__call__(z)
        elif isinstance(z, complex):
            return super().__call__(z.conjugate())
        elif isinstance(z, Line | Circle):
            # Pick three points on the generalized circle
            x_min, x_max = -1, 1
            sample_points = z.sample_points(x_min, x_max, 3)
            # Convert to complex
            complex_sample_points = util.coords_to_complex(util.arr_to_coords(sample_points[0], sample_points[1]))
            z1 = super().__call__(complex_sample_points[0].conjugate()) 
            z2 = super().__call__(complex_sample_points[1].conjugate()) 
            z3 = super().__call__(complex_sample_points[2].conjugate())

            if util.c_is_collinear(z1, z2, z3):
                return Line(np.array([z1.real, z1.imag]), np.array([z2.real, z2.imag]))
            return Circle.from_three_points(z1, z2, z3)
    """



"""
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'MobiusTransform':
        return cls(matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1])

    def __call__(self, z: complex) -> complex:
        return (self.a * z + self.b) / (self.c * z + self.d)

    def inverse(self) -> 'MobiusTransform':
        return MobiusTransform(a=-self.d, b=self.b, c=-self.c, d=self.a)

    def __mul__(self, other: 'MobiusTransform') -> 'MobiusTransform':
        return MobiusTransform.from_matrix(self.matrix @ other.matrix)

    def __matmul__(self, other: 'MobiusTransform'):
        pass

    def __str__(self) -> str:
        return f'[{self.a}, {self.b}]\n[{self.c}, {self.d}]'

    def __repr__(self) -> str:
        return f'MobiusTransform({self.a}, {self.b}, {self.c}, {self.d})'

    def __eq__(self, other) -> bool:
        if not isinstance(other, MobiusTransform):
            return NotImplemented
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    @staticmethod
    def identity() -> 'MobiusTransform':
        return MobiusTransform(1, 0, 0, 1)
"""


"""
class AffineTransform:
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def __call__(self, z: complex) -> complex:
        return (self.a * z + self.b) / (self.c * z + self.d) + self.e

    def inverse(self) -> 'AffineTransform':
        det = self.a * self.d - self.b * self.c
        return AffineTransform(self.d/det, -self.b/det, -self.c/det, self.a/det, -self.e * self.a + self.b * self.f, -self.e * self.c + self.d * self.f)

    def __mul__(self, other: 'AffineTransform') -> 'AffineTransform':
        return AffineTransform(self.a * other.a + self.b * other.c, self.a * other.b + self.b * other.d, self.c * other.a + self.d * other.c, self.c * other.b + self.d * other.d, self.e + other.e, self.f + other.f)

    def __str__(self) -> str:
        return f'[{self.a}, {self.b}, {self.e}]\n[{self.c}, {self.d}, {self.f}]\n[0, 0, 1]'

    def __repr__(self) -> str:
        return f'AffineTransform({self.a}, {self.b}, {self.c}, {self.d}, {self.e}, {self.f})'

    def __eq__(self, other) -> bool:
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d and self.e == other.e and self.f == other.f

    @staticmethod
    def identity() -> 'AffineTransform':
        return AffineTransform(1, 0, 0, 1, 0, 0)

    @staticmethod
    def translation(t: complex) -> 'AffineTransform':
        return AffineTransform(1, 0, 0, 1, t.real, t.imag)

    @staticmethod
    def scaling(s: complex):
        pass


class InvertibleMatrix:
    def __init__(self, A: np.ndarray):
        self.A = A
        if np.linalg.det(A) == 0:
            raise ValueError('Matrix is not invertible')

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.A @ v

    def inverse(self) -> 'InvertibleMatrix':
        return InvertibleMatrix(np.linalg.inv(self.A))



class ProjectiveTransform:
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, i: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.i = i

    def __call__(self, z: complex) -> complex:
        return (self.a * z + self.b) / (self.g * z + self.h)

    def inverse(self) -> 'ProjectiveTransform':
        det = self.a * self.d - self.b * self.c
        return ProjectiveTransform(self.d/det, -self.b/det, -self.c/det, self.a/det, -self.e * self.a + self.b * self.f, -self.e * self.c + self.d * self.f, self.g, self.h, self.i)

    def __mul__(self, other: 'ProjectiveTransform') -> 'ProjectiveTransform':
        return ProjectiveTransform(self.a * other.a + self.b * other.g, self.a * other.b + self.b * other.h, self.c * other.a + self.d * other.g, self.c * other.b + self.d * other.h, self.e * other.a + self.f * other.g + self.g * other.i, self.e * other.b + self.f * other.h + self.h * other.i, self.g * other.c + self.h * other.d, self.g * other.d + self.h * other.h, self.g * other.i + self.h * other.i + self.i)

    def __str__(self) -> str:
        return f'[{self.a}, {self.b}, {self.e}]\n[{self.c}, {self.d}, {self.f}]\n[{self.g}, {self.h}, {self.i}]'

    @staticmethod
    def translation(t: complex) -> 'MobiusTransform':
        return MobiusTransform(t.real, t.imag, 0, 1)

    @staticmethod
    def scaling(s: complex) -> 'MobiusTransform':
        #return MobiusTransform(s, 0, 0, 1)

    @staticmethod
    def rotation(theta: float) -> 'MobiusTransform':
        #return MobiusTransform(np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta))

    @staticmethod
    def reflection() -> 'MobiusTransform':
        #return MobiusTransform(1, 0, 0, -1)

    @staticmethod
    def inversion() -> 'MobiusTransform':
"""

