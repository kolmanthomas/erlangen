import numpy as np

class Group:
    pass

class MobiusTransform:
    def __init__(self, a: complex, b: complex, c: complex, d: complex):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.matrix = np.array([[a, b], [c, d]])

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
    """
        Frontend of the InvertibleMatrix class

        This is because P(n) = GL(n + 1), where P(n) is the set of projective transformations in n dimensions and GL(n + 1) is the set of invertible matrices in n + 1 dimensions

    """
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

"""
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

