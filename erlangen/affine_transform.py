from typing import Any
import numpy as np
import numpy.typing as npt

import transform
from projective_transform import *

class AffineTransform(ProjectiveTransform):
    """

    """
    def __init__(self, A: np.ndarray[tuple[int, int], np.dtype[np.float_]], b: np.ndarray[Any, np.dtype[np.float_]]):
        """
        if not A.dtype == b.dtype:
            raise TypeError('b must be of the same type as the entries of A')
        """
        transform._assert_2d(A)
        r, c = A.shape
        P = np.zeros((r + 1, c + 1))
        P[:r, :c] = A
        P[:r, c] = b
        P[r, c] = 1
        print(P)

        self.A = A
        self.b = b

    """
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
    """



"""
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = np.array([1, 2, 3])
AffineTransform(A, b)

A = np.array([[1. + 0.j, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
b = np.array([1. + 5.j, 2., 3.])
AffineTransform(A, b)

a = 3 + 42j
print(type(a.real))

"""
"""
A = np.array([[1, 0, 0]])
b = np.array([1, 2, 3])
AffineTransform(A, b)

A = np.array([1, 0, 0])
b = np.array([1, 2, 3])
AffineTransform(A, b)
A = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
b = np.array([1, 2, 3])
AffineTransform(A, b)
"""
