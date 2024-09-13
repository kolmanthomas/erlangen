"""
import erlangen 
import erlangen.affine_transform as at

import numpy as np

def test_conic_construction():
    a = at.AffineTransform(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 1]))
    assert a.A[0, 0] == 1
"""
import erlangen.conics as conics

def test_eq():
    e1 = conics.Ellipse(a=1, b=1, xpos=0, ypos=0)
    e2 = conics.Circle(xpos=0, ypos=0, r=1)
    assert e1 == e2
