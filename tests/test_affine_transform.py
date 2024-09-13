"""
import erlangen.affine_transform as at
import pytest
import numpy as np

def test_affine_transform_construction():
    # Nn-invertible matrices raise an error
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    b = np.array([0, 0, 1])
    with pytest.raises(ValueError):
         at.AffineTransform(A, b)

    # Mismatch of types of A and b raises an erro
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([0, 0, 0])
    with pytest.raises(TypeError):
         at.AffineTransform(A, b)
"""
    
