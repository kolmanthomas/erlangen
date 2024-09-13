import numpy as np

class InvertibleMatrix():
    def __init__(self, matrix: np.ndarray):
        if np.linalg.det(matrix) == 0:
            raise ValueError('Matrix is not invertible')
        self.M = matrix

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.M @ v

    def inverse(self) -> 'InvertibleMatrix':
        return InvertibleMatrix(np.linalg.inv(self.M))





