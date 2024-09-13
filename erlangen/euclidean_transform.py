import numpy as np
import affine_transform as at
class EuclideanTransform(at.AffineTransform):

    def __init__(self, A, b):
        super().__init__(A, b)
        if A @ A.T != np.identity(A.shape[0]):
            raise ValueError('Matrix is not a Euclidean transform, must be orthogonal.')
        
