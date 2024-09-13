import numpy as np

class TransformError(ValueError):
    pass

def _assert_2d(a: np.ndarray) -> None:
    if a.ndim != 2:
        raise TransformError('%d-dimensional array given. Array must be '
                'two-dimensional' % a.ndim)

def _assert_inv(a: np.ndarray) -> np.ndarray:
    """
        Checks if numpy has inve
    """
    try:
        ainv = np.linalg.inv(a)
        return ainv
    except np.linalg.LinAlgError:
        raise TransformError('Matrix is not invertible')


