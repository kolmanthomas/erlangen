import numpy as np
import numpy.typing as npt

def arr_to_coords(a, b):
    return np.concatenate([a[:, None], b[:, None]], axis=1)

def _is_1D(a: npt.NDArray) -> bool:
    return a.ndim == 1

def assert_is_1D(a: npt.NDArray) -> None:
    assert _is_1D(a)

def _is_2D(a: npt.NDArray) -> bool:
    return a.ndim == 2

def assert_is_2D(a: npt.NDArray) -> None:
    assert _is_2D(a) 

def _is_2D_tensor(a: npt.NDArray[np.float_]) -> bool:
    return (_is_2D(a) and a.shape[1] == 2) or (_is_1D(a) and a.shape[0] == 2)

def _is_3D(a: npt.NDArray) -> bool:
    return a.ndim == 3

def assert_is_3D(a: npt.NDArray) -> None:
    assert _is_3D(a)

def _is_3D_tensor(a: npt.NDArray[np.float_]) -> bool:
    return (_is_3D(a) and a.shape[1] == 3) 

def assert_is_3D_tensor(a: npt.NDArray[np.float_]) -> None:
    assert _is_3D_tensor(a)

"""
def assert_is_2D_coords(a: npt.NDArray[np.float_]) -> None:
    assert _is_2D_coords(a)

def coords_to_complex(a: npt.NDArray[np.float_]) -> npt.NDArray[np.complex_]:
    assert_is_2D_tensor(a) 
    return a[:, 0] + 1j * a[:, 1]
"""

def complex_to_real(a: npt.NDArray[np.complex_]) -> npt.NDArray[np.float_]:
    """
        [z1, z2, ...] -> [[z1.real, z1.imag], [z2.real, z2.imag], ...]
    """
    assert np.iscomplexobj(a)
    return a.view(float).reshape(-1, 2)

def point_2D_to_3D(a):
    return np.insert(a, np.shape(a)[-1], 0, axis=-1)

def convert_2D_coords_to_3D(z: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    return np.concatenate((z, np.zeros((z.shape[0], 1))), axis=1)

def is_collinear_2D(x1, x2, x3) -> bool:
    print(x1)
    print(x2)
    print(x3)
    return np.isclose(np.linalg.det(np.array([[1, x1[0], x1[1]],
                                              [1, x2[0], x2[1]], 
                                              [1, x3[0], x3[1]]])), 0, atol=0.1)

def is_collinear_3D(x1: npt.NDArray[np.complex_], x2, x3) -> bool:
    print(x1)
    print(x2)
    print(x3)
    return np.allclose(np.cross(x2 - x1, x3 - x1), 0, atol=0.1)

def c_is_collinear(z1, z2, z3):
    w = (z3 - z1)/(z2 - z1)
    if np.isclose(w.imag, 0):
        return True
