import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

from shapes import Circle

class ProjectiveTransform:
    """
        Class containing 

    """

    def __init__(self, A: np.ndarray):
        # Check if matrix is invertible
        if np.linalg.det(A) == 0:
            raise ValueError("Matrix is not a projective transform, must be invertible.")
        self.A = A

    def __call__(self, x: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
        return np.einsum('ij, bj -> bi', self.A, x)

"""
   
M = np.array([[2, 0, 1], [-1, 2, 3], [1, -1, 5]])
proj = ProjectiveTransform(M)
res = proj(np.array([1, 2, 3]))

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(1, 2, 3)
ax.scatter(res[0], res[1], res[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

"""

class ProjectivePoint():
    def __init__(self, x, y, z=1):
        self.x = x/z
        self.y = y/z
        self.z = z


"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x_t = lambda t : np.cos(t)
y_t = lambda t : np.sin(t)
t = np.linspace(0, 2 * np.pi, 100)
x = x_t(t)
y = y_t(t)
f = lambda x, y, z: x/z, y/z, z
proj_onto_z = np.vectorize(ProjectivePoint)(x, y)
z_np = np.linspace(-1, 1, 100)
for z in z_np:
    proj = np.vectorize(ProjectivePoint)(x, y, z)
    ax.scatter(proj.x, proj.y, proj.z)

plt.show()
# 3d line
ax.plot(np.array([1, -1]), np.array([0, 0]), np.array([1, -1]))
ax.plot(np.array([0, 0]), np.array([1, -1]), np.array([1, -1]))
ax.plot(np.array([-1, 1]), np.array([0, 0]), np.array([1, -1]))
ax.plot(np.array([0, 0]), np.array([-1, 1]), np.array([1, -1]))

ax.plot(x_t(t), y_t(t), np.ones(100))
ax.plot(x_t(t), y_t(t), -np.ones(100))

plt.show()
"""



