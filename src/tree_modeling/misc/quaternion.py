# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

"""
Quaternion - Class (Python)

The class is adapted from:
https://github.com/PyMesh/PyMesh
"""

import numpy as np
from math import sqrt
from numpy.linalg import norm, svd
from typing import Union


class Quaternion:
    """This class implements quaternion used for 3D rotations.
    Attributes:
        w (``float``): same as ``quaternion[0]``.
        x (``float``): same as ``quaternion[1]``.
        y (``float``): same as ``quaternion[2]``.
        z (``float``): same as ``quaternion[3]``.
    """

    def __init__(self, quat: Union[np.ndarray, list[float], tuple[float, float, float, float]] = [1, 0, 0, 0]) -> None:
        """Initializes the quaternion with a given array or list."""
        self.__quat = np.array(quat, dtype=float)
        self.normalize()

    def fromData(self, v1: np.ndarray, v2: np.ndarray) -> "Quaternion":
        """Create the rotation to rotate v1 to v2
        Args:
            v1 (``numpy.ndarray``): From vector. Normalization not necessary.
            v2 (``numpy.ndarray``): To vector. Normalization not necessary.
        Returns:
            The following values are returned.
            * ``quat`` (:class:`Quaternion`): Corresponding quaternion that
              rotates ``v1`` to ``v2``.
        """
        eps = 1e-12
        v1 /= norm(v1)
        v2 /= norm(v2)
        c = np.dot(v1, v2)
        if c < -1.0 + eps:
            # v1 is parallel and opposite of v2
            u, s, v = svd(np.array([v1, v2]))
            axis = v[2, :]
        else:
            axis = np.cross(v1, v2)
            length = norm(axis)
            if length > 0.0:
                axis /= norm(axis)
            else:
                # Parallel vectors.
                axis = v1

        w_sq = 0.5 * (1.0 + c)
        scaled_axis = sqrt(1.0 - w_sq) * axis
        quat = np.array([sqrt(w_sq), scaled_axis[0], scaled_axis[1], scaled_axis[2]])

        return Quaternion(quat)

    def norm(self) -> float:
        """Quaternion norm."""
        n: float = norm(self.__quat)
        return n

    def normalize(self) -> None:
        """Normalize quaterion to have length 1."""
        n = self.norm()
        if n == 0:
            print(self.__quat)
            raise ZeroDivisionError("quaternion cannot be 0!")

        self.__quat /= n

    def __str__(self) -> str:
        return str(self.__quat)

    def __getitem__(self, i: int) -> float:
        return float(self.__quat[i])

    def __setitem__(self, i: int, val: float) -> None:
        self.__quat[i] = val

    def __mul__(self, other: Union["Quaternion", np.ndarray]) -> "Quaternion":
        """Multiplication: ``self`` * ``other``
        Args:
            ``other`` (:class:`Quaternion` or ``numpy.ndarray``): If it is of
                type ``numpy.ndarray``, it must be normalized.
        """
        r = Quaternion()
        a = self
        b = other
        r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return r

    def __rmul__(self, other: Union["Quaternion", np.ndarray]) -> "Quaternion":
        """Multiplication: ``other`` * ``self``
        This method is called only if other is not a :class:`Quaternion`.
        Args:
            ``other`` (:class:`Quaternion` or ``numpy.ndarray``): If it is of
                type ``numpy.ndarray``, it must be normalized.
        """
        r = Quaternion()
        a = other
        b = self
        r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return r

    def to_matrix(self) -> np.ndarray:
        """Convert to rotational matrix.
        Returns:
            ``numpy.ndarray``: The corresponding rotational matrix.
        """
        a = self.__quat
        return np.array(
            [
                [
                    1 - 2 * a[2] * a[2] - 2 * a[3] * a[3],
                    2 * a[1] * a[2] - 2 * a[3] * a[0],
                    2 * a[1] * a[3] + 2 * a[2] * a[0],
                ],
                [
                    2 * a[1] * a[2] + 2 * a[3] * a[0],
                    1 - 2 * a[1] * a[1] - 2 * a[3] * a[3],
                    2 * a[2] * a[3] - 2 * a[1] * a[0],
                ],
                [
                    2 * a[1] * a[3] - 2 * a[2] * a[0],
                    2 * a[2] * a[3] + 2 * a[1] * a[0],
                    1 - 2 * a[1] * a[1] - 2 * a[2] * a[2],
                ],
            ]
        )

    @property
    def w(self) -> float:
        """Returns the scalar part of the quaternion."""
        return float(self.__quat[0])

    @property
    def x(self) -> float:
        """Returns the first component of the vector part."""
        return float(self.__quat[1])

    @property
    def y(self) -> float:
        """Returns the second component of the vector part."""
        return float(self.__quat[2])

    @property
    def z(self) -> float:
        """Returns the third component of the vector part."""
        return float(self.__quat[3])
