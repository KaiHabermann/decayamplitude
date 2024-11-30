from typing import Union
from sympy import Rational, Symbol, lambdify
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation
import numpy as np
from functools import cache
from sympy.abc import x as placeholder


class Angular:
    def __init__(self, angular_momentum:int):
        if not isinstance(angular_momentum, int):
            raise TypeError("Angular momentum must be an integer")

        self.angular_momentum = angular_momentum
    
    def __str__(self):
        return f"J={self.angular_momentum}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.angular_momentum == other.angular_momentum

    def value(self):
        return self.angular_momentum / 2

    def index(self):
        return self.angular_momentum

    def projections(self):
        """
        Returns the possible projections of the angular momentum
        """
        return [Angular(i) for i in range(-self.index(), self.index() + 1, 2)]    
    
    def __add__(self, other):
        return Angular(self.angular_momentum + other.angular_momentum)
    
    def __sub__(self, other):
        return Angular(self.angular_momentum - other.angular_momentum)
    
    def couple(self, other):
        """
        Couple two angular momenta
        """
        minimum = abs(self.angular_momentum - other.angular_momentum)
        maximum = self.angular_momentum + other.angular_momentum
        return [Angular(i) for i in range(minimum, maximum + 1, 2)]

class QN:
    def __init__(self, angular_momentum:Union[int, Angular], parity: int) -> None:
        if isinstance(angular_momentum, int):
            self.angular = Angular(angular_momentum)
        else:
            self.angular = angular_momentum
        if not isinstance(parity, int):
            raise TypeError("Parity must be an integer not {}".format(type(parity)))
        if not parity in [-1, 1]:
            raise ValueError("Parity must be either -1 or 1 not {}".format(parity))
        self.parity = parity     

    def __str__(self):
        return f"{self.angular}^{self.parity}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.angular == other.angular and self.parity == other.parity

    def __add__(self, other):
        return QN(self.angular + other.angular, self.parity * other.parity)
    
    def __sub__(self, other):
        return QN(self.angular - other.angular, self.parity * other.parity)
    
    def couple(self, other):
        """
        Couple two quantum numbers
        """
        return [QN(j, p) for j in self.angular.couple(other.angular) for p in [self.parity * other.parity]]


@cache
def clebsch_gordan(j1, m1, j2, m2, J, M):
    """
    Return clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
    (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
    """

    cg = (
        CG(
            Rational(j1, 2),
            Rational(m1, 2),
            Rational(j2, 2),
            Rational(m2, 2),
            Rational(J, 2),
            Rational(M, 2),
        )
        .doit()
        .evalf()
    )
    cg = float(cg)
    if str(cg) == "nan":
        raise ValueError(f"CG({j1/2},{m1/2},{j2/2},{m2/2},{J/2},{M/2}) is not a number")
    return cg


@cache
def get_wigner_function(j: int, m1: int, m2: int):
    """
    Return Wigner small-d function. Note that all arguments should be multiplied by 2
    (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
    """
    j, m1, m2 = int(j), int(m1), int(m2)
    d = Rotation.d(Rational(j, 2), Rational(m1, 2), Rational(m2, 2), placeholder).doit().evalf()
    d = lambdify(placeholder, d, "numpy")
    return d

def wigner_small_d(theta, j, m1, m2):
    """Calculate Wigner small-d function. Needs sympy.
      theta : angle
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2)

    :param theta:
    :param j:
    :param m1: before rotation
    :param m2: after rotation

    """
    d_func = get_wigner_function(j, m1, m2)
    d = d_func(theta)
    # d = np.array(d)
    # d[np.isnan(d)] = 0
    d = np.nan_to_num(d, copy=True, nan=0.0)
    d = d.astype(np.complex128)
    return d


def wigner_capital_d(phi, theta, psi, j, m1, m2):
    return (
        np.exp(-1j * phi * m1 / 2)
        * wigner_small_d(theta, j, m1, m2)
        * np.exp(-1j * psi * m2 / 2)
    )