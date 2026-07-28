from typing import Union, Generator
from math import lgamma as _lgamma, exp as _exp
from sympy import Rational, Symbol, lambdify
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation
from decayamplitude.backend import numpy as np
from functools import lru_cache as cache
from sympy.abc import x as placeholder
from itertools import product

def convert_angular(f):
    """
    Wrapper to convert all passed values of type Angular to type int
    """
    def wrapped(*args, **kwargs):
        args = [arg.value2 if isinstance(arg, Angular) else arg for arg in args]
        kwargs = {key: value.value2 if isinstance(value, Angular) else value for key, value in kwargs.items()}
        return f(*args, **kwargs)
    return wrapped

class Angular:

    @classmethod
    def generate_helicities(cls, *angular_momenta:list["Angular"]) -> list[tuple[int]]:
        """
        Generate all possible helicities for a given set of angular momenta
        """
        return list(product(*[angular.projections(return_int=True) for angular in angular_momenta]))


    def __init__(self, angular_momentum:int):
        if not isinstance(angular_momentum, int):
            raise TypeError("Angular momentum must be an integer")

        self.angular_momentum = angular_momentum
    
    def __str__(self):
        return f"J={self.value}"
    
    def __hash__(self) -> int:
        return hash(self.angular_momentum)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.angular_momentum == other.angular_momentum

    @property
    def value(self):
        return self.angular_momentum / 2

    @property
    def value2(self):
        return self.angular_momentum
    
    @property
    def parity(self):
        if self.angular_momentum % 2 != 0:
            raise ValueError(f"Angular momentum must be even. {self} has no defined parity!")
        return (-1) ** int(self.value)

    def index(self):
        return self.angular_momentum

    def projections(self, return_int=False):
        """
        Returns the possible projections of the angular momentum
        """
        if return_int:
            return [i for i in range(-self.index(), self.index() + 1, 2)]
        return [Angular(i) for i in range(-self.index(), self.index() + 1, 2)]    
    
    def __add__(self, other):
        if isinstance(other, int):
            return Angular(self.angular_momentum + other)
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
    
    def __neq__(self, other):
        return not self == other

    def __add__(self, other):
        return QN(self.angular + other.angular, self.parity * other.parity)
    
    def __sub__(self, other):
        return QN(self.angular - other.angular, self.parity * other.parity)
    
    def couple(self, other):
        """
        Couple two quantum numbers
        """
        return [QN(j, p) for j in self.angular.couple(other.angular) for p in [self.parity * other.parity]]
    
    @classmethod
    def generate_L_states(cls, state0: "QN", state1:"QN", state2:"QN")-> Generator[tuple["QN", "QN"], None, None]:
        """
        state0 -> state1 + state2
        S = coupled spins of state 1 and state 2
        L = angular momentum of the 1-2 system
        L can be covered by coupling J with state0, since the possible L values arise from coupling J with a series of L values and looking which one can in turn couple to state0 spin. This is in pricinple due to detailed balance.

        params:
        state0 : QN
            Quantum numbers of the initial state
        state1 : QN
            Quantum numbers of the first final state
        state2 : QN
            Quantum numbers of the second final state

        returns:
        Generator[tuple[QN, QN]]
            Generator of possible (L, S) states
            Partity is not given, since it is defined by state0

        """
        for S in state1.angular.couple(state2.angular):
            for L in S.couple(state0.angular):
                if (L.parity * state1.parity * state2.parity) == state0.parity:
                    yield (L, S)
    
    def projections(self, return_int=False):
        return self.angular.projections(return_int=return_int)
        

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

def wigner_small_d_sympy(theta, j, m1, m2):
    """Wigner small-d via sympy/lambdify. Kept for numerical validation.
    j, m1, m2 are 2× physical values. Cannot be used inside jax.jit."""
    d_func = get_wigner_function(j, m1, m2)
    d = d_func(theta)
    d = np.array(d, dtype=np.float64)
    d = np.nan_to_num(d, copy=True, nan=0.0)
    return d.astype(np.complex128)


@cache
def _wigner_d_coefficients(j2: int, m1_2: int, m2_2: int):
    """Pre-compute scalar (coeff, cos_power, sin_power) terms for
    d^j_{m'm}(β) = Σ coeff · cos(β/2)^cos_power · sin(β/2)^sin_power.

    All args are 2× physical values (integers so half-integers are exact).
    Formula: https://handwiki.org/wiki/Wigner_D-matrix
    """
    if abs(m1_2) > j2 or abs(m2_2) > j2:
        return ()   # d = 0 for projections outside [-j, j]
    jp_m1 = (j2 + m1_2) // 2   # j + m'
    jm_m1 = (j2 - m1_2) // 2   # j - m'
    jp_m2 = (j2 + m2_2) // 2   # j + m
    jm_m2 = (j2 - m2_2) // 2   # j - m
    log_pre = 0.5 * (_lgamma(jp_m1 + 1) + _lgamma(jm_m1 + 1) +
                     _lgamma(jp_m2 + 1) + _lgamma(jm_m2 + 1))
    mp_m_2 = (m1_2 - m2_2) // 2   # m' - m  (always an integer)
    s_min = max(0, -mp_m_2)        # max(0, m - m')
    s_max = min(jp_m2, jm_m1)     # min(j + m, j - m')
    terms = []
    for s in range(s_min, s_max + 1):
        log_den = (_lgamma(jp_m2 - s + 1) + _lgamma(s + 1) +
                   _lgamma(mp_m_2 + s + 1) + _lgamma(jm_m1 - s + 1))
        sign = 1 if (mp_m_2 + s) % 2 == 0 else -1
        coeff = sign * _exp(log_pre - log_den)
        cos_pow = j2 - mp_m_2 - 2 * s   # 2j + m - m' - 2s  (≥ 0)
        sin_pow = mp_m_2 + 2 * s         # m' - m + 2s       (≥ 0)
        terms.append((coeff, cos_pow, sin_pow))
    return tuple(terms)


def wigner_small_d(theta, j, m1, m2):
    """Wigner small-d matrix element. j, m1, m2 are 2× physical values (integers).
    JAX-traceable: works inside jit and vmap."""
    terms = _wigner_d_coefficients(j, m1, m2)
    if not terms:
        return np.zeros_like(np.asarray(theta, dtype=np.float64)).astype(np.complex128)
    cb2 = np.cos(theta * 0.5)
    sb2 = np.sin(theta * 0.5)
    result = sum(float(c) * cb2**cp * sb2**sp for c, cp, sp in terms)
    return np.asarray(result, dtype=np.complex128)


@convert_angular
def wigner_capital_d(phi, theta, psi, j, m1, m2):
    return (
        np.exp(-1j * phi * m1 / 2)
        * wigner_small_d(theta, j, m1, m2)
        * np.exp(-1j * psi * m2 / 2)
    )