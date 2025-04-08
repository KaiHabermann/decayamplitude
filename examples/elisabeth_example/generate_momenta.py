from decayangle.decay_topology import Topology, TopologyCollection, HelicityAngles
from decayangle.lorentz import LorentzTrafo
from decayangle.config import config as decayangle_config
import numpy as np

def make_numpy(f):
    def wrapper(*args, **kwargs):
        args = [np.array(arg) for arg in args]
        kwargs = {k: np.array(v,dtype=np.float64) for k, v in kwargs.items()}
        return f(*args, **kwargs)
    return wrapper

@make_numpy
def make_four_vectors_from_dict(mkpisq, mkpsq, mppisq, phip, thetap, chi, phi_Kst = None, theta_Kst=None, phi_K=None, theta_K=None):
    import numpy as np


    # Make sure, the sorting is turned off

    # Given values
    # Lc -> p K pi
    # 0 -> 1 2 3
    m12 = np.sqrt(mkpsq)
    m23 = np.sqrt(mkpisq)
    m13 = np.sqrt(mppisq)
    m1, m2, m3 = 0.938272088, 0.493677, 0.13957039
    m0 = np.sqrt((mkpisq + mkpsq + mppisq) - m1**2  - m2**2 - m3**2)

    # Squared masses
    m0sq, m1sq, m2sq, m3sq, m12sq, m23sq = [x**2 for x in [m0, m1, m2, m3, m12, m23]]

    # Källén function
    def Kallen(x, y, z):
        return x**2 + y**2 + z**2 - 2 * (x * y + x * z + y * z)

    # Calculating missing mass squared using momentum conservation
    m31sq = m0sq + m1sq + m2sq + m3sq - m12sq - m23sq

    # Momenta magnitudes
    p1a = np.sqrt(Kallen(m23sq, m1sq, m0sq)) / (2 * m0)
    p2a = np.sqrt(Kallen(m31sq, m2sq, m0sq)) / (2 * m0)

    # Directions and components
    cos_zeta_12_for0_numerator = (m0sq + m1sq - m23sq) * (
        m0sq + m2sq - m31sq
    ) - 2 * m0sq * (m12sq - m1sq - m2sq)
    cos_zeta_12_for0_denominator = np.sqrt(Kallen(m0sq, m2sq, m31sq)) * np.sqrt(
        Kallen(m0sq, m23sq, m1sq)
    )
    cos_zeta_12_for0 = cos_zeta_12_for0_numerator / cos_zeta_12_for0_denominator

    p1z = -p1a
    p2z = -p2a * cos_zeta_12_for0
    p2x = np.sqrt(p2a**2 - p2z**2)
    p3z = -p2z - p1z
    p3x = -p2x

    # Energy calculations based on the relativistic energy-momentum relation
    E1 = np.sqrt(p1z**2 + m1sq)
    E2 = np.sqrt(p2z**2 + p2x**2 + m2sq)
    E3 = np.sqrt(p3z**2 + p3x**2 + m3sq)

    # Vectors such that we align with proton momentum
    p1 = np.array([0, 0, p1z, E1])
    p2 = np.array([p2x, 0, p2z, E2])
    p3 = np.array([p3x, 0, p3z, E3])

    # Lorentz transformation
    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))
    # momenta are now in x-z plane

    phip = -np.pi + phip 

    thetap = np.pi - thetap
    chi = -np.pi + chi
    # rotation = LorentzTrafo(0, 0, 0, phip, thetap, chi)
    rotation = LorentzTrafo(0, 0, 0, phi_Kst, theta_Kst, phi_K)

    momenta_23_rotated = tree1.root.transform(rotation, momenta)
    return momenta_23_rotated


decayangle_config.sorting = "off"