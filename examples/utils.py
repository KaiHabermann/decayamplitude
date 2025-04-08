from decayangle.decay_topology import Topology, HelicityAngles, WignerAngles
from decayangle.lorentz import LorentzTrafo
from decayamplitude.backend import numpy as np

def constant_lineshape(*args):
    return 1

def BW_lineshape(mass):
    # usually the function is called for a fixed set of points in the phasespace. 
    # Thus we can give the mass values as fixed
    # decayamplitude will provide l and s values per default for the resonances, but not the mass values
    def BW(l,s, m0, gamma):
        return 1 / (mass**2 - m0**2 + 1j * mass * gamma) 
    return BW

def make_four_vectors(phi_rf, theta_rf, psi_rf):
    from decayamplitude.backend import numpy as np

    # Make sure, the sorting is turned off

    # Given values
    # Lc -> p K pi
    m0 = 6.32397
    m12 = 9.55283383**0.5
    m23 = 26.57159046**0.5
    m13 = 17.86811729**0.5
    m1, m2, m3 = 1, 2, 3
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
    p3a = np.sqrt(Kallen(m12sq, m3sq, m0sq)) / (2 * m0)

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

    # Vectors
    p1 = np.array([0, 0, p1z, E1])
    p2 = np.array([p2x, 0, p2z, E2])
    p3 = np.array([p3x, 0, p3z, E3])

    # Lorentz transformation
    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))

    # momenta = Topology(root=0, decay_topology=((1, 2), 3)).align_with_daughter(momenta, 3)
    # momenta = tree1.root.transform(LorentzTrafo(0, 0, 0, 0, -np.pi, 0), momenta)
    rotation = LorentzTrafo(0, 0, 0, phi_rf, theta_rf, psi_rf)

    momenta_23_rotated = tree1.root.transform(rotation, momenta)
    return momenta_23_rotated