import numpy as np
import decayamplitude
from decayamplitude.chain import DecayChain
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN

from decayangle.decay_topology import Topology, Node
from decayangle.config import config as decayangle_config

from decayangle.lorentz import LorentzTrafo

from decayamplitude.rotation import wigner_capital_d

def constant_lineshape(*args):
    return 1

def make_four_vectors(phi_rf, theta_rf, psi_rf):
    import numpy as np

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

def test_threebody_1():
    decayangle_config.sorting = "off" 
    topology1 = Topology(
        0,
        decay_topology=((2,3), 1)
    )

    topology2 = Topology(
        0,
        decay_topology=((1, 2), 3)
    )

    resonances1 = {
        (2,3): Resonance(Node((2, 3)), 0, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    resonances2 = {
        (1, 2): Resonance(Node((1, 2)), 3, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    resonances3 = {
        (1, 2): Resonance(Node((1, 2)), 1, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    resonances_dpd = {
        (2,3): Resonance(Node((2, 3)), 4, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    import numpy as np
    momenta = make_four_vectors(np.linspace(0,np.pi,10), np.linspace(0,np.pi,10), np.linspace(0,np.pi,10))
    momenta = make_four_vectors(0.3, np.arccos(0.4), 0.5)


    final_state_qn = {
            1: QN(1, 1),
            2: QN(2, 1),
            3: QN(0, 1)
        }
    decay = DecayChain(
        topology = topology1,
        resonances = resonances1,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    decay2 = DecayChain(
        topology = topology2,
        resonances = resonances2,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    decay3 = DecayChain(
        topology = topology2,
        resonances = resonances3,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    decay_dpd = DecayChain(
        topology = topology1,
        resonances = resonances_dpd,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    arguments1 = {
        resonances1[(2,3)].id : {
            "ls_couplings":{
                (2, 2) : 1
            }
        }, 
        resonances1[0].id : {
            "ls_couplings":{
                (2, 1) : 1
            }
        }
    }
    arguments2 = {
        resonances2[(1,2)].id : {
            "ls_couplings":{
                (2, 3) : 1
            }
        }, 
        resonances2[0].id : {
            "ls_couplings":{
                (2, 3) : 1
            }
        }
    }

    arguments3 = {
        resonances3[(1,2)].id : {
            "ls_couplings":{
                (2, 3) : 1
            }
        }, 
        resonances3[0].id : {
            "ls_couplings":{
                (2, 1) : 1
            }
        }
    }

    arguments_dpd = {
        resonances_dpd[(2,3)].id : {
            "ls_couplings": {
                (4, 2): 1
            },
        },
        resonances_dpd[0].id : {
            "ls_couplings": {
                (4, 3): 1
            }
        }
    }


    total = 0
    amp_dict = {}
    amp_dict2 = {}
    amp_dict3 = {}
    amp_dict_dpd = {}

    for lambdas in [
        {1:1, 2:2, 3:0},
        {1:1, 2:0, 3:0},
        {1:1, 2:-2, 3:0},
        {1:-1, 2:2, 3:0},
        {1:-1, 2:0, 3:0},
        {1:-1, 2:-2, 3:0}
    ]:
        for h0 in [-1, 1]:
            l1, l2, l3 = lambdas[1], lambdas[2], lambdas[3]
            amp = decay.chain_function(h0, lambdas=lambdas, helicity_angles=topology1.helicity_angles(momenta), arguments=arguments1)
            amp2 = decay2.chain_function(h0, lambdas=lambdas, helicity_angles=topology2.helicity_angles(momenta), arguments=arguments2)
            amp_dict[(h0, l1, l2, l3)] = amp
            amp_dict2[(h0, l1, l2, l3)] = amp2
            amp_dict3[(h0, l1, l2, l3)] = decay3.chain_function(h0, lambdas=lambdas, helicity_angles=topology2.helicity_angles(momenta), arguments=arguments3)
            amp_dict_dpd[(h0, l1, l2, l3)] = decay_dpd.chain_function(h0, lambdas=lambdas, helicity_angles=topology1.helicity_angles(momenta), arguments=arguments_dpd)

    def basis_change(dtc, rotation, final_state_qn):
        """
        Small helper function, which will perform a basis change on the dictionary of amplitudes.
        One needs a rotation for all final state particles.
        """
        new_dtc = {}
        for key, value in dtc.items():
            l0, l1, l2, l3 = key
            new_dtc[key] = sum(
                dtc[(l0, l1_, l2_, l3_)]
                * np.conj(wigner_capital_d(*(rotation[1]), final_state_qn[1].angular.value2, l1, l1_))
                * np.conj(wigner_capital_d(*(rotation[2]), final_state_qn[2].angular.value2, l2, l2_))
                * np.conj(wigner_capital_d(*(rotation[3]), final_state_qn[3].angular.value2, l3, l3_))
                for l1_ in final_state_qn[1].angular.projections(True)
                for l2_ in final_state_qn[2].angular.projections(True)
                for l3_ in final_state_qn[3].angular.projections(True)
            )
        return new_dtc

    def add_dict(dtc, dtc2):
        return {
            key: value + dtc2[key]
            for key, value in dtc.items()
        }

    full_amp = add_dict(amp_dict, basis_change(amp_dict2, topology2.relative_wigner_angles(topology1, momenta), final_state_qn))

    def unpolarized(dtc):
        return sum(
            abs(value)**2
            for value in dtc.values()
        )

    dpd_value = decay_dpd.chain_function(-1, lambdas={1:1, 2:2,3:0}, helicity_angles=topology1.helicity_angles(momenta), arguments=arguments_dpd)
    print(dpd_value/2**0.5)

    # this is a reference value copied from the output of the decayangle code
    # We can use this to harden against mistakes in the decayamplitude code
    print((-0.14315554700441074 + 0.12414558894503328j))


if __name__ == "__main__":
    test_threebody_1()