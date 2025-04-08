from decayamplitude.backend import numpy as np
import decayamplitude
from decayamplitude.chain import DecayChain, MultiChain, AlignedChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN

from decayangle.decay_topology import Topology, Node
from decayangle.config import config as decayangle_config

from decayangle.lorentz import LorentzTrafo

from decayamplitude.rotation import wigner_capital_d

from collections import defaultdict

def constant_lineshape(*args):
    return 1

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

def resonances():
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
    return resonances1, resonances2, resonances3, resonances_dpd

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

    resonances1, resonances2, resonances3, resonances_dpd = resonances()

    from decayamplitude.backend import numpy as np
    # momenta = make_four_vectors(np.linspace(0,np.pi,10), np.linspace(0,np.pi,10), np.linspace(0,np.pi,10))
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

    decay_dpd_m = DecayChain(
        topology = topology1,
        resonances = resonances_dpd,
        momenta = momenta,
        final_state_qn = final_state_qn,
        convention = "minus_phi"
    )

    arguments1 = {
        resonances1[(2,3)].id : {
            "couplings":{
                (2, 2) : 1
            }
        }, 
        resonances1[0].id : {
            "couplings":{
                (2, 1) : 1
            }
        }
    }
    arguments2 = {
        resonances2[(1,2)].id : {
            "couplings":{
                (2, 3) : 1
            }
        }, 
        resonances2[0].id : {
            "couplings":{
                (2, 3) : 1
            }
        }
    }

    arguments3 = {
        resonances3[(1,2)].id : {
            "couplings":{
                (2, 3) : 1
            }
        }, 
        resonances3[0].id : {
            "couplings":{
                (2, 1) : 1
            }
        }
    }

    arguments_dpd = {
        resonances_dpd[(2,3)].id : {
            "couplings": {
                (4, 2): 1
            },
        },
        resonances_dpd[0].id : {
            "couplings": {
                (4, 3): 1
            }
        }
    }


    dpd_value = decay_dpd.matrix(-1, arguments_dpd)[(1, 2,0)]
    dpd_value_m = decay_dpd_m.matrix(-1, arguments_dpd)[(1, 2,0)]

    aligned_decay3 = AlignedChain(
        topology = topology2,
        resonances = resonances3,
        momenta = momenta,
        final_state_qn = final_state_qn,
        reference=decay_dpd,
        convention="helicity"
    )

    aligned_decay3_m = AlignedChain(
        topology = topology2,
        resonances = resonances3,
        momenta = momenta,
        final_state_qn = final_state_qn,
        reference=decay_dpd_m,
        convention="minus_phi"
    )
    value3 = aligned_decay3.aligned_matrix(-1, arguments3)[(1, 2, 0)]
    value3_m = aligned_decay3_m.aligned_matrix(-1, arguments3)[(1, 2, 0)]
    # this is a reference value copied from the output of the decayangle code
    # We can use this to harden against mistakes in the decayamplitude code
    assert np.allclose(dpd_value, (-0.14315554700441074 + 0.12414558894503328j))
    assert np.allclose(
        dpd_value_m, -0.03883258888101088 + 0.1854660829732478j
    )
    assert np.allclose(
        value3, -0.49899891547281655 + 0.030820810874496913j
    )
    assert np.allclose(
        value3_m, -0.37859261634645197 + 0.32652330831650717j
    )



    # assert np.allclose(
    #     terms_2_m[(-1, 1, 2, 0)][-1], -0.37859261634645197 + 0.32652330831650717j
    # )
    # assert np.allclose(unpolarized(full_amp), unpolarized(full_amp)[0])

def testShortThreeBodyAmplitude():
    """
    We can also combine the chains in a shorter way, by using the MultiChain class
    """
    final_state_qn = {
            1: QN(1, 1),
            2: QN(2, 1),
            3: QN(0, 1)
        }
    resonances1, resonances2, resonances3, resonances_dpd = resonances()
    momenta = make_four_vectors(1,2,np.linspace(0,np.pi,10))
    topology1 = Topology(
        0,
        decay_topology=((2,3), 1)
    )
    topology2 = Topology(
        0,
        decay_topology=((1, 2), 3)
    )

    chain1 = MultiChain.from_chains([
        DecayChain(
            topology = topology1,
            resonances = resonances1,
            momenta = momenta,
            final_state_qn = final_state_qn
        ),
        DecayChain(
            topology = topology1,
            resonances = resonances_dpd,
            momenta = momenta,
            final_state_qn = final_state_qn
        )
    ])

    chain2 = MultiChain.from_chains([
        DecayChain(
            topology = topology2,
            resonances = resonances2,
            momenta = momenta,
            final_state_qn = final_state_qn,
        ),
        DecayChain(
            topology = topology2,
            resonances = resonances3,
            momenta = momenta,
            final_state_qn = final_state_qn
        )
    ])

    merged_resonances = defaultdict(list)
    for key, resonance in resonances3.items():
        merged_resonances[key].append(resonance)
    for key, resonance in resonances2.items():
        merged_resonances[key].append(resonance)
    merged_resonances = dict(merged_resonances)

    merged_resonances[0] = [merged_resonances[0][0]]

    full = ChainCombiner([chain1, chain2])
    arguments = full.generate_couplings()

    matrix1 = full.combined_matrix(-1, arguments)
    matrix2 = full.combined_matrix(1, arguments)

    unpolarized, argnames = full.unpolarized_amplitude(full.generate_couplings(), complex_couplings=False)
    assert np.allclose(sum(abs(v)**2 for v in matrix1.values()) + sum(abs(v)**2 for v in matrix2.values()), unpolarized(*([1] * len(argnames))))
    assert np.allclose(unpolarized(*([1] * len(argnames))), unpolarized(*([1] * len(argnames)))[0])

    full2 = ChainCombiner([chain1, MultiChain(topology2, momenta=momenta, resonances=merged_resonances, final_state_qn=final_state_qn)])
  
    unpolarized2, argnames2 = full2.unpolarized_amplitude(full2.generate_couplings(), complex_couplings=False)

    # initial orientation should not affect the result
    assert np.allclose(unpolarized2(*([1] * len(argnames))), unpolarized2(*([1] * len(argnames2)))[0])

    assert np.allclose(unpolarized2(*([1] * len(argnames))) , unpolarized(*([1] * len(argnames))))


if __name__ == "__main__":
    testShortThreeBodyAmplitude()
    test_threebody_1()