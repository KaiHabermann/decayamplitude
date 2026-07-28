from __future__ import annotations
from decayamplitude.rotation import QN
from decayamplitude.chain import DecayChain, MultiChain, AlignedMultiChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.resonance import Resonance
from decayangle.decay_topology import Topology, Node
from decayamplitude.kinematics_helpers import mass_from_node

import numpy as np
def constant_lineshape(*args):
    return 1.0


def test_multi_chain():
    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
            1: QN(0, 1), 
            2: QN(0, 1), 
            3: QN(1, 1), 
            4: QN(1, -1) 
        }


    m = mass_from_node(Node((1,2,3)), momenta)
    resonances_hadronic = {
        (1,2): [
            # Here the hadronic resonances go+
            # These will decay strong, so we need to conserve parity
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=["D_2300_M", "D_2300_Gamma"], preserve_partity=True, name="Resnance1"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=constant_lineshape, argnames=["D_2460_M", "D_2460_Gamma"], preserve_partity=True, name="Resonance2"),
        ],
        (3, 4): 
        [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance3")],
        (1,2,3): 
        [Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance4")],

        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }

    topology1 = Topology(
        0,
        decay_topology=((1,2), (3, 4))
    )

    momenta = topology1.to_rest_frame(momenta)

    topology2 = Topology(
        0,
        decay_topology=(((1,2), 3) ,4 )
    )

    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain2 = MultiChain(
        topology = topology2,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    assert len(chain1.chains) == 2
    for chain in chain1.chains:
        # root resonance + 2 chain resonances
        assert len(chain.resonances) == 3

    assert len(chain2.chains) == 2
    for chain in chain2.chains:
        assert len(chain.resonances) == 3

    combined = ChainCombiner([chain1, chain2])
    func, params = combined.unpolarized_amplitude(combined.generate_couplings())


def test_single_chain_unpolarized_amplitude():
    """Test that we can extract a single chain from a combiner and call unpolarized_amplitude on it."""
    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
            1: QN(0, 1), 
            2: QN(0, 1), 
            3: QN(1, 1), 
            4: QN(1, -1) 
        }

    m = mass_from_node(Node((1,2,3)), momenta)
    resonances_hadronic = {
        (1,2): [
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=["D_2300_M", "D_2300_Gamma"], preserve_partity=True, name="Resnance1"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=constant_lineshape, argnames=["D_2460_M", "D_2460_Gamma"], preserve_partity=True, name="Resonance2"),
        ],
        (3, 4): 
        [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance3")],
        (1,2,3): 
        [Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance4")],

        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }

    topology1 = Topology(
        0,
        decay_topology=((1,2), (3, 4))
    )

    momenta = topology1.to_rest_frame(momenta)

    topology2 = Topology(
        0,
        decay_topology=(((1,2), 3) ,4 )
    )

    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain2 = MultiChain(
        topology = topology2,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    combined = ChainCombiner([chain1, chain2])
    
    # Get a single chain from the combiner
    single_chains = combined.single_chains
    assert len(single_chains) > 0, "single_chains should return at least one chain"
    
    # Get the first single chain
    single_chain = single_chains[0]
    
    # Generate couplings for the single chain
    ls_couplings = single_chain.generate_couplings()
    
    # Call unpolarized_amplitude on the single chain
    func, params = single_chain.unpolarized_amplitude(ls_couplings)
    func(*([1] * len(params)))


def test_single_chains_are_decay_chains():
    """Test that all elements returned by single_chains are DecayChain instances and not MultiChain or AlignedMultiChain."""
    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
            1: QN(0, 1), 
            2: QN(0, 1), 
            3: QN(1, 1), 
            4: QN(1, -1) 
        }

    m = mass_from_node(Node((1,2,3)), momenta)
    resonances_hadronic = {
        (1,2): [
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=["D_2300_M", "D_2300_Gamma"], preserve_partity=True, name="Resnance1"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=constant_lineshape, argnames=["D_2460_M", "D_2460_Gamma"], preserve_partity=True, name="Resonance2"),
        ],
        (3, 4): 
        [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance3")],
        (1,2,3): 
        [Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance4")],

        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }

    topology1 = Topology(
        0,
        decay_topology=((1,2), (3, 4))
    )

    momenta = topology1.to_rest_frame(momenta)

    topology2 = Topology(
        0,
        decay_topology=(((1,2), 3) ,4 )
    )

    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain2 = MultiChain(
        topology = topology2,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    combined = ChainCombiner([chain1, chain2])
    
    # Get all single chains from the combiner
    single_chains = combined.single_chains
    assert len(single_chains) > 0, "single_chains should return at least one chain"
    
    # Verify all elements are DecayChain instances
    for chain in single_chains:
        assert isinstance(chain, DecayChain), f"Expected DecayChain, got {type(chain)}"
        assert not isinstance(chain, MultiChain), f"single_chains should not contain MultiChain instances, got {type(chain)}"
        assert not isinstance(chain, AlignedMultiChain), f"single_chains should not contain AlignedMultiChain instances, got {type(chain)}"


def test_helicity_scheme_unpolarized_amplitude():
    """Test that Resonances with scheme='helicity' can produce an unpolarized amplitude."""
    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
        1: QN(0, 1),
        2: QN(0, 1),
        3: QN(1, 1),
        4: QN(1, -1),
    }

    resonances = {
        (1, 2): [
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="R12_helicity", scheme="helicity"),
        ],
        (3, 4): [
            Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="R34_helicity", scheme="helicity"),
        ],
        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0_helicity", scheme="helicity")],
    }

    topology = Topology(0, decay_topology=((1, 2), (3, 4)))
    momenta = topology.to_rest_frame(momenta)

    chain = MultiChain(
        topology=topology,
        resonances=resonances,
        momenta=momenta,
        final_state_qn=final_state_qn,
    )

    combined = ChainCombiner([chain])
    func, params = combined.unpolarized_amplitude(combined.generate_couplings())
    result = func(*([1] * len(params)))
    assert result is not None


def test_external_wigner_matrices():
    """Test that helicity D-matrices and alignment wigner matrices can be passed from the outside."""
    import copy
    import jax

    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
            1: QN(0, 1),
            2: QN(0, 1),
            3: QN(1, 1),
            4: QN(1, -1)
        }

    resonances_hadronic = {
        (1,2): [
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=["D_2300_M", "D_2300_Gamma"], preserve_partity=True, name="Resnance1"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=constant_lineshape, argnames=["D_2460_M", "D_2460_Gamma"], preserve_partity=True, name="Resonance2"),
        ],
        (3, 4):
        [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance3")],
        (1,2,3):
        [Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance4")],

        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }

    topology1 = Topology(
        0,
        decay_topology=((1,2), (3, 4))
    )

    momenta = topology1.to_rest_frame(momenta)

    topology2 = Topology(
        0,
        decay_topology=(((1,2), 3) ,4 )
    )

    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain2 = MultiChain(
        topology = topology2,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    combined = ChainCombiner([chain1, chain2])
    couplings = combined.generate_couplings()

    # unpolarized amplitude
    func, params = combined.unpolarized_amplitude(couplings)
    func_ext, params_ext, wigner = combined.unpolarized_amplitude(couplings, external_wigner=True)
    assert "wigner_matrices" in params_ext
    coupling_values = {name: 1.0 for name in params_ext if name != "wigner_matrices"}

    reference_value = func(*([1.0] * len(params)))
    external_value = func_ext(wigner_matrices=wigner, **coupling_values)
    assert np.allclose(reference_value, external_value)

    # the externally passed values are actually used: perturbing the D-matrix elements changes the result
    modified = copy.deepcopy(wigner)
    for chain_data in [modified["reference"], *modified["aligned"]]:
        if "helicity_wigner_dict" in chain_data:
            for node_key in chain_data["helicity_wigner_dict"]:
                for mn in chain_data["helicity_wigner_dict"][node_key]:
                    chain_data["helicity_wigner_dict"][node_key][mn] = (
                        chain_data["helicity_wigner_dict"][node_key][mn] * 2.0
                    )
        if "wigner_dict" in chain_data:
            for particle_key in chain_data["wigner_dict"]:
                for hh in chain_data["wigner_dict"][particle_key]:
                    chain_data["wigner_dict"][particle_key][hh] = (
                        chain_data["wigner_dict"][particle_key][hh] * 2.0
                    )
    modified_value = func_ext(wigner_matrices=modified, **coupling_values)
    assert not np.allclose(reference_value, modified_value)

    # the internal structures are restored after the call
    assert np.allclose(func(*([1.0] * len(params))), reference_value)

    # matrix function
    mfunc, margnames = combined.matrix_function(couplings)
    mfunc_ext, margnames_ext, wigner_m = combined.matrix_function(couplings, external_wigner=True)
    margs = {name: 1.0 for name in margnames if name != "h0"}
    m_ref = mfunc(h0=0, **margs)
    m_ext = mfunc_ext(h0=0, wigner_matrices=wigner_m, **margs)
    assert all(np.allclose(m_ref[k], m_ext[k]) for k in m_ref)

    # polarized amplitude
    pol, hel_names, coupling_names = combined.polarized_amplitude(couplings)
    pol_ext, hel_names_ext, rest_ext, wigner_p = combined.polarized_amplitude(couplings, external_wigner=True)
    hel_values = {"h0": 0, "h_1": 0, "h_2": 0, "h_3": 1, "h_4": 1}
    p_ref = pol(**hel_values, **{n: 1.0 for n in coupling_names})
    p_ext = pol_ext(**hel_values, wigner_matrices=wigner_p, **{n: 1.0 for n in rest_ext if n != "wigner_matrices"})
    assert np.allclose(p_ref, p_ext)


def test_helicity_scheme_external_wigner():
    """Test that external_wigner=True works when resonances use scheme='helicity'."""
    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
        1: QN(0, 1),
        2: QN(0, 1),
        3: QN(1, 1),
        4: QN(1, -1),
    }
    resonances = {
        (1, 2): [
            Resonance(Node((1, 2)), quantum_numbers=QN(2, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="R12h", scheme="helicity"),
        ],
        (3, 4): [
            Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="R34h", scheme="helicity"),
        ],
        (1, 2, 3): [
            Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="R123h", scheme="helicity"),
        ],
        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0h", scheme="helicity")],
    }

    topology1 = Topology(0, decay_topology=((1, 2), (3, 4)))
    momenta = topology1.to_rest_frame(momenta)

    topology2 = Topology(0, decay_topology=(((1, 2), 3), 4))

    chain1 = MultiChain(topology=topology1, resonances=resonances, momenta=momenta, final_state_qn=final_state_qn)
    chain2 = MultiChain(topology=topology2, resonances=resonances, momenta=momenta, final_state_qn=final_state_qn)

    combined = ChainCombiner([chain1, chain2])
    couplings = combined.generate_couplings()

    func, params = combined.unpolarized_amplitude(couplings)
    func_ext, params_ext, wigner = combined.unpolarized_amplitude(couplings, external_wigner=True)

    assert "wigner_matrices" in params_ext
    coupling_values = {name: 1.0 for name in params_ext if name != "wigner_matrices"}
    reference_value = func(*([1.0] * len(params)))
    external_value = func_ext(wigner_matrices=wigner, **coupling_values)
    assert np.allclose(reference_value, external_value)
    assert reference_value != 0.0


if __name__ == "__main__":
    test_multi_chain()
    test_single_chain_unpolarized_amplitude()
    test_single_chains_are_decay_chains()
    test_helicity_scheme_unpolarized_amplitude()
    test_external_wigner_matrices()
    test_helicity_scheme_external_wigner()