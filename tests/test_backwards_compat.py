"""
Backwards compatibility with the decayamplitude <= 0.1.9 API, where momenta were bound
to the chains at construction and lineshapes did not receive the resonance mass.
The legacy API must give the same numbers as the >= 0.2.0 API.
"""
import warnings

import pytest

from decayamplitude.backend import numpy as np
from decayamplitude.chain import DecayChain, MultiChain, AlignedChain, AlignedMultiChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.resonance import Resonance, _infer_lineshape_takes_mass
from decayamplitude.rotation import QN
from decayamplitude.kinematics_helpers import mass_from_node
from decayangle.decay_topology import Topology, Node

from test_threebody import make_four_vectors

FINAL_STATE_QN = {1: QN(1, 1), 2: QN(2, 1), 3: QN(0, 1)}
TOPOLOGY1 = Topology(0, decay_topology=((2, 3), 1))
TOPOLOGY2 = Topology(0, decay_topology=((1, 2), 3))


def constant_lineshape(*args):
    return 1


def bw_new(mass, l, s, m0, gamma):
    return 1 / (m0**2 - mass**2 - 1j * m0 * gamma)


def bw_legacy(mass):
    # <=0.1.9 style: the mass is closed over, the lineshape only gets (l, s, *params)
    def bw(l, s, m0, gamma):
        return 1 / (m0**2 - mass**2 - 1j * m0 * gamma)
    return bw


def make_resonances(momenta=None):
    """momenta=None -> new style lineshapes, otherwise legacy closures over the masses."""
    def ls(topology, node):
        if momenta is None:
            return bw_new
        return bw_legacy(mass_from_node(topology.nodes[node], momenta))

    root = lambda: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Lc")
    resonances1 = {
        (2, 3): Resonance(Node((2, 3)), 2, -1, lineshape=ls(TOPOLOGY1, (2, 3)), argnames=["m_K", "g_K"], preserve_partity=False, name="K23"),
        0: root(),
    }
    resonances2 = {
        (1, 2): [
            Resonance(Node((1, 2)), 3, -1, lineshape=ls(TOPOLOGY2, (1, 2)), argnames=["m_D", "g_D"], preserve_partity=False, name="D12a"),
            Resonance(Node((1, 2)), 1, 1, lineshape=ls(TOPOLOGY2, (1, 2)), argnames=["m_L", "g_L"], preserve_partity=False, name="D12b"),
        ],
        0: [root()],
    }
    return resonances1, resonances2


PARAMS = {"m_K": 4.9, "g_K": 0.3, "m_D": 3.1, "g_D": 0.2, "m_L": 3.3, "g_L": 0.5}


@pytest.fixture(scope="module")
def momenta():
    return make_four_vectors(np.linspace(0.1, 3.0, 7), np.arccos(0.4), 0.5)


@pytest.fixture(scope="module")
def combiners(momenta):
    r1, r2 = make_resonances()
    new = ChainCombiner([
        DecayChain(TOPOLOGY1, r1, FINAL_STATE_QN),
        MultiChain(TOPOLOGY2, FINAL_STATE_QN, resonances=r2),
    ])
    r1_old, r2_old = make_resonances(momenta)
    with pytest.warns(DeprecationWarning):
        # legacy positional order: (topology, resonances, momenta, final_state_qn)
        c1 = DecayChain(TOPOLOGY1, r1_old, momenta, FINAL_STATE_QN)
    with pytest.warns(DeprecationWarning):
        # legacy keyword usage
        c2 = MultiChain(topology=TOPOLOGY2, momenta=momenta, final_state_qn=FINAL_STATE_QN, resonances=r2_old)
    legacy = ChainCombiner([c1, c2])
    return new, legacy


def _coupling_values(names):
    return {name: (0.3 + 0.1 * i) for i, name in enumerate(names)}


def test_legacy_mode_detection(combiners):
    new, legacy = combiners
    assert not new.legacy
    assert legacy.legacy
    assert all(chain.legacy for chain in legacy.aligned_chains)


def test_lineshape_signature_inference():
    assert _infer_lineshape_takes_mass(bw_new, 2) is True
    assert _infer_lineshape_takes_mass(bw_legacy(1.0), 2) is False
    assert _infer_lineshape_takes_mass(constant_lineshape, 0) is None
    assert _infer_lineshape_takes_mass(lambda l, s, *a, d1_mass=None, d2_mass=None: 1, 0) is None
    assert _infer_lineshape_takes_mass(lambda l, s, m0, g, *, d1_mass=None, d2_mass=None: 1, 2) is False
    r = Resonance(Node((1, 2)), 1, 1, lineshape=constant_lineshape, argnames=[], lineshape_takes_mass=True)
    assert r.lineshape_takes_mass is True
    assert r.copy().lineshape_takes_mass is True


def test_unpolarized_amplitude(combiners, momenta):
    new, legacy = combiners
    f_new, names_new = new.unpolarized_amplitude(new.generate_couplings())
    f_old, names_old = legacy.unpolarized_amplitude(legacy.generate_couplings())
    assert names_new[0] == "momenta"
    assert "momenta" not in names_old
    assert sorted(names_new[1:]) == sorted(names_old)

    values = {**_coupling_values(sorted(names_old)), **PARAMS}
    values = {k: v for k, v in values.items() if k in names_old}
    result_old = f_old(**values)
    result_new = f_new(momenta, **values)
    assert np.all(np.isfinite(result_old))
    assert np.allclose(result_old, result_new)
    # positional call as in 0.1.9
    assert np.allclose(f_old(*[values[n] for n in names_old]), result_old)


def test_polarized_amplitude(combiners, momenta):
    new, legacy = combiners
    f_old, hel_old, names_old = legacy.polarized_amplitude(legacy.generate_couplings())
    f_new, hel_new, names_new = new.polarized_amplitude(new.generate_couplings(), complex_couplings=False)
    assert hel_old == hel_new == ["h0", "h_1", "h_2", "h_3"]
    # 0.1.9 used real couplings in polarized_amplitude
    assert not any(n.endswith("_imaginary") for n in names_old)
    assert sorted(names_old) == sorted(names_new)

    values = {**_coupling_values(sorted(names_old)), **PARAMS}
    values = {k: v for k, v in values.items() if k in names_old}
    hel = {"h0": 1, "h_1": -1, "h_2": 2, "h_3": 0}
    assert np.allclose(f_old(**hel, **values), f_new(momenta, **hel, **values))


def test_matrix_function_and_combined_matrix(combiners, momenta):
    new, legacy = combiners
    f_old, names_old = legacy.matrix_function(legacy.generate_couplings())
    f_new, names_new = new.matrix_function(new.generate_couplings())
    assert names_old[0] == "h0"
    assert names_new[:2] == ["momenta", "h0"]
    values = {**_coupling_values(sorted(names_old)), **PARAMS}
    values = {k: v for k, v in values.items() if k in names_old and k != "h0"}
    m_old = f_old(h0=-1, **values)
    m_new = f_new(momenta, h0=-1, **values)
    assert m_old.keys() == m_new.keys()
    for key in m_old:
        assert np.allclose(m_old[key], m_new[key])

    # 0.1.9 style direct calls without momenta
    arguments = {**legacy.generate_couplings(), **PARAMS}
    m_direct = legacy.combined_matrix(-1, arguments)
    lambdas = {1: 1, 2: 0, 3: 0}
    key = tuple(lambdas[k] for k in sorted(lambdas))
    assert np.allclose(legacy.combined_function(-1, lambdas, arguments), m_direct[key])
    new_arguments = {**new.generate_couplings(), **PARAMS}
    assert np.allclose(new.combined_matrix(-1, new_arguments, momenta)[key], m_direct[key])


def test_aligned_chain_legacy_api(momenta):
    r1, r2 = make_resonances(momenta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reference = DecayChain(TOPOLOGY1, r1, momenta, FINAL_STATE_QN)
        # legacy positional order: (topology, resonances, momenta, final_state_qn, reference)
        aligned = AlignedChain(TOPOLOGY2, {k: v[0] for k, v in r2.items()}, momenta, FINAL_STATE_QN, reference)
        multi = MultiChain(TOPOLOGY2, momenta, FINAL_STATE_QN, resonances=r2)
        aligned_multi = AlignedMultiChain.from_multichain(multi, reference)

    assert set(aligned.wigner_rotation.keys()) >= {1, 2, 3}
    assert aligned.helicity_angles is not None
    arguments = {**aligned.generate_couplings(), **PARAMS}
    assert aligned.aligned_matrix(1, arguments).keys() == aligned.aligned_matrix(1, arguments, momenta).keys()
    for key, value in aligned.aligned_matrix(1, arguments).items():
        assert np.allclose(value, aligned.aligned_matrix(1, arguments, momenta)[key])

    # precomputed wigner rotation can be passed explicitly as in 0.1.9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aligned2 = AlignedChain(TOPOLOGY2, {k: v[0] for k, v in r2.items()}, momenta, FINAL_STATE_QN, reference, aligned.wigner_rotation)
    assert aligned2.wigner_rotation is aligned.wigner_rotation

    f, names = aligned.aligned_matrix_function(aligned.generate_couplings())
    assert names[0] == "h0"
    assert aligned_multi.legacy and aligned_multi.wigner_dict.keys() == {1, 2, 3}


def test_resonance_amplitude_legacy_call_signature(combiners):
    _, legacy = combiners
    chain = legacy.reference
    node = chain.root
    resonance = node.resonance
    arguments = {**chain.generate_couplings(), **PARAMS}
    # 0.1.9: amplitude(h0, h1, h2, arguments, d1_mass, d2_mass)
    old = resonance.amplitude(1, 1, 0, arguments, 2.0, 3.0)
    new = resonance.amplitude(1, 1, 0, arguments, 6.0, 2.0, 3.0)
    assert np.allclose(old, new)
