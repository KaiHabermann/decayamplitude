"""Regression tests for the `static_momenta` mode of unpolarized_amplitude /
polarized_amplitude / matrix_function.

static_momenta bakes a fixed momenta set into a compiled function via an
eagerly-precomputed cache of momenta-only quantities (helicity angles,
per-node masses, per-node Wigner-D factors, alignment rotations). This is
easy to get subtly wrong in two specific ways that the tests below target
directly:

1. The precomputed cache must be keyed so that it's actually *hit* by the
   amplitude recursion at call time (not silently dead, forcing a live
   recompute every call -- see test_node_tree_identity).
2. Building a static_momenta function must never mutate state shared with
   any other function -- static or not -- built from the same chain/combiner
   (see the *_does_not_corrupt_* tests below).
"""
from decayamplitude.backend import numpy as np
from decayamplitude.chain import DecayChain, MultiChain, AlignedChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.kinematics_helpers import mass_from_node

from decayangle.decay_topology import Topology, Node
from decayangle.lorentz import LorentzTrafo
from decayangle.config import config as decayangle_config


def constant_lineshape(*args):
    return 1


def make_four_vectors(phi_rf, theta_rf, psi_rf):
    # Lc -> p K pi, same reference kinematics as test_threebody.py
    m0 = 6.32397
    m12 = 9.55283383**0.5
    m23 = 26.57159046**0.5
    m13 = 17.86811729**0.5
    m1, m2, m3 = 1, 2, 3
    m0sq, m1sq, m2sq, m3sq, m12sq, m23sq = [x**2 for x in [m0, m1, m2, m3, m12, m23]]

    def Kallen(x, y, z):
        return x**2 + y**2 + z**2 - 2 * (x * y + x * z + y * z)

    m31sq = m0sq + m1sq + m2sq + m3sq - m12sq - m23sq

    p1a = np.sqrt(Kallen(m23sq, m1sq, m0sq)) / (2 * m0)
    p2a = np.sqrt(Kallen(m31sq, m2sq, m0sq)) / (2 * m0)

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

    E1 = np.sqrt(p1z**2 + m1sq)
    E2 = np.sqrt(p2z**2 + p2x**2 + m2sq)
    E3 = np.sqrt(p3z**2 + p3x**2 + m3sq)

    p1 = np.array([0, 0, p1z, E1])
    p2 = np.array([p2x, 0, p2z, E2])
    p3 = np.array([p3x, 0, p3z, E3])

    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))
    rotation = LorentzTrafo(0, 0, 0, phi_rf, theta_rf, psi_rf)
    return tree1.root.transform(rotation, momenta)


def make_threebody_chain(spinful=False):
    """A simple 3-body chain: 0 -> (2,3) 1, with a resonance in the (2,3) isobar."""
    decayangle_config.sorting = "off"
    topology = Topology(0, decay_topology=((2, 3), 1))
    final_state_qn = {1: QN(1, 1), 2: QN(2, 1), 3: QN(0, 1)}
    resonances = {
        (2, 3): Resonance(Node((2, 3)), 4 if spinful else 0, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[]),
    }
    chain = DecayChain(topology=topology, resonances=resonances, final_state_qn=final_state_qn)
    return chain, final_state_qn


def make_fourbody_combiner():
    """Two-topology, MultiChain-based ChainCombiner, mirroring test_chains.test_multi_chain."""
    final_state_qn = {
        1: QN(0, 1),
        2: QN(0, 1),
        3: QN(1, 1),
        4: QN(1, -1),
    }
    resonances_hadronic = {
        (1, 2): [
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=True, name="Resonance1"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=True, name="Resonance2"),
        ],
        (3, 4): [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance3")],
        (1, 2, 3): [Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance4")],
        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }
    topology1 = Topology(0, decay_topology=((1, 2), (3, 4)))
    topology2 = Topology(0, decay_topology=(((1, 2), 3), 4))

    chain1 = MultiChain(topology=topology1, resonances=resonances_hadronic, final_state_qn=final_state_qn)
    chain2 = MultiChain(topology=topology2, resonances=resonances_hadronic, final_state_qn=final_state_qn)
    combiner = ChainCombiner([chain1, chain2])

    def make_momenta(scale):
        raw = {
            1: np.array([1.0, 0.1, 0.4, 3]) * scale,
            2: np.array([0.5, -0.1, -0.4, 3]) * scale,
            3: np.array([1.1, 0.2, 0.5, 3]) * scale,
            4: np.array([0.6, -0.2, -0.5, 3]) * scale,
        }
        return topology1.to_rest_frame(raw)

    return combiner, make_momenta


# ---------------------------------------------------------------------------
# Node-tree identity (regression for the dead per-node static cache)
# ---------------------------------------------------------------------------

def _collect_root_tree_ids(root):
    ids = set()
    stack = [root]
    while stack:
        node = stack.pop()
        ids.add(id(node))
        stack.extend(node.daughters)
    return ids


def test_node_tree_identity_decay_chain():
    """chain.nodes must be object-identical to the tree chain.root.amplitude()
    actually recurses through, or the static-momenta per-node cache (keyed by
    id(node)) is silently dead and every lookup falls back to a live recompute.
    """
    chain, _ = make_threebody_chain()
    root_ids = _collect_root_tree_ids(chain.root)
    nodes_ids = {id(n) for n in chain.nodes}
    assert root_ids == nodes_ids
    assert len(root_ids) == len(chain.topology.nodes)


def test_node_tree_identity_multi_chain_siblings():
    combiner, _ = make_fourbody_combiner()
    for chain in combiner.single_chains:
        root_ids = _collect_root_tree_ids(chain.root)
        nodes_ids = {id(n) for n in chain.nodes}
        assert root_ids == nodes_ids


def test_static_cache_keys_are_hit_at_runtime():
    """The masses/wigner_d entries _static_cache precomputes must be keyed by
    ids that DecayChainNode.amplitude actually looks up -- i.e. every
    non-final-state node's id must appear in both.
    """
    chain, _ = make_threebody_chain()
    momenta = make_four_vectors(0.3, np.arccos(0.4), 0.5)
    static_entries = chain._static_cache(momenta)

    internal_node_ids = {id(n) for n in chain.nodes if not n.final_state}
    mass_keyed_ids = {k[0] for k in static_entries if isinstance(k, tuple) and len(k) == 2 and k[1] == "masses"}
    assert mass_keyed_ids == internal_node_ids
    assert len(internal_node_ids) > 0


# ---------------------------------------------------------------------------
# static_momenta correctness: must match the non-static result for the same momenta
# ---------------------------------------------------------------------------

def test_decay_chain_static_matches_non_static():
    chain, _ = make_threebody_chain(spinful=True)
    momenta = make_four_vectors(0.3, np.arccos(0.4), 0.5)
    couplings = chain.generate_couplings()

    func, argnames = chain.unpolarized_amplitude(couplings, complex_couplings=False)
    n_coupling = len(argnames) - 1
    params = tuple([1.3] * n_coupling)
    expected = func(momenta, *params)

    static_func, static_argnames = chain.unpolarized_amplitude(couplings, complex_couplings=False, static_momenta=momenta)
    assert len(static_argnames) == n_coupling
    result = static_func(*params)

    assert np.allclose(result, expected)


def test_chain_combiner_unpolarized_amplitude_static_matches_non_static():
    combiner, make_momenta = make_fourbody_combiner()
    momenta = make_momenta(1.0)
    couplings = combiner.generate_couplings()

    func, argnames = combiner.unpolarized_amplitude(couplings, complex_couplings=False)
    n_coupling = len(argnames) - 1
    params = tuple([1.0] * n_coupling)
    expected = func(momenta, *params)

    static_func, static_argnames = combiner.unpolarized_amplitude(couplings, complex_couplings=False, static_momenta=momenta)
    result = static_func(*params)

    assert np.allclose(result, expected)


def test_chain_combiner_matrix_function_static_matches_non_static():
    combiner, make_momenta = make_fourbody_combiner()
    momenta = make_momenta(1.0)
    couplings = combiner.generate_couplings()

    func, argnames = combiner.matrix_function(couplings, complex_couplings=False)
    n_coupling = len(argnames) - 2  # momenta, h0
    h0 = 0
    params = tuple([1.0] * n_coupling)
    expected = func(momenta, h0, *params)

    static_func, static_argnames = combiner.matrix_function(couplings, complex_couplings=False, static_momenta=momenta)
    result = static_func(h0, *params)

    assert set(expected.keys()) == set(result.keys())
    for key in expected:
        assert np.allclose(expected[key], result[key])


def test_chain_combiner_polarized_amplitude_static_matches_non_static():
    combiner, make_momenta = make_fourbody_combiner()
    momenta = make_momenta(1.0)
    couplings = combiner.generate_couplings()

    # h0 must be a valid projection of the root resonance's spin, and each
    # h_<n> a valid projection of that final-state particle's spin (both in
    # value2 convention) -- e.g. 0 is invalid for the spin-1/2 particles here.
    sorted_final_state_nodes = sorted(n.node.value for n in combiner.reference.final_state_nodes)
    h0 = combiner.root_resonance.quantum_numbers.angular.projections(return_int=True)[0]
    h_finals = tuple(
        combiner.reference.final_state_qn[node].angular.projections(return_int=True)[0]
        for node in sorted_final_state_nodes
    )

    func, helicity_names, coupling_names = combiner.polarized_amplitude(couplings, complex_couplings=False)
    params = tuple([1.0] * len(coupling_names))
    expected = func(momenta, h0, *h_finals, *params)

    static_func, static_helicity_names, static_coupling_names = combiner.polarized_amplitude(
        couplings, complex_couplings=False, static_momenta=momenta
    )
    result = static_func(h0, *h_finals, *params)

    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# Cross-contamination regressions: building a static_momenta function must not
# affect any other function built from the same chain/combiner.
# ---------------------------------------------------------------------------

def test_static_build_does_not_corrupt_existing_non_static_decay_chain_function():
    chain, _ = make_threebody_chain(spinful=True)
    momenta_A = make_four_vectors(0.3, np.arccos(0.4), 0.5)
    momenta_B = make_four_vectors(1.1, np.arccos(-0.2), 2.0)
    couplings = chain.generate_couplings()

    func, argnames = chain.unpolarized_amplitude(couplings, complex_couplings=False)
    n_coupling = len(argnames) - 1
    params = tuple([1.3] * n_coupling)

    result_before = func(momenta_A, *params)

    # Building a second, static function off the SAME chain must not mutate
    # any state the first (non-static) function reads.
    chain.unpolarized_amplitude(couplings, complex_couplings=False, static_momenta=momenta_B)

    result_after = func(momenta_A, *params)
    assert np.allclose(result_before, result_after)


def test_two_static_builds_from_same_decay_chain_do_not_interfere():
    chain, _ = make_threebody_chain(spinful=True)
    momenta_A = make_four_vectors(0.3, np.arccos(0.4), 0.5)
    momenta_B = make_four_vectors(1.1, np.arccos(-0.2), 2.0)
    couplings = chain.generate_couplings()

    non_static_func, non_static_argnames = chain.unpolarized_amplitude(couplings, complex_couplings=False)
    n_coupling = len(non_static_argnames) - 1
    params = tuple([1.3] * n_coupling)
    expected_A = non_static_func(momenta_A, *params)
    expected_B = non_static_func(momenta_B, *params)

    static_func_A, _ = chain.unpolarized_amplitude(couplings, complex_couplings=False, static_momenta=momenta_A)
    result_A_first = static_func_A(*params)

    # Build a second static function for different momenta off the same chain.
    static_func_B, _ = chain.unpolarized_amplitude(couplings, complex_couplings=False, static_momenta=momenta_B)
    result_B = static_func_B(*params)

    # The FIRST static function must still be correct after the second build.
    result_A_second = static_func_A(*params)

    assert np.allclose(result_A_first, expected_A)
    assert np.allclose(result_A_second, expected_A)
    assert np.allclose(result_B, expected_B)


def test_static_build_does_not_corrupt_existing_non_static_combiner_function():
    combiner, make_momenta = make_fourbody_combiner()
    momenta_A = make_momenta(1.0)
    momenta_B = make_momenta(1.7)
    couplings = combiner.generate_couplings()

    func, argnames = combiner.unpolarized_amplitude(couplings, complex_couplings=False)
    n_coupling = len(argnames) - 1
    params = tuple([1.0] * n_coupling)

    result_before = func(momenta_A, *params)

    combiner.unpolarized_amplitude(couplings, complex_couplings=False, static_momenta=momenta_B)

    result_after = func(momenta_A, *params)
    assert np.allclose(result_before, result_after)


def test_static_build_does_not_corrupt_aligned_chain():
    """AlignedChain shares the same _static_cache/momenta_cache machinery --
    exercise it directly (not just via ChainCombiner) since it overrides
    _static_cache and aligned_matrix.
    """
    topology1 = Topology(0, decay_topology=((2, 3), 1))
    topology2 = Topology(0, decay_topology=((1, 2), 3))
    final_state_qn = {1: QN(1, 1), 2: QN(2, 1), 3: QN(0, 1)}

    resonances_dpd = {
        (2, 3): Resonance(Node((2, 3)), 4, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[]),
    }
    resonances3 = {
        (1, 2): Resonance(Node((1, 2)), 1, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[]),
    }

    reference = DecayChain(topology=topology1, resonances=resonances_dpd, final_state_qn=final_state_qn)
    aligned = AlignedChain(
        topology=topology2, resonances=resonances3, final_state_qn=final_state_qn, reference=reference
    )

    momenta_A = make_four_vectors(0.3, np.arccos(0.4), 0.5)
    momenta_B = make_four_vectors(1.1, np.arccos(-0.2), 2.0)

    args = {
        aligned.resonances[(1, 2)].id: {"couplings": {(2, 3): 1}},
        aligned.resonances[0].id: {"couplings": {(2, 1): 1}},
    }
    m = aligned.aligned_matrix
    expected_A = m(-1, args, momenta_A)

    # A static build for DIFFERENT momenta, off the same AlignedChain, must
    # not affect subsequent non-static calls on that same chain.
    static_cache = aligned._static_cache(momenta_B)
    m(-1, args, momenta_B, momenta_cache=static_cache)

    result_A_after = m(-1, args, momenta_A)

    for key in expected_A:
        assert np.allclose(expected_A[key], result_A_after[key])
