"""
Integration test: decayamplitude with DecayShape lineshapes.

DecayShape Lineshape instances are automatically adapted to the decayamplitude
calling convention by _maybe_wrap_lineshape in resonance.py — no manual wrapper
needed. Tests verify both the auto-compat layer and full amplitude evaluation.

Skipped automatically if decayshape is not installed.
"""

import pytest
import numpy as np

decayshape = pytest.importorskip("decayshape")

# Configure decayangle backend BEFORE any decayangle/decayamplitude imports.
from decayangle.config import config as decayangle_config
decayangle_config.backend = "numpy"
decayangle_config.sorting = "value"

from decayamplitude.resonance import Resonance, _maybe_wrap_lineshape
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayangle.decay_topology import Topology, Node


def constant_lineshape(mass, l, s, *args):
    return 1.0


def make_momenta(n_events=None):
    """
    Phase-space momenta for B0 → π+(1) π-(2) π0(3) at rest.

    Returns single-event arrays (shape (4,)) when n_events is None,
    or batched arrays (shape (n_events, 4)) otherwise.
    """
    if n_events is None:
        return {
            1: np.array([0.12,  0.05,  0.48, 0.504]),
            2: np.array([-0.08, -0.03, -0.30, 0.322]),
            3: np.array([-0.04, -0.02, -0.18, 0.199]),
        }
    rng = np.random.default_rng(42)
    def rand_p(m):
        p3 = rng.normal(size=(n_events, 3)) * 0.3
        E  = np.sqrt((p3 ** 2).sum(-1) + m ** 2)
        return np.concatenate([p3, E[:, None]], axis=1)
    return {1: rand_p(0.140), 2: rand_p(0.140), 3: rand_p(0.135)}


# ── Auto-compat layer tests ────────────────────────────────────────────────────

def test_maybe_wrap_returns_callable_for_decayshape_instance():
    """_maybe_wrap_lineshape wraps a DecayShape instance to the decayamplitude convention."""
    from decayshape import RelativisticBreitWigner, Channel, CommonParticles

    channel = Channel(particle1=CommonParticles.PI_PLUS,
                      particle2=CommonParticles.PI_MINUS)
    rho = RelativisticBreitWigner(channel=channel, pole_mass=0.775, width=0.150, r=1.0)
    ls  = _maybe_wrap_lineshape(rho)

    mass_pts = np.linspace(0.40, 1.20, 50)
    vals = ls(mass_pts, 2, 2)   # decayamplitude convention: (mass, 2L, 2S)

    assert vals.shape == (50,)
    assert np.all(np.isfinite(np.abs(vals)))
    assert np.any(np.abs(vals) > 0)


def test_maybe_wrap_is_noop_for_plain_callable():
    """_maybe_wrap_lineshape is a no-op for ordinary Python callables."""
    def plain(mass, l, s, *args):
        return np.ones_like(mass)

    assert _maybe_wrap_lineshape(plain) is plain


def test_maybe_wrap_passes_daughter_masses():
    """Daughter masses propagate through the auto-compat layer into DecayShape."""
    from decayshape import RelativisticBreitWigner, Channel, CommonParticles

    channel = Channel(particle1=CommonParticles.PI_PLUS,
                      particle2=CommonParticles.PI_MINUS)
    rho = RelativisticBreitWigner(channel=channel)
    ls  = _maybe_wrap_lineshape(rho)

    mass_pts = np.linspace(0.40, 1.20, 20)
    val_default  = ls(mass_pts, 2, 2)
    val_override = ls(mass_pts, 2, 2, d1_mass=0.140, d2_mass=0.140)

    assert val_default.shape == val_override.shape
    assert np.all(np.isfinite(np.abs(val_default)))
    assert np.all(np.isfinite(np.abs(val_override)))


# ── Full amplitude integration tests ──────────────────────────────────────────

def test_three_body_amplitude_decayshape_no_free_params():
    """
    Three-body amplitude B0 → π+(1) π-(2) π0(3) with two ρ(770) topologies.
    DecayShape instances are passed directly — auto-compat wraps them.
    Lineshape parameters are fixed at DecayShape construction time (argnames=[]).
    """
    from decayshape import RelativisticBreitWigner, Channel, CommonParticles

    pion_channel = Channel(particle1=CommonParticles.PI_PLUS,
                           particle2=CommonParticles.PI_MINUS)
    rho = RelativisticBreitWigner(channel=pion_channel, pole_mass=0.775,
                                  width=0.150, r=1.0)

    final_state_qn = {
        1: QN(0, -1),   # π+
        2: QN(0, -1),   # π-
        3: QN(0, -1),   # π0
    }

    def mother(name):
        return [Resonance(Node(0), quantum_numbers=QN(0, -1),
                          lineshape=constant_lineshape, argnames=[],
                          preserve_partity=False, name=name)]

    resonances_12 = {
        (1, 2): [Resonance(Node((1, 2)), quantum_numbers=QN(2, -1),
                           lineshape=rho, argnames=[],
                           preserve_partity=True, name="rho0_12")],
        0: mother("B0_a"),
    }
    resonances_13 = {
        (1, 3): [Resonance(Node((1, 3)), quantum_numbers=QN(2, -1),
                           lineshape=rho, argnames=[],
                           preserve_partity=True, name="rho-_13")],
        0: mother("B0_b"),
    }

    topology1 = Topology(0, decay_topology=((1, 2), 3))
    topology2 = Topology(0, decay_topology=((1, 3), 2))

    chain1 = MultiChain(topology=topology1, resonances=resonances_12,
                        final_state_qn=final_state_qn)
    chain2 = MultiChain(topology=topology2, resonances=resonances_13,
                        final_state_qn=final_state_qn)

    combined = ChainCombiner([chain1, chain2])
    unpolarized, param_names = combined.unpolarized_amplitude(
        combined.generate_couplings()
    )

    momenta = topology1.to_rest_frame(make_momenta())
    start  = {name: 1.0 for name in param_names if name != "momenta"}
    result = unpolarized(momenta, **start)

    assert np.isfinite(float(np.abs(result)))
    assert float(np.abs(result)) > 0


def test_three_body_amplitude_decayshape_free_params():
    """
    Same decay with lineshape parameters exposed as free fit parameters
    (argnames=["rho_mass", "rho_width", "rho_r"]).
    """
    from decayshape import RelativisticBreitWigner, Channel, CommonParticles

    pion_channel = Channel(particle1=CommonParticles.PI_PLUS,
                           particle2=CommonParticles.PI_MINUS)
    rho = RelativisticBreitWigner(channel=pion_channel)

    final_state_qn = {1: QN(0, -1), 2: QN(0, -1), 3: QN(0, -1)}

    def mother(name):
        return [Resonance(Node(0), quantum_numbers=QN(0, -1),
                          lineshape=constant_lineshape, argnames=[],
                          preserve_partity=False, name=name)]

    resonances_12 = {
        (1, 2): [Resonance(Node((1, 2)), quantum_numbers=QN(2, -1),
                           lineshape=rho,
                           argnames=["rho_mass", "rho_width", "rho_r"],
                           preserve_partity=True, name="rho0_12")],
        0: mother("B0_a"),
    }

    topology = Topology(0, decay_topology=((1, 2), 3))
    chain    = MultiChain(topology=topology, resonances=resonances_12,
                          final_state_qn=final_state_qn)
    combined = ChainCombiner([chain])
    unpolarized, param_names = combined.unpolarized_amplitude(
        combined.generate_couplings()
    )

    assert any("rho_mass" in n for n in param_names)
    assert any("rho_width" in n for n in param_names)

    momenta = topology.to_rest_frame(make_momenta())
    start   = {name: 1.0 for name in param_names if name != "momenta"}
    result  = unpolarized(momenta, **start)

    assert np.isfinite(float(np.abs(result)))


def test_three_body_amplitude_decayshape_batched():
    """Amplitude evaluates correctly over a batch of events."""
    from decayshape import RelativisticBreitWigner, Channel, CommonParticles

    pion_channel = Channel(particle1=CommonParticles.PI_PLUS,
                           particle2=CommonParticles.PI_MINUS)
    rho = RelativisticBreitWigner(channel=pion_channel, pole_mass=0.775,
                                  width=0.150, r=1.0)

    final_state_qn = {1: QN(0, -1), 2: QN(0, -1), 3: QN(0, -1)}

    resonances = {
        (1, 2): [Resonance(Node((1, 2)), quantum_numbers=QN(2, -1),
                           lineshape=rho, argnames=[],
                           preserve_partity=True, name="rho0")],
        0: [Resonance(Node(0), quantum_numbers=QN(0, -1),
                      lineshape=constant_lineshape, argnames=[],
                      preserve_partity=False, name="B0")],
    }

    topology = Topology(0, decay_topology=((1, 2), 3))
    chain    = MultiChain(topology=topology, resonances=resonances,
                          final_state_qn=final_state_qn)
    combined = ChainCombiner([chain])
    unpolarized, param_names = combined.unpolarized_amplitude(
        combined.generate_couplings()
    )

    n_events = 50
    momenta  = topology.to_rest_frame(make_momenta(n_events=n_events))
    start    = {name: 1.0 for name in param_names if name != "momenta"}
    result   = unpolarized(momenta, **start)

    assert result.shape == (n_events,)
    assert np.all(np.isfinite(np.abs(result)))
    assert np.any(np.abs(result) > 0)


if __name__ == "__main__":
    test_maybe_wrap_returns_callable_for_decayshape_instance()
    test_maybe_wrap_is_noop_for_plain_callable()
    test_maybe_wrap_passes_daughter_masses()
    test_three_body_amplitude_decayshape_no_free_params()
    test_three_body_amplitude_decayshape_free_params()
    test_three_body_amplitude_decayshape_batched()
    print("All DecayShape integration tests passed.")
