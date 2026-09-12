"""
Regression harness for n-body (3, 4, 5) unpolarized amplitudes.

This module builds a simple "staircase" decay chain for n final-state
particles (0 -> 1 + (2 + (3 + (... + n))) ), with one constant (spin-0)
resonance per subsystem, and generates random phase-space momenta for it
using the `phasespace` package.

Run this file directly to (re-)generate the momenta json files under
tests/test_data/. test_nbody_regression below then checks that the
unpolarized amplitude computed from that saved momenta still matches the
saved reference values in tests/test_data/nbody_{n}_expected.json.
"""

import os
import json

import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_disable_hlo_passes=constant_folding")
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "True")

from decayangle.config import config as decayangle_config

decayangle_config.backend = "jax"
decayangle_config.use_rust = False
decayangle_config.sorting = "value"

from decayangle.decay_topology import Topology, Node

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import DecayChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.backend import numpy as np

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

MOTHER_MASS = 2.0
DAUGHTER_MASSES = [0.5, 0.4, 0.3, 0.2, 0.1]


def constant_lineshape(L, S, *args):
    # Dummy lineshape, models a stable, constant (spin-0) resonance.
    return 1.0


def staircase_topology(n: int) -> Topology:
    """Build the nested topology 0 -> 1 + (2 + (3 + (... + n))).

    For n=3: ((1, 2), 3) style nesting -> here (1, (2, 3))
    For n=4: (1, (2, (3, 4)))
    For n=5: (1, (2, (3, (4, 5))))
    """
    decay_topology = (n - 1, n)
    for i in range(n - 2, 0, -1):
        decay_topology = (i, decay_topology)
    return Topology(0, decay_topology=decay_topology)


def subsystem_nodes(n: int) -> list[tuple]:
    """The non-final-state, non-root node tuples of the staircase topology."""
    return [tuple(range(i, n + 1)) for i in range(2, n)]


def build_chain(n: int) -> ChainCombiner:
    """Build a ChainCombiner around a single staircase DecayChain for n final
    state particles, with one constant spin-0 resonance per subsystem
    (including the root)."""
    topology = staircase_topology(n)
    # Particle 1 (and thus the mother) carries spin 1 so the couplings admit
    # more than one L, S term and the amplitude genuinely depends on the
    # decay angles rather than trivially reducing to a constant. All other
    # final-state particles and subsystems are spin 0.
    final_state_qn = {i: QN(0, 1) for i in range(1, n + 1)}
    final_state_qn[1] = QN(2, 1)

    resonances = {
        0: Resonance(
            Node(0),
            quantum_numbers=QN(2, 1),
            lineshape=constant_lineshape,
            argnames=[],
            preserve_partity=False,
            name="mother",
        )
    }
    for node in subsystem_nodes(n):
        resonances[node] = Resonance(
            Node(node),
            quantum_numbers=QN(0, 1),
            lineshape=constant_lineshape,
            argnames=[],
            preserve_partity=True,
            name=f"R_{''.join(map(str, node))}",
        )

    chain = DecayChain(topology=topology, resonances=resonances, final_state_qn=final_state_qn)
    return ChainCombiner([chain])


def generate_momenta(n: int, n_events: int = 100, seed: int = 42) -> dict:
    """Generate n_events unweighted phase-space points for mother -> 1..n
    using the `phasespace` package. Returns {i: (n_events, 4) array} with
    columns [px, py, pz, E].
    """
    import phasespace
    import numpy as onp

    daughter_masses = DAUGHTER_MASSES[:n]
    generator = phasespace.nbody_decay(MOTHER_MASS, daughter_masses)

    accepted = {i: [] for i in range(1, n + 1)}
    n_accepted = 0
    batch_seed = seed
    while n_accepted < n_events:
        batch_size = max(4 * (n_events - n_accepted), 200)
        weights, particles = generator.generate(n_events=batch_size, seed=batch_seed)
        batch_seed += 1
        weights = onp.asarray(weights)
        max_weight = weights.max()
        u = onp.random.RandomState(batch_seed).uniform(size=weights.shape)
        keep = u < (weights / max_weight)
        for i in range(1, n + 1):
            values = onp.asarray(particles[f"p_{i - 1}"])[keep]
            n_take = min(len(values), n_events - n_accepted)
            accepted[i].append(values[:n_take])
        n_accepted += min(int(keep.sum()), n_events - n_accepted)

    return {i: onp.concatenate(accepted[i], axis=0)[:n_events] for i in range(1, n + 1)}


def save_momenta_json(n: int, momenta: dict) -> str:
    path = os.path.join(TEST_DATA_DIR, f"nbody_{n}_momenta.json")
    with open(path, "w") as f:
        json.dump({str(k): v.tolist() for k, v in momenta.items()}, f)
    return path


def load_momenta_json(n: int) -> dict:
    path = os.path.join(TEST_DATA_DIR, f"nbody_{n}_momenta.json")
    with open(path) as f:
        raw = json.load(f)
    import numpy as onp

    return {int(k): onp.array(v) for k, v in raw.items()}


def unpolarized_amplitude_for(n: int, momenta: dict):
    """Build the n-body chain and evaluate the unpolarized amplitude at the
    given momenta, with all couplings set to 1."""
    full = build_chain(n)
    unpolarized, param_names = full.unpolarized_amplitude(full.generate_couplings(), complex_couplings=False)
    start_params = {name: 1.0 for name in param_names if name != "momenta"}
    values = unpolarized(momenta, **start_params)
    return np.asarray(values)


def load_expected_json(n: int):
    import numpy as onp

    path = os.path.join(TEST_DATA_DIR, f"nbody_{n}_expected.json")
    with open(path) as f:
        return onp.array(json.load(f))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_nbody_regression(n):
    momenta = load_momenta_json(n)
    expected = load_expected_json(n)
    result = unpolarized_amplitude_for(n, momenta)
    assert np.allclose(result, expected)


if __name__ == "__main__":
    import numpy as onp

    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    for n in (3, 4, 5):
        momenta = generate_momenta(n, n_events=100)
        path = save_momenta_json(n, momenta)
        values = unpolarized_amplitude_for(n, momenta)
        print(f"n={n}: saved momenta to {path}, unpolarized amplitude sample = {onp.asarray(values)[:5]}")
