"""Instrument the amplitude construction to count how many scalar terms get
traced for the 12-resonance / 3-topology benchmark model, without paying for
a full jit/compile cycle."""
import warnings
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"
decayangle_config.use_rust = False
decayangle_config.sorting = "off"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import decayamplitude.rotation as rotation_mod
import decayamplitude.resonance as resonance_mod

counts = {"wigner_capital_d": 0, "helicity_from_ls": 0, "resonance_amplitude": 0}

_orig_wigner = rotation_mod.wigner_capital_d.__wrapped__ if hasattr(rotation_mod.wigner_capital_d, "__wrapped__") else None

_orig_helicity_from_ls = resonance_mod.Resonance.helicity_from_ls
def counted_helicity_from_ls(self, *a, **k):
    counts["helicity_from_ls"] += 1
    return _orig_helicity_from_ls(self, *a, **k)
resonance_mod.Resonance.helicity_from_ls = counted_helicity_from_ls

_orig_amplitude = resonance_mod.Resonance.amplitude
def counted_amplitude(self, *a, **k):
    counts["resonance_amplitude"] += 1
    return _orig_amplitude(self, *a, **k)
resonance_mod.Resonance.amplitude = counted_amplitude

_orig_wcd = rotation_mod.wigner_capital_d
def counted_wcd(*a, **k):
    counts["wigner_capital_d"] += 1
    return _orig_wcd(*a, **k)
rotation_mod.wigner_capital_d = counted_wcd
import decayamplitude.chain as chain_mod
chain_mod.wigner_capital_d = counted_wcd

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayangle.decay_topology import Topology, Node

def bw(mass, l, s, m0, gamma):
    return 1.0 / (mass**2 - m0**2 + 1j * mass * gamma)

def const_ls(mass, l, s, *a):
    return 1.0

FINAL_STATE_QN = {1: QN(1, 1), 2: QN(2, -1), 3: QN(2, -1)}
ROOT_QN = QN(1, 1)
TOPO_DEFS = {(2, 3): ((2, 3), 1), (1, 3): ((1, 3), 2), (1, 2): ((1, 2), 3)}
SPIN_SETS = {(2, 3): [0, 2, 4, 6], (1, 3): [1, 3, 5, 7], (1, 2): [1, 3, 5, 7]}

chains = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for node_tuple, decay_topology in TOPO_DEFS.items():
        topology = Topology(0, decay_topology=decay_topology)
        tag = "".join(str(n) for n in node_tuple)
        spins = SPIN_SETS[node_tuple]
        isobars = [
            Resonance(Node(node_tuple), quantum_numbers=QN(spin2, 1 if i % 2 == 0 else -1),
                      lineshape=bw, argnames=[f"m0_{tag}_{spin2}", f"gamma_{tag}_{spin2}"],
                      preserve_partity=True, name=f"R{tag}_{spin2}")
            for i, spin2 in enumerate(spins)
        ]
        root = [Resonance(Node(0), quantum_numbers=ROOT_QN, lineshape=const_ls, argnames=[],
                           preserve_partity=False, name=f"root_{tag}")]
        resonances = {node_tuple: isobars, 0: root}
        chains.append(MultiChain(topology=topology, resonances=resonances, final_state_qn=FINAL_STATE_QN))

combiner = ChainCombiner(chains)
couplings = combiner.generate_couplings()
unpolarized, param_names = combiner.unpolarized_amplitude(couplings, complex_couplings=True)
coupling_names = [p for p in param_names if p != "momenta"]

import numpy as onp
rng = onp.random.default_rng(0)
def rand_p(mass, n):
    p3 = rng.normal(size=(n, 3)) * 0.3
    E = onp.sqrt((p3 ** 2).sum(-1) + mass ** 2)
    return jnp.array(onp.concatenate([p3, E[:, None]], axis=1))

M1, M2, M3 = 0.938272, 0.89166, 0.77526
momenta = {1: rand_p(M1, 3), 2: rand_p(M2, 3), 3: rand_p(M3, 3)}
momenta = chains[0].topology.to_rest_frame(momenta)
param_vals = tuple(1.0 for _ in coupling_names)

# a single eager call traces every python-level term exactly once
_ = unpolarized(momenta, *param_vals)

print("Call counts for ONE evaluation of unpolarized(momenta, *params):")
for k, v in counts.items():
    print(f"  {k:<22s}: {v}")
print(f"\n(these counts are independent of N_EVENTS -- each call becomes one traced scalar op per event-batch)")
