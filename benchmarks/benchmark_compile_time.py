"""
Benchmark: XLA compile time for an arbitrary three-body decay with 12 resonances.

This is the "gold standard" benchmark for compile-time optimization work.
It builds a generic three-body decay 0 -> 1 2 3 with all three possible
sub-topologies populated:

    (2,3)+1   4 resonances (integer spin 0,2,4,6 / value2)
    (1,3)+2   4 resonances (half-integer spin 1,3,5,7 / value2)
    (1,2)+3   4 resonances (half-integer spin 1,3,5,7 / value2)

combined via ChainCombiner (Wigner-rotation alignment across all three
topologies), for a total of 12 resonances and >100 free LS-coupling
parameters. 100k phase-space events are used for the execution timing.

All three final-state particles carry nonzero spin (1/2, 1, 1) rather than
leaving any at spin-0. A spin-0 final-state particle has exactly one
helicity state, which makes its contribution to the ChainCombiner alignment
step (the sum over final-state helicity combinations in aligned_matrix)
trivial. With every particle spin-full there are 2x3x3=18 helicity
combinations instead of 2, so the benchmark is actually sensitive to the
cost of the final-state Wigner-rotation alignment machinery, not just the
per-chain amplitude recursion.

Compile time is split into two phases using JAX's AOT API:
    trace   : jax.jit(f).lower(...)   Python tracing -> StableHLO
    compile : lowered.compile()       StableHLO -> XLA executable

The second number is "XLA compile time" in the strict sense and is the
number this benchmark is meant to drive down.

Usage
-----
    python3 benchmarks/benchmark_compile_time.py [n_events]
"""

import sys
import time
import warnings

# Must configure backend BEFORE any decayangle/decayamplitude imports,
# because decay_topology.py binds `cb = cfg.backend` at module load time.
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"
decayangle_config.use_rust = False
decayangle_config.sorting = "off"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayangle.decay_topology import Topology, Node

N_EVENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

# 0 -> 1 2 3, roughly Lambda_b0 -> p K*(892)- rho(770)0 masses (GeV).
# Mother bumped up to a Lambda_b-like mass since all three daughters are now
# spin-full vector/baryon states (heavier than the pseudoscalar K/pi used
# previously), and the sum of daughter masses must stay below it.
MOTHER_MASS = 5.61951
M1, M2, M3 = 0.938272, 0.89166, 0.77526

FINAL_STATE_QN = {
    1: QN(1, 1),    # spin-1/2, parity +   (baryon, e.g. proton)
    2: QN(2, -1),   # spin-1,   parity -   (vector meson, e.g. K*(892))
    3: QN(2, -1),   # spin-1,   parity -   (vector meson, e.g. rho(770))
}
ROOT_QN = QN(1, 1)  # spin-1/2 mother, weak decay -> parity not conserved

TOPO_DEFS = {
    (2, 3): ((2, 3), 1),
    (1, 3): ((1, 3), 2),
    (1, 2): ((1, 2), 3),
}
# spin2 sequences chosen so every entry yields a valid (non-empty) LS coupling
# set for its node: integer spins for the vector-vector isobar (2,3), half-integer
# spins for the two baryon-vector isobars (1,3) and (1,2).
SPIN_SETS = {
    (2, 3): [0, 2, 4, 6],
    (1, 3): [1, 3, 5, 7],
    (1, 2): [1, 3, 5, 7],
}


def bw(mass, l, s, m0, gamma):
    return 1.0 / (mass**2 - m0**2 + 1j * mass * gamma)


def const_ls(mass, l, s, *a):
    return 1.0


def build_combiner() -> ChainCombiner:
    chains = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for node_tuple, decay_topology in TOPO_DEFS.items():
            topology = Topology(0, decay_topology=decay_topology)
            tag = "".join(str(n) for n in node_tuple)
            spins = SPIN_SETS[node_tuple]
            isobars = [
                Resonance(
                    Node(node_tuple),
                    quantum_numbers=QN(spin2, 1 if i % 2 == 0 else -1),
                    lineshape=bw,
                    argnames=[f"m0_{tag}_{spin2}", f"gamma_{tag}_{spin2}"],
                    preserve_partity=True,
                    name=f"R{tag}_{spin2}",
                )
                for i, spin2 in enumerate(spins)
            ]
            root = [
                Resonance(
                    Node(0), quantum_numbers=ROOT_QN, lineshape=const_ls, argnames=[],
                    preserve_partity=False, name=f"root_{tag}",
                )
            ]
            resonances = {node_tuple: isobars, 0: root}
            chains.append(MultiChain(topology=topology, resonances=resonances, final_state_qn=FINAL_STATE_QN))
    return ChainCombiner(chains)


def generate_events(n: int) -> dict:
    try:
        import phasespace
        weights, particles = phasespace.nbody_decay(MOTHER_MASS, [M1, M2, M3]).generate(n_events=n)
        return {
            1: jnp.array(particles["p_0"]),
            2: jnp.array(particles["p_1"]),
            3: jnp.array(particles["p_2"]),
        }
    except ImportError:
        import numpy as onp
        rng = onp.random.default_rng(42)

        def rand_p(mass):
            p3 = rng.normal(size=(n, 3)).astype("float64") * 0.3
            E = onp.sqrt((p3**2).sum(-1) + mass**2)
            return jnp.array(onp.concatenate([p3, E[:, None]], axis=1))

        return {1: rand_p(M1), 2: rand_p(M2), 3: rand_p(M3)}


def main():
    print(f"N_EVENTS = {N_EVENTS:,}", flush=True)

    print("Building amplitude (3 topologies x 4 resonances = 12 resonances) ...", flush=True)
    t0 = time.perf_counter()
    combiner = build_combiner()
    couplings = combiner.generate_couplings()
    unpolarized, param_names = combiner.unpolarized_amplitude(couplings, complex_couplings=True)
    coupling_names = [p for p in param_names if p != "momenta"]
    n_params = len(coupling_names)
    print(f"  model built in {time.perf_counter() - t0:.2f} s | {n_params} free LS-coupling parameters", flush=True)

    print(f"Generating {N_EVENTS:,} phase-space events ...", flush=True)
    raw_momenta = generate_events(N_EVENTS)
    topology_ref = combiner.reference.topology
    momenta = topology_ref.to_rest_frame(raw_momenta)

    param_vals = tuple(1.0 for _ in coupling_names)
    event_axes = {k: 0 for k in momenta}

    def f_1e(momenta_1e, *params):
        return unpolarized(momenta_1e, *params)

    f_batched = jax.vmap(f_1e, in_axes=(event_axes,) + (None,) * n_params)
    jitted = jax.jit(f_batched)

    print("\n── XLA compile-time breakdown ──────────────────────────────────────────", flush=True)

    t0 = time.perf_counter()
    lowered = jitted.lower(momenta, *param_vals)
    t_trace = time.perf_counter() - t0
    print(f"  trace        (Python -> jaxpr/StableHLO):  {t_trace * 1e3:9.1f} ms", flush=True)

    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_xla_compile = time.perf_counter() - t0
    print(f"  XLA compile  (StableHLO -> executable):    {t_xla_compile * 1e3:9.1f} ms", flush=True)

    t0 = time.perf_counter()
    r = compiled(momenta, *param_vals)
    jax.block_until_ready(r)
    t_first_exec = time.perf_counter() - t0
    print(f"  first execution (dispatch + run):          {t_first_exec * 1e3:9.1f} ms", flush=True)

    print(f"  {'─' * 60}")
    print(f"  total (trace + compile + first exec):      {(t_trace + t_xla_compile + t_first_exec) * 1e3:9.1f} ms")

    reps = 5
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = compiled(momenta, *param_vals)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)
    t_exec = sum(times) / len(times)
    print(f"  steady-state exec (avg of {reps}):             {t_exec * 1e3:9.1f} ms")

    print("\n── graph size diagnostics ──────────────────────────────────────────────", flush=True)
    jaxpr = jax.make_jaxpr(f_batched)(momenta, *param_vals)
    n_eqns = sum(1 for _ in jaxpr.jaxpr.eqns)
    print(f"  jaxpr equations:  {n_eqns}")
    try:
        hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        n_hlo_lines = hlo_text.count("\n")
        n_hlo_instr = hlo_text.count(" = ")
        print(f"  HLO text lines:   {n_hlo_lines}")
        print(f"  HLO instructions (approx, ' = ' count): {n_hlo_instr}")
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"  (could not retrieve HLO text: {e})")

    print("\n── summary ──────────────────────────────────────────────────────────────")
    print(f"  resonances: 12  |  topologies: 3  |  fit params: {n_params}  |  N_EVENTS: {N_EVENTS:,}")
    print(f"  trace: {t_trace * 1e3:.1f} ms   xla_compile: {t_xla_compile * 1e3:.1f} ms   "
          f"first_exec: {t_first_exec * 1e3:.1f} ms   steady_exec: {t_exec * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
