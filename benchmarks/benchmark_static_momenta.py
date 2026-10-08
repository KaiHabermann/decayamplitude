"""
Benchmark: XLA compile time for the same "gold standard" 12-resonance,
3-topology, all-spinful-final-state model as benchmark_compile_time.py, but
built with static_momenta enabled instead of the default variable-momenta
path.

See benchmark_compile_time.py for the model definition and the rationale for
using all-spinful final-state particles. This script reuses that model
verbatim (build_combiner / generate_events) so the two benchmarks are
directly comparable.

With static_momenta, momenta is baked in as a fixed dataset: every
momenta-only computation (helicity angles, masses, alignment rotations) is
precomputed once, eagerly, and jax.jit is forced to trace, XLA-compile, and
run once (warmup) before ChainCombiner.unpolarized_amplitude returns -- see
decayamplitude.utils._warmup. Unlike the default path, trace and XLA compile
cannot be timed separately here: that fusion is the whole point of the
static_momenta API (a caller never sees an unwarmed function).

Usage
-----
    python3 benchmarks/benchmark_static_momenta.py [n_events]
"""

import sys
import time

# Must configure backend BEFORE any decayangle/decayamplitude imports,
# because decay_topology.py binds `cb = cfg.backend` at module load time.
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"
decayangle_config.use_rust = False
decayangle_config.sorting = "off"

import jax
jax.config.update("jax_enable_x64", True)

from benchmark_compile_time import build_combiner, generate_events

N_EVENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000


def main():
    print(f"N_EVENTS = {N_EVENTS:,}", flush=True)

    print("Building amplitude (3 topologies x 4 resonances = 12 resonances) ...", flush=True)
    t0 = time.perf_counter()
    combiner = build_combiner()
    couplings = combiner.generate_couplings()
    print(f"  model built in {time.perf_counter() - t0:.2f} s", flush=True)

    print(f"Generating {N_EVENTS:,} phase-space events ...", flush=True)
    raw_momenta = generate_events(N_EVENTS)
    topology_ref = combiner.reference.topology
    momenta = topology_ref.to_rest_frame(raw_momenta)

    print("\n── static_momenta construction (trace + XLA compile + warmup, fused) ──", flush=True)
    t0 = time.perf_counter()
    unpolarized, param_names = combiner.unpolarized_amplitude(couplings, complex_couplings=True, static_momenta=momenta)
    t_build = time.perf_counter() - t0
    coupling_names = param_names  # static_momenta drops "momenta" from the signature entirely
    n_params = len(coupling_names)
    print(f"  build (trace + XLA compile + warmup):      {t_build * 1e3:9.1f} ms", flush=True)
    print(f"  {n_params} free LS-coupling parameters", flush=True)

    param_vals = tuple(1.0 for _ in coupling_names)

    reps = 5
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = unpolarized(*param_vals)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)
    t_exec = sum(times) / len(times)
    print(f"  steady-state exec (avg of {reps}):             {t_exec * 1e3:9.1f} ms")

    print("\n── graph size diagnostics ──────────────────────────────────────────────", flush=True)
    # unpolarized is wrapped by _no_momenta_guard, which preserves the
    # underlying jax.jit object as __wrapped__ specifically so it stays
    # introspectable this way (see decayamplitude.utils._no_momenta_guard).
    jitted = unpolarized.__wrapped__
    jaxpr = jax.make_jaxpr(unpolarized)(*param_vals)
    n_eqns = sum(1 for _ in jaxpr.jaxpr.eqns)
    print(f"  jaxpr equations:  {n_eqns}")
    try:
        lowered = jitted.lower(*param_vals)
        hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        n_hlo_lines = hlo_text.count("\n")
        n_hlo_instr = hlo_text.count(" = ")
        print(f"  HLO text lines:   {n_hlo_lines}")
        print(f"  HLO instructions (approx, ' = ' count): {n_hlo_instr}")
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"  (could not retrieve HLO text: {e})")

    print("\n── summary ──────────────────────────────────────────────────────────────")
    print(f"  resonances: 12  |  topologies: 3  |  fit params: {n_params}  |  N_EVENTS: {N_EVENTS:,}")
    print(f"  build(trace+compile+warmup): {t_build * 1e3:.1f} ms   steady_exec: {t_exec * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
