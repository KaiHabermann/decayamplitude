"""
Benchmark: JIT compilation and vmap speedup for per-event amplitude evaluation.

Per-event API:  unpolarized(momenta, *coupling_params) -> shape (N_EVENTS,)

Scenarios
---------
1. Eager       — plain Python call, no JIT
2. JIT         — jax.jit(unpolarized): compile time + steady-state exec time
3. JIT + grad  — jax.jit(grad(nll)): compile time + exec time (fitting use case)
4. vmap        — vectorise over N_PARAM_SETS parameter combinations with fixed
                 momenta (useful for parameter scans, ensemble fits, bootstrapping)
                 vs. sequential JIT calls for the same N_PARAM_SETS evaluations

Usage
-----
    python3 benchmarks/benchmark_wigner.py
"""

import time

# Must configure backend BEFORE any decayangle/decayamplitude imports,
# because decay_topology.py binds `cb = cfg.backend` at module load time.
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"
decayangle_config.use_rust = False
decayangle_config.sorting = "off"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from decayamplitude.backend import numpy as np
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayangle.decay_topology import Topology, Node

# ── Run parameters ────────────────────────────────────────────────────────────
N_EVENTS     = 100_000
N_PARAM_SETS = 32        # parallel parameter combinations for vmap benchmark
EXEC_REPS    = 5

# B0 → π- π+ ρ0  (ρ0 treated as stable spin-1 final-state particle)
MOTHER_MASS  = 5.279     # GeV
M1, M2, M3   = 0.140, 0.140, 0.775


# ── Lineshape ─────────────────────────────────────────────────────────────────
def bw(mass, l, s, m0, gamma):
    return 1.0 / (mass**2 - m0**2 + 1j * mass * gamma)


# ── Phase-space generator ─────────────────────────────────────────────────────
def generate_events(n: int) -> dict:
    try:
        import phasespace
        weights, particles = phasespace.nbody_decay(
            MOTHER_MASS, [M1, M2, M3]
        ).generate(n_events=n)
        return {
            1: jnp.array(particles["p_0"]),
            2: jnp.array(particles["p_1"]),
            3: jnp.array(particles["p_2"]),
        }
    except ImportError:
        # Fallback: random 4-vectors (unphysical but sufficient for timing)
        import numpy as onp
        key = onp.random.default_rng(42)
        def rand_p(mass):
            p3 = key.normal(size=(n, 3)).astype("float64") * 0.5
            E  = onp.sqrt((p3**2).sum(-1) + mass**2)
            return jnp.array(onp.concatenate([p3, E[:, None]], axis=1))
        return {1: rand_p(M1), 2: rand_p(M2), 3: rand_p(M3)}


# ── Amplitude builder ─────────────────────────────────────────────────────────
def build_combiner() -> ChainCombiner:
    topology1 = Topology(0, decay_topology=((2, 3), 1))
    topology2 = Topology(0, decay_topology=((1, 2), 3))

    final_state_qn = {
        1: QN(0,  1),   # π-  spin-0
        2: QN(0,  1),   # π+  spin-0
        3: QN(2, -1),   # ρ0  spin-1
    }

    def root(name):
        return [Resonance(Node(0), quantum_numbers=QN(0, 1),
                          lineshape=lambda *a: 1.0, argnames=[],
                          preserve_partity=False, name=name)]

    resonances1 = {
        (2, 3): [Resonance(Node((2, 3)), quantum_numbers=QN(2, 1),
                           lineshape=bw, argnames=["m_R1", "G_R1"],
                           preserve_partity=False, name="R23")],
        0: root("B1"),
    }
    resonances2 = {
        (1, 2): [Resonance(Node((1, 2)), quantum_numbers=QN(4, 1),
                           lineshape=bw, argnames=["m_R2", "G_R2"],
                           preserve_partity=False, name="R12")],
        0: root("B2"),
    }

    chain1 = MultiChain(topology=topology1, resonances=resonances1,
                        final_state_qn=final_state_qn)
    chain2 = MultiChain(topology=topology2, resonances=resonances2,
                        final_state_qn=final_state_qn)
    return ChainCombiner([chain1, chain2])


# ── Timing helper ─────────────────────────────────────────────────────────────
def timed(label, fn, args, reps=EXEC_REPS):
    t0 = time.perf_counter()
    r  = fn(*args)
    jax.block_until_ready(r)
    t_first = time.perf_counter() - t0

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r  = fn(*args)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)
    t_avg = sum(times) / len(times)

    print(f"  {label:<44s}  first={t_first*1e3:8.1f} ms   avg={t_avg*1e3:8.1f} ms")
    return t_first, t_avg


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"N_EVENTS = {N_EVENTS:,}  |  N_PARAM_SETS = {N_PARAM_SETS}", flush=True)

    print("Generating phase-space events …", flush=True)
    raw_momenta = generate_events(N_EVENTS)

    print("Building amplitude (2 topologies, BW lineshapes) …", flush=True)
    combined     = build_combiner()
    couplings    = combined.generate_couplings()
    unpolarized, param_names = combined.unpolarized_amplitude(
        couplings, complex_couplings=True
    )

    coupling_names = [p for p in param_names if p != "momenta"]
    param_vals     = tuple(1.0 for _ in coupling_names)
    n_params       = len(param_vals)
    print(f"Coupling parameters ({n_params}): {coupling_names}", flush=True)

    # Put momenta in the mother rest frame (same as test does)
    topology_ref = Topology(0, decay_topology=((2, 3), 1))
    momenta = topology_ref.to_rest_frame(raw_momenta)

    # ── Single-event function: XLA traces with shape (4,) per particle ─────────
    # All benchmarks are built on this so momenta are always a traced leaf,
    # never embedded as XLA constants.
    event_axes = {k: 0 for k in momenta}  # vmap axis: first dim of each particle

    def f_1e(momenta_1e, *params):
        """Amplitude for a single phase-space point."""
        return unpolarized(momenta_1e, *params)

    # Vectorise over events → shape (N_EVENTS,)
    f_batched = jax.vmap(f_1e, in_axes=(event_axes,) + (None,) * n_params)

    # ── 1. Variable momenta vs baked-in momenta ───────────────────────────────
    print("\n── 1. Variable momenta vs baked-in ───────────────────────────────────────")

    # Without vmap: pass full (N_EVENTS, 4) momenta directly to unpolarized.
    # unpolarized already handles batched inputs; no intermediate vmap needed.
    jit_direct = jax.jit(unpolarized)
    t_direct_jit_compile, t_direct_jit_exec = timed("variable  jit(unpolarized) no vmap", jit_direct, (momenta, *param_vals))

    # With vmap: compile for single-event (4,) shapes, vmap handles the batch.
    jit_batched = jax.jit(f_batched)
    t_jit_compile, t_jit_exec = timed("variable  jit(vmap(f_1e))", jit_batched, (momenta, *param_vals))

    # Baked-in: momenta closed over — XLA embeds arrays as constants and can
    # constant-fold the helicity-angle computation away entirely.
    def make_baked(momenta_const):
        def f(*params):
            return f_batched(momenta_const, *params)
        return jax.jit(f)

    t0 = time.perf_counter()
    jit_baked = make_baked(momenta)
    r = jit_baked(*param_vals); jax.block_until_ready(r)
    t_baked_compile = time.perf_counter() - t0
    baked_times = []
    for _ in range(EXEC_REPS):
        t0 = time.perf_counter()
        r = jit_baked(*param_vals); jax.block_until_ready(r)
        baked_times.append(time.perf_counter() - t0)
    t_baked_exec = sum(baked_times) / len(baked_times)
    print(f"  {'baked-in  jit(closure(f_batched))':<44s}  first={t_baked_compile*1e3:8.1f} ms   avg={t_baked_exec*1e3:8.1f} ms")
    print(f"  exec speedup baked vs variable: {t_jit_exec / t_baked_exec:.2f}×")

    # ── 2. JIT + gradient of NLL ──────────────────────────────────────────────
    print("\n── 2. JIT + gradient of NLL ──────────────────────────────────────────────")

    def nll(momenta_batch, *params):
        w = f_batched(momenta_batch, *params)
        return -jnp.sum(jnp.log(jnp.abs(w) + 1e-300))

    argnums  = list(range(1, 1 + n_params))
    grad_nll = jax.jit(jax.grad(nll, argnums=argnums))
    t_grad_compile, t_grad_exec = timed(
        "jit(grad(nll))", grad_nll, (momenta, *param_vals)
    )

    # ── 3. vmap over grad — population-based fitting ──────────────────────────
    # N_PARAM_SETS independent fitters each compute grad(NLL) on the same
    # dataset simultaneously. Each fitter has its own parameter set.
    # vmap over params_row → output (N_PARAM_SETS, n_params) gradient matrix.
    print("\n── 3. vmap over grad (population-based fitting) ─────────────────────────")

    def nll_one(momenta_batch, params_row):
        w = f_batched(momenta_batch, *params_row)
        return -jnp.sum(jnp.log(jnp.abs(w) + 1e-300))

    # Gradient w.r.t. params_row (argnums=1) for a single fitter
    grad_one  = jax.grad(nll_one, argnums=1)

    # vmap over N_PARAM_SETS fitters — momenta broadcast, param rows mapped
    vmap_grad = jax.jit(jax.vmap(grad_one, in_axes=(None, 0)))
    t_vmap_grad_compile, t_vmap_grad_exec = timed(
        f"jit(vmap(grad(nll))) [{N_PARAM_SETS} fitters]",
        vmap_grad, (momenta, param_grid)
    )

    # Sequential baseline: N_PARAM_SETS individual grad calls
    _ = grad_nll(momenta, *param_vals); jax.block_until_ready(_)
    seq_grad_times = []
    for _ in range(EXEC_REPS):
        t0 = time.perf_counter()
        for i in range(N_PARAM_SETS):
            r = grad_nll(momenta, *param_grid[i])
            jax.block_until_ready(r)
        seq_grad_times.append(time.perf_counter() - t0)
    t_seq_grad = sum(seq_grad_times) / len(seq_grad_times)
    print(f"  {'sequential jit(grad(nll)) ×' + str(N_PARAM_SETS):<44s}  avg={t_seq_grad*1e3:8.1f} ms")

    # ── 4. vmap over parameter sets ───────────────────────────────────────────
    # Fixed momenta, N_PARAM_SETS coupling combinations in one compiled pass.
    # Useful for: parameter scans, ensemble fits, bootstrapping.
    # Two nested vmaps, both with momenta as an explicit leaf:
    #   inner vmap — over events       → output (N_EVENTS,)
    #   outer vmap — over param rows   → output (N_PARAM_SETS, N_EVENTS)
    print("\n── 4. vmap over parameter sets (fixed momenta) ──────────────────────────")

    param_grid = jnp.ones((N_PARAM_SETS, n_params), dtype=jnp.float64)

    def f_1e_row(momenta_1e, params_row):
        return f_1e(momenta_1e, *params_row)

    # Double vmap from f_1e: traces with (4,) shapes per particle
    f_events  = jax.vmap(f_1e_row, in_axes=(event_axes, None))   # → (N_EVENTS,)
    f_all     = jax.vmap(f_events, in_axes=(None, 0))             # → (N_PARAM_SETS, N_EVENTS)
    vmap_fn   = jax.jit(f_all)
    t_vmap_compile, t_vmap_exec = timed(
        f"jit(vmap×2 from f_1e) [{N_PARAM_SETS} sets]",
        vmap_fn, (momenta, param_grid)
    )

    # Direct: single vmap over params using f_batched (already event-vectorised)
    # Traces with (N_EVENTS, 4) shapes — larger graph, but only one vmap level.
    def eval_params_direct(momenta_batch, params_row):
        return f_batched(momenta_batch, *params_row)

    vmap_direct = jax.jit(jax.vmap(eval_params_direct, in_axes=(None, 0)))
    t_direct_compile, t_direct_exec = timed(
        f"jit(vmap_params(f_batched)) [{N_PARAM_SETS} sets]",
        vmap_direct, (momenta, param_grid)
    )

    # ── 5. Recompile cost: swap in a different N_EVENTS ───────────────────────
    # JAX JIT caches by shape — different N_EVENTS → recompile.
    # vmap×2 (inner graph traces (4,)) should recompile faster than
    # vmap_params(f_batched) (inner graph traces (N_EVENTS, 4)).
    print("\n── 5. Recompile with N_EVENTS_2 ──────────────────────────────────────────")
    N_EVENTS_2 = N_EVENTS * 5
    print(f"  Generating {N_EVENTS_2:,} events …", flush=True)
    raw_momenta_2 = generate_events(N_EVENTS_2)
    momenta_2     = topology_ref.to_rest_frame(raw_momenta_2)

    t0 = time.perf_counter()
    r  = jit_direct(momenta_2, *param_vals); jax.block_until_ready(r)
    t_recompile_no_vmap = time.perf_counter() - t0
    print(f"  {'no-vmap recompile':<44s}  {t_recompile_no_vmap*1e3:8.1f} ms")

    t0 = time.perf_counter()
    r  = jit_batched(momenta_2, *param_vals); jax.block_until_ready(r)
    t_recompile_vmap = time.perf_counter() - t0
    print(f"  {'vmap recompile':<44s}  {t_recompile_vmap*1e3:8.1f} ms")

    t0 = time.perf_counter()
    r  = vmap_fn(momenta_2, param_grid); jax.block_until_ready(r)
    t_recompile_vmap2 = time.perf_counter() - t0
    print(f"  {'vmap×2 recompile':<44s}  {t_recompile_vmap2*1e3:8.1f} ms")

    t0 = time.perf_counter()
    r  = vmap_direct(momenta_2, param_grid); jax.block_until_ready(r)
    t_recompile_direct = time.perf_counter() - t0
    print(f"  {'vmap_params(f_batched) recompile':<44s}  {t_recompile_direct*1e3:8.1f} ms")

    # Baked-in recompile cost when events change
    t0 = time.perf_counter()
    jit_baked_2 = make_baked(momenta_2)
    r = jit_baked_2(*param_vals); jax.block_until_ready(r)
    t_baked_recompile = time.perf_counter() - t0

    # Sequential baseline: N_PARAM_SETS individual jit_batched calls
    _ = jit_batched(momenta, *param_vals); jax.block_until_ready(_)
    seq_times = []
    for _ in range(EXEC_REPS):
        t0 = time.perf_counter()
        for i in range(N_PARAM_SETS):
            r = jit_batched(momenta, *param_grid[i])
            jax.block_until_ready(r)
        seq_times.append(time.perf_counter() - t0)
    t_seq = sum(seq_times) / len(seq_times)
    print(f"  {'sequential jit(vmap) ×' + str(N_PARAM_SETS):<44s}  avg={t_seq*1e3:8.1f} ms")

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 76
    R = f"recompile {N_EVENTS_2//N_EVENTS}× events (ms)"
    print(f"\n{'═'*W}")
    print(f"  N_EVENTS = {N_EVENTS:,}   |   N_PARAM_SETS = {N_PARAM_SETS}")
    print(f"{'─'*W}")
    print(f"  {'Scenario':<44s}  {'compile':>8s}  {'exec':>8s}  {R:>20s}")
    print(f"{'─'*W}")
    print(f"  {'─── single param eval ───'}")
    print(f"  {'no-vmap   jit(unpolarized)':<44s}  {t_direct_jit_compile*1e3:>8.1f}  {t_direct_jit_exec*1e3:>8.1f}  {t_recompile_no_vmap*1e3:>20.1f}")
    print(f"  {'with-vmap jit(vmap_events(f_1e))':<44s}  {t_jit_compile*1e3:>8.1f}  {t_jit_exec*1e3:>8.1f}  {t_recompile_vmap*1e3:>20.1f}")
    print(f"  {'baked-in  jit(closure(vmap_events))':<44s}  {t_baked_compile*1e3:>8.1f}  {t_baked_exec*1e3:>8.1f}  {t_baked_recompile*1e3:>20.1f}")
    print(f"  {'JIT + grad(NLL)':<44s}  {t_grad_compile*1e3:>8.1f}  {t_grad_exec*1e3:>8.1f}  {'—':>20s}")
    print(f"{'─'*W}")
    print(f"  {'─── gradient, {N_PARAM_SETS} fitters (population fitting) ───'}")
    print(f"  {'vmap(grad)  jit(vmap(grad(nll)))':<44s}  {t_vmap_grad_compile*1e3:>8.1f}  {t_vmap_grad_exec*1e3:>8.1f}  {'—':>20s}")
    print(f"  {'sequential  jit(grad) ×' + str(N_PARAM_SETS):<44s}  {'—':>8s}  {t_seq_grad*1e3:>8.1f}  {'—':>20s}")
    print(f"{'─'*W}")
    print(f"  {'─── {N_PARAM_SETS} param sets in parallel (vmap over params) ───'}")
    print(f"  {'vmap_events+params  jit(vmap×2)':<44s}  {t_vmap_compile*1e3:>8.1f}  {t_vmap_exec*1e3:>8.1f}  {t_recompile_vmap2*1e3:>20.1f}")
    print(f"  {'vmap_params(f_batched)  jit(vmap)':<44s}  {t_direct_compile*1e3:>8.1f}  {t_direct_exec*1e3:>8.1f}  {t_recompile_direct*1e3:>20.1f}")
    print(f"{'─'*W}")
    print(f"  {'─── {N_PARAM_SETS} param sets sequential (no vmap over params) ───'}")
    print(f"  {'sequential  jit(vmap_events) ×' + str(N_PARAM_SETS):<44s}  {'—':>8s}  {t_seq*1e3:>8.1f}  {'—':>20s}")
    print(f"{'─'*W}")
    print(f"  compile speedup  vmap vs no-vmap:              {t_direct_jit_compile / t_jit_compile:>6.2f}×")
    print(f"  recompile speedup  vmap vs no-vmap:           {t_recompile_no_vmap / t_recompile_vmap:>6.2f}×")
    print(f"  exec speedup  baked vs variable:               {t_jit_exec / t_baked_exec:>6.2f}×")
    print(f"  exec speedup  vmap(grad) vs sequential grad:   {t_seq_grad / t_vmap_grad_exec:>6.2f}×")
    print(f"  exec speedup  vmap×2 vs sequential:            {t_seq / t_vmap_exec:>6.2f}×")
    print(f"  exec speedup  vmap_params vs sequential:       {t_seq / t_direct_exec:>6.2f}×")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()
