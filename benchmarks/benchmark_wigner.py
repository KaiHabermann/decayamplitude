"""
Benchmark: external alignment Wigner matrices vs. baked-in for a 3-body amplitude.

Background
----------
The AlignedChain.wigner_dict contains one complex JAX array of shape (N_EVENTS,)
per helicity-projection pair per final-state particle.  When these arrays are
captured as closure constants inside jax.jit they are embedded as XLA literal
constants, which makes compilation slow and the compiled artifact large.

When passed as explicit function arguments ("external" mode) they become HLO
parameters: JAX traces the computation graph abstractly with respect to their
shape/dtype, so the constants are never baked in.  This should dramatically
reduce compile time.

Note: only the alignment wigner_dict (AlignedChain) is passed externally here,
because the helicity-angle D-matrices for the reference chain go through a
sympy-lambdified numpy function that cannot handle JAX traced values.
Externalising the reference helicity angles would require porting wigner_small_d
to use jax.numpy ops (out of scope for this benchmark).

Usage
-----
    cd /path/to/decayamplitude
    python3 benchmarks/benchmark_wigner.py
"""

import time

# ── Force the decayangle JAX backend ─────────────────────────────────────────
# Four decayangle modules cache `cb = cfg.backend` at import time.  Setting the
# config string is not enough once those modules are already loaded, so we also
# patch their module-level `cb` variable to jax.numpy directly.
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"
decayangle_config.sorting = "off"

import jax
import jax.numpy as jnp
import phasespace

import decayangle.kinematics
import decayangle.lorentz
import decayangle.decay_topology
import decayangle.numerics_helpers

for _mod in (
    decayangle.kinematics,
    decayangle.lorentz,
    decayangle.decay_topology,
    decayangle.numerics_helpers,
):
    _mod.cb = jnp

from decayamplitude.backend import numpy as np  # also enables JAX 64-bit mode
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayangle.decay_topology import Topology, Node

# ── Physics / run parameters ──────────────────────────────────────────────────
N_EVENTS    = 10_000_000
EXEC_REPS   = 5          # number of post-compile calls used for exec timing

# B0 → π- π+ ρ0 (ρ0 treated as a stable spin-1 final-state particle)
# final-state QN: p1=π-(J=0), p2=π+(J=0), p3=ρ0(J=1)
# Topology 1: 0 → ((2,3) rho-like[J=1])  + p1(J=0)
# Topology 2: 0 → ((1,2) f2-like[J=2])   + p3(J=1)
# Angular-momentum conservation (J_mother=0):
#   T1: J_R=1, J_p1=0 → S=1, L=1 → J=0 ✓
#   T2: J_R=2, J_p3=1 → S∈{1,2,3}, S=2,L=2→J=0 ✓
MOTHER_MASS = 5.279      # GeV  (B0)
M1          = 0.140      # GeV  (π-)
M2          = 0.140      # GeV  (π+)
M3          = 0.775      # GeV  (ρ0, treated as stable in phase space)


# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_events(n: int) -> dict:
    weights, particles = phasespace.nbody_decay(
        MOTHER_MASS, [M1, M2, M3]
    ).generate(n_events=n)
    return {
        1: np.array(particles["p_0"]),
        2: np.array(particles["p_1"]),
        3: np.array(particles["p_2"]),
    }


def bw(mass):
    """Breit-Wigner lineshape closed over the per-event invariant mass array."""
    def lineshape(l, s, m0, gamma):
        return 1.0 / (mass ** 2 - m0 ** 2 + 1j * mass * gamma)
    return lineshape


def build_combiner(raw_momenta: dict) -> tuple[ChainCombiner, dict]:
    """
    Build a two-topology 3-body amplitude for B0 → π- π+ ρ0.

    Topology 1:  0 → ((2,3) rho-like[J=1]) + 1(π-)
    Topology 2:  0 → ((1,2) f2-like[J=2])  + 3(ρ0)

    Having p3 as spin-1 (ρ0) makes the alignment wigner_dict non-trivial
    (3×3 matrices), giving 9 complex arrays of shape (N_EVENTS,) — large
    enough to show a clear difference in XLA constant embedding.
    """
    topology1 = Topology(0, decay_topology=((2, 3), 1))
    topology2 = Topology(0, decay_topology=((1, 2), 3))
    momenta   = topology1.to_rest_frame(raw_momenta)

    m23 = topology1.nodes[(2, 3)].mass(momenta)
    m12 = topology2.nodes[(1, 2)].mass(momenta)

    final_state_qn = {
        1: QN(0,  1),   # π-  spin-0
        2: QN(0,  1),   # π+  spin-0
        3: QN(2, -1),   # ρ0  spin-1 (negative intrinsic parity)
    }

    def root_resonance(name):
        return [Resonance(
            Node(0), quantum_numbers=QN(0, 1),
            lineshape=lambda *a: 1.0, argnames=[],
            preserve_partity=False, name=name,
        )]

    resonances1 = {
        (2, 3): [Resonance(
            Node((2, 3)), quantum_numbers=QN(2, 1),
            lineshape=bw(m23), argnames=["m_R1", "G_R1"],
            preserve_partity=False, name="R23",
        )],
        0: root_resonance("B1"),
    }
    resonances2 = {
        (1, 2): [Resonance(
            Node((1, 2)), quantum_numbers=QN(4, 1),
            lineshape=bw(m12), argnames=["m_R2", "G_R2"],
            preserve_partity=False, name="R12",
        )],
        0: root_resonance("B2"),
    }

    chain1 = MultiChain(
        topology=topology1, resonances=resonances1,
        momenta=momenta, final_state_qn=final_state_qn,
    )
    chain2 = MultiChain(
        topology=topology2, resonances=resonances2,
        momenta=momenta, final_state_qn=final_state_qn,
    )
    return ChainCombiner([chain1, chain2]), momenta


def measure(label: str, jitted_fn, args: tuple, n_reps: int = EXEC_REPS):
    """
    First call  → compile + first execution (compile time).
    Next n_reps → average execution time.
    """
    t0 = time.perf_counter()
    r  = jitted_fn(*args)
    jax.block_until_ready(r)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_reps):
        r = jitted_fn(*args)
        jax.block_until_ready(r)
    exec_s = (time.perf_counter() - t0) / n_reps

    print(f"  [{label}] compile={compile_s:.2f}s  exec={exec_s*1e3:.2f}ms/call")
    return compile_s, exec_s


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"N_EVENTS = {N_EVENTS:,}", flush=True)
    print("Generating phase-space events …", flush=True)
    raw_momenta = generate_events(N_EVENTS)

    print("Building amplitude (2 topologies, BW lineshapes) …", flush=True)
    combined, momenta = build_combiner(raw_momenta)
    couplings = combined.generate_couplings()

    # ── Warm up: one eager call to populate sympy/Resonance caches ────────────
    func_in, params_in = combined.unpolarized_amplitude(couplings, complex_couplings=False)
    vals_in = [1.0] * len(params_in)
    ref = func_in(*vals_in)
    print(f"Reference (eager) amplitude sum = {float(jnp.sum(ref)):.6g}", flush=True)

    # ── Case 1: INTERNAL – wigner_dict baked into closure ────────────────────
    # The D-matrix arrays live inside self.wigner_dict (an AlignedChain attribute)
    # and are captured as XLA literal constants when the function is jit-compiled.
    # print("\n── INTERNAL wigner ──────────────────────────────────────")

    # def nll_in(*args):
    #     return -jnp.sum(jnp.log(func_in(*args)))

    # grad_in  = jax.jit(jax.grad(nll_in, argnums=list(range(len(params_in)))))
    # c_in, e_in = measure("internal", grad_in, tuple(vals_in))

    # ── Case 2: EXTERNAL – all Wigner matrices passed as function arguments ──
    # The full wigner_matrices dict contains:
    #   "reference": {"helicity_wigner_dict": {str(decay_tuple): {(m,n): array}}}
    #   "aligned":   [{"helicity_wigner_dict": ..., "wigner_dict": {key: {(h',h): array}}}]
    # All D-matrix elements are plain JAX arrays — no lambdify call inside jit.
    print("\n── EXTERNAL wigner ──────────────────────────────────────")

    func_ext, params_ext, _ = combined.unpolarized_amplitude(
        couplings, complex_couplings=False, external_wigner=True
    )
    coupling_names = [p for p in params_ext if p != "wigner_matrices"]
    vals_ext = [1.0] * len(coupling_names)

    full_wigner = combined.wigner_matrices

    def nll_ext(wigner, *coupling_args):
        return -jnp.sum(jnp.log(func_ext(wigner, *coupling_args)))

    argnums_ext = list(range(1, 1 + len(vals_ext)))
    grad_ext    = jax.jit(jax.grad(nll_ext, argnums=argnums_ext))
    c_ext, e_ext = measure("external", grad_ext, (full_wigner, *vals_ext))

    # ── Case 3: REUSE – compile once, swap events without recompiling ─────────
    # This is the key practical advantage of external wigner: different event
    # arrays share the same compiled artifact (same shape/dtype → same treedef).
    # With internal wigner a new event array requires a new closure → recompile.
    print("\n── REUSE (external, 2nd event set) ──────────────────────")
    raw_momenta2  = generate_events(N_EVENTS)
    combined2, _  = build_combiner(raw_momenta2)
    full_wigner2  = combined2.wigner_matrices

    # First call with wigner2 reuses the compiled grad_ext (no recompile).
    c_reuse, e_reuse = measure("reuse", grad_ext, (full_wigner2, *vals_ext))

    # Internal baseline: must recompile for a new closure over different event data.
    # combined2 has fresh Resonance instances, so we generate its own couplings.
    couplings2   = combined2.generate_couplings()
    func_in2, params_in2 = combined2.unpolarized_amplitude(couplings2, complex_couplings=False)
    vals_in2     = [1.0] * len(params_in2)
    def nll_in2(*args):
        return -jnp.sum(jnp.log(func_in2(*args)))
    grad_in2     = jax.jit(jax.grad(nll_in2, argnums=list(range(len(params_in2)))))
    c_in2, e_in2 = measure("internal-2nd", grad_in2, tuple(vals_in2))

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 56
    print(f"\n{'═'*W}")
    print(f"{'':30s} {'internal':>9s}  {'external':>9s}")
    print(f"{'─'*W}")
    print(f"{'1st compile jit(grad(nll)) [s]':30s} {c_in:>9.2f}  {c_ext:>9.2f}")
    print(f"{'execution / call         [ms]':30s} {e_in*1e3:>9.2f}  {e_ext*1e3:>9.2f}")
    print(f"{'─'*W}")
    print(f"{'2nd compile (new events) [s]':30s} {c_in2:>9.2f}  {c_reuse:>9.2f}  ← reuse speedup: {c_in2/c_reuse:.1f}×")
    print(f"{'2nd exec / call          [ms]':30s} {e_in2*1e3:>9.2f}  {e_reuse*1e3:>9.2f}")
    print(f"{'─'*W}")
    print(f"1st compile speedup:  {c_in / c_ext:.2f}×  (external is {'faster' if c_ext < c_in else 'slower'})")
    print(f"exec speedup (same events): {e_in / e_ext:.2f}×")
    print(f"2nd compile speedup:  {c_in2 / c_reuse:.2f}×  (external reuse vs internal recompile)")
    print(f"{'':30s} N_EVENTS = {N_EVENTS:,}")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()
