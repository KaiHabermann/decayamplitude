# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

`decayamplitude` is a Python package for building full cascade reaction amplitudes in particle physics amplitude analyses. It works alongside `decayangle` (which handles decay topologies) to construct helicity amplitudes with proper Wigner rotations.

Key reference: "Wigner Rotations for Cascade Reactions" (PhysRevD.111.056015).

## Commands

```bash
# Run all tests with coverage
hatch run test:cov

# Run a single test file
hatch run test:pytest tests/test_threebody.py

# Run a single test
hatch run test:pytest tests/test_threebody.py::test_function_name

# Run pre-commit checks
pre-commit run --all-files
```

## Architecture

The physics model builds up in layers:

```
decayangle Topology
    ↓
Resonance objects (decay vertices with lineshapes + quantum numbers)
    ↓
DecayChain / MultiChain (single topology with one or all resonance combinations)
    ↓
ChainCombiner (aligns multiple chains to a reference frame)
    ↓
amplitude function (callable with LS/helicity coupling parameters → complex amplitude)
```

### Core modules

**`chain.py`** — Central orchestrator. `DecayChain` takes a topology, resonances dict, momenta, and final-state quantum numbers; produces amplitude functions and helicity matrices. `AlignedChain` applies Wigner D-rotations to align a chain to a reference frame. `MultiChain`/`AlignedMultiChain` iterate over all valid resonance combinations for a topology. The recursive amplitude calculation lives in `DecayChainNode.amplitude()`.

**`resonance.py`** — `Resonance` models a single decay vertex: quantum numbers, a lineshape function, and either LS or helicity couplings. `helicity_from_ls()` expands LS couplings into the helicity basis via Clebsch-Gordan coefficients. `generate_couplings()` auto-generates all parity-allowed couplings. `ResonanceDict` is a normalizing dict that maps node tuples to resonances.

**`combiner.py`** — `ChainCombiner` takes a list of `DecayChain`/`MultiChain` objects, uses the first as the reference frame, aligns all others with Wigner rotations, and sums amplitudes. Provides `unpolarized_amplitude()`, `polarized_amplitude()`, and `combined_function()`.

**`rotation.py`** — Angular momentum algebra. `Angular` wraps 2J integers; `QN` adds parity. `clebsch_gordan()` is cached via SymPy. `wigner_small_d` and `wigner_capital_d` compute rotation matrices used for frame alignment.

**`particle.py`** — `Particle` extends `QN` with a name and `type_id` for handling identical final-state particles. `DecaySetup` creates the `TopologyCollection` from particles and filters valid topologies given available resonances.

**`utils.py`** — `_create_function()` dynamically builds a Python function whose signature matches the coupling parameters (real + imaginary parts for complex couplings). This is how the library exposes user-facing amplitude callables.

**`backend.py`** — Single-line module: enables JAX 64-bit mode and imports `jax.numpy` as `numpy`. All numerical computation uses JAX for GPU/TPU compatibility.

### Key conventions

- Angular momenta are stored internally as **2J** (integers) to avoid half-integer floating point issues. Use `.value` for J and `.value2` for 2J.
- Coupling parameters for complex couplings are split into `_re` / `_im` suffixed arguments in generated functions.
- The `decayangle` library owns topology construction; `decayamplitude` adds resonances and amplitudes on top.
- Lineshape functions are plain callables (e.g., Breit-Wigner) passed into `Resonance`; they receive the invariant mass of the relevant subsystem.
