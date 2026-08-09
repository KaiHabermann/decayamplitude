from decayamplitude.chain import DecayChain, AlignedChain, MultiChain, AlignedMultiChain
from typing import Callable
from decayamplitude.utils import _create_function, _no_momenta_guard, _warmup
import jax

class ChainCombiner:
    """
    Combines multiple decay chains into a single amplitude.
    The first chain is the reference frame; all others are aligned to it via Wigner rotations.
    """

    def __init__(self, chains: list[DecayChain | MultiChain]) -> None:
        self.chains = chains
        self.reference = chains[0]
        self.aligned_chains = [
            AlignedMultiChain.from_multichain(chain, self.reference)
            if isinstance(chain, MultiChain) else
            AlignedChain(chain.topology, chain.resonances, chain.final_state_qn, self.reference)
            for chain in chains[1:]
        ]

    @property
    def root_resonance(self):
        if all(chain.root_resonance.quantum_numbers == self.reference.root_resonance.quantum_numbers for chain in self.chains):
            return self.reference.root_resonance
        return None

    def _static_cache(self, momenta) -> dict:
        """Eagerly precompute every momenta-only cache entry used across the
        whole combiner: the reference chain's own entries plus each aligned
        chain's (which additionally include the alignment rotation). Used by
        the `static_momenta` mode of unpolarized_amplitude/polarized_amplitude/
        matrix_function -- see DecayChain._static_cache and
        AlignedChain._static_cache for what's actually computed.
        """
        cache = dict(self.reference._static_cache(momenta))
        for aligned in self.aligned_chains:
            cache.update(aligned._static_cache(momenta))
        return cache

    @property
    def single_chains(self) -> list[DecayChain]:
        """Flattens the aligned chains and multi chains into a list of single (non-Multi) chains."""
        chains = [self.reference] if not hasattr(self.reference, "chains") else list(self.reference.chains)
        for aligned_chain in self.aligned_chains:
            if isinstance(aligned_chain, AlignedChain):
                chains.append(aligned_chain)
            elif isinstance(aligned_chain, AlignedMultiChain):
                chains.extend(aligned_chain.chains)
        return chains

    @property
    def combined_function(self):
        """Returns a function f(h0, lambdas, arguments, momenta, cache=None) that sums the aligned amplitudes of all chains."""
        def f(h0, lambdas: dict, arguments: dict, momenta: dict, cache: dict | None = None, momenta_cache: dict | None = None):
            amplitudes = [
                chain.aligned_matrix(h0, arguments, momenta, cache=cache, momenta_cache=momenta_cache)[tuple(lambdas[k] for k in sorted(lambdas.keys()))]
                for chain in self.aligned_chains
            ]
            return sum(amplitudes) + self.reference.chain_function(h0, lambdas, arguments, momenta, cache=cache, momenta_cache=momenta_cache)
        return f

    @property
    def combined_matrix(self) -> Callable:
        """Returns a function f(h0, arguments, momenta, cache=None, momenta_cache=None) that sums the aligned helicity matrices of all chains.

        `cache`, if given, is a dict shared across multiple calls that only
        differ in h0 (see DecayChainNode.amplitude): it lets the h0-independent
        parts of every chain's computation be reused across h0 values instead
        of being recomputed from scratch on every h0 (see unpolarized_amplitude,
        which is the caller that actually loops over h0 and populates this).
        """
        def matrix(h0, arguments: dict, momenta: dict, cache: dict | None = None, momenta_cache: dict | None = None) -> dict:
            matrices = [chain.aligned_matrix(h0, arguments, momenta, cache=cache, momenta_cache=momenta_cache) for chain in self.aligned_chains]
            matrices.append(self.reference.matrix(h0, arguments, momenta, cache=cache, momenta_cache=momenta_cache))
            return {
                key: sum(m[key] for m in matrices)
                for key in matrices[0].keys()
            }
        return matrix

    def generate_couplings(self):
        """Generates the couplings for the ls basis, merged across all chains."""
        couplings = {}
        for chain in self.chains:
            couplings.update(chain.generate_couplings())
        return couplings

    @property
    def resonance_params(self) -> list[str]:
        resonance_parameter_names = [name for chain in self.chains for name in chain.resonance_params]
        return list(set(resonance_parameter_names))

    def unpolarized_amplitude(self, ls_couplings: dict, complex_couplings=True, static_momenta=None) -> tuple[Callable, list[str]]:
        """
        Returns (func, param_names) where func(momenta, *params) -> per-event scalar intensity.
        Vectorize over events with jax.vmap(func, in_axes=({k: 0 for k in momenta}, None, ...)).

        If `static_momenta` is given, momenta is treated as a fixed dataset:
        every momenta-only computation (helicity angles, alignment rotations,
        masses) is precomputed once, eagerly, right now -- not retraced from
        a momenta argument on every call. The returned func(*params) does not
        take momenta at all (calling it with momenta raises TypeError -- see
        decayamplitude.utils._no_momenta_guard) and is pre-compiled (warmed
        up) before being returned. Use this for fitting, where momenta never
        changes between calls; naively closing over momenta as a constant
        does NOT give this benefit (XLA does not fold the angle computation
        away), which is why this precomputes concrete values up front instead.
        """
        if self.root_resonance is None:
            raise ValueError(f"The root resonance must be the same for all chains! Root = {self.reference.topology.root}.")

        # Built fresh per call and closed over below -- never stored on self --
        # so this build's momenta_cache can't affect any other function (static
        # or not) built from this combiner, or from any chain it wraps.
        momenta_cache = self._static_cache(static_momenta) if static_momenta is not None else None

        def f(arguments: dict):
            # Momenta-only lookups (helicity angles, masses, alignment
            # rotations) are served by momenta_cache when static_momenta is
            # given, or recomputed per call otherwise -- either way this
            # closure doesn't need to know which. `cache` is only for
            # h0_independent_terms, which must stay scoped to this call since
            # it bakes in `arguments`; sharing it across the h0 loop below
            # reuses the h0-independent parts of the computation (everything
            # below the top Wigner-D rotation, see DecayChainNode.amplitude)
            # instead of redoing them from scratch per h0.
            momenta = static_momenta if static_momenta is not None else arguments.pop("momenta")
            cache: dict = {}
            return sum(
                abs(v)**2
                for h0 in self.root_resonance.quantum_numbers.angular.projections()
                for v in self.combined_matrix(h0, arguments, momenta, cache=cache, momenta_cache=momenta_cache).values()
            )

        if static_momenta is not None:
            func, argnames = _create_function(self.resonance_params, ls_couplings, f, complex_couplings=complex_couplings)
            # h0 is only ever a Python-level loop variable here (looped over
            # above, never a function argument), so unlike polarized_amplitude/
            # matrix_function this needs no static_argnums to be jit-safe.
            func = jax.jit(func)
            func = _no_momenta_guard(func)
            _warmup(func, argnames)
            return func, argnames

        names = ["momenta"] + self.resonance_params
        return _create_function(names, ls_couplings, f, complex_couplings=complex_couplings)

    def polarized_amplitude(self, ls_couplings: dict, complex_couplings: bool = True, static_momenta=None) -> tuple[Callable, list[str], list[str]]:
        """
        Returns (func, helicity_names, coupling_names) where func(momenta, h0, *h_finals, *params).

        If `static_momenta` is given, momenta is baked in as a fixed dataset
        the same way as in unpolarized_amplitude: func(h0, *h_finals, *params)
        does not take momenta at all, and is pre-compiled before being
        returned. See unpolarized_amplitude for details.
        """
        sorted_final_state_nodes = sorted([n.node.value for n in self.reference.final_state_nodes])
        final_state_lambdas = sorted([f"h_{n}" for n in sorted_final_state_nodes])

        # See unpolarized_amplitude: built fresh per call, never stored on
        # self, so this build can't affect any other function built from
        # this combiner.
        momenta_cache = self._static_cache(static_momenta) if static_momenta is not None else None

        def fun(arguments: dict):
            momenta = static_momenta if static_momenta is not None else arguments.pop("momenta")
            h0 = arguments.pop("h0")
            lambdas = {n: arguments.pop(k) for k, n in zip(final_state_lambdas, sorted_final_state_nodes)}
            cache: dict = {}
            return self.combined_function(h0, lambdas, arguments, momenta, cache=cache, momenta_cache=momenta_cache)

        if static_momenta is not None:
            names = ["h0", *final_state_lambdas]
            func, argnames = _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
            # h0 and each h_<n> flow into wigner_capital_d's underlying
            # _wigner_d_coefficients, which does Python-level integer
            # comparisons/lru_cache lookups -- they must be static_argnums or
            # jax.jit fails with "unhashable type: DynamicJaxprTracer".
            static_argnums = tuple(argnames.index(n) for n in names)
            func = jax.jit(func, static_argnums=static_argnums)
            func = _no_momenta_guard(func)
            # h0/h_<n> need real valid helicity projections for the warmup
            # call, not an arbitrary placeholder: they're looked up as dict
            # keys against the set of physically valid helicity combinations,
            # so e.g. 1 is invalid for any spin != 1/2 and raises KeyError.
            warmup_overrides = {"h0": self.root_resonance.quantum_numbers.angular.projections(return_int=True)[0]}
            warmup_overrides.update({
                lam_name: self.reference.final_state_qn[node].angular.projections(return_int=True)[0]
                for lam_name, node in zip(final_state_lambdas, sorted_final_state_nodes)
            })
            _warmup(func, argnames, overrides=warmup_overrides)
            return func, ["h0", *final_state_lambdas], argnames[len(final_state_lambdas) + 1:]

        names = ["momenta", "h0", *final_state_lambdas]
        func, argnames = _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
        return func, ["h0", *final_state_lambdas], argnames[len(final_state_lambdas) + 2:]

    def matrix_function(self, ls_couplings: dict, complex_couplings: bool = True, static_momenta=None) -> tuple[Callable, list[str]]:
        """
        Returns (func, param_names) where func(momenta, h0, *params) -> helicity matrix dict.

        If `static_momenta` is given, momenta is baked in as a fixed dataset
        the same way as in unpolarized_amplitude: func(h0, *params) does not
        take momenta at all, and is pre-compiled before being returned. See
        unpolarized_amplitude for details.
        """
        # See unpolarized_amplitude: built fresh per call, never stored on
        # self, so this build can't affect any other function built from
        # this combiner.
        momenta_cache = self._static_cache(static_momenta) if static_momenta is not None else None

        def fun(arguments: dict):
            momenta = static_momenta if static_momenta is not None else arguments.pop("momenta")
            h0 = arguments["h0"]
            cache: dict = {}
            return self.combined_matrix(h0, arguments, momenta, cache=cache, momenta_cache=momenta_cache)

        if static_momenta is not None:
            names = ["h0"]
            func, argnames = _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
            # h0 flows into wigner_capital_d's underlying _wigner_d_coefficients,
            # which does Python-level integer comparisons/lru_cache lookups --
            # it must be a static_argnum or jax.jit fails with "unhashable
            # type: DynamicJaxprTracer".
            static_argnums = (argnames.index("h0"),)
            func = jax.jit(func, static_argnums=static_argnums)
            func = _no_momenta_guard(func)
            warmup_overrides = {"h0": self.root_resonance.quantum_numbers.angular.projections(return_int=True)[0]}
            _warmup(func, argnames, overrides=warmup_overrides)
            return func, argnames

        names = ["momenta", "h0"]
        return _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
