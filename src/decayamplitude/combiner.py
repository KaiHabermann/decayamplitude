from decayamplitude.chain import DecayChain, AlignedChain, MultiChain, AlignedMultiChain
from typing import Callable
from decayamplitude.utils import _create_function

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
        """Returns a function f(h0, lambdas, arguments, momenta) that sums the aligned amplitudes of all chains."""
        def f(h0, lambdas: dict, arguments: dict, momenta: dict):
            amplitudes = [
                chain.aligned_matrix(h0, arguments, momenta)[tuple(lambdas[k] for k in sorted(lambdas.keys()))]
                for chain in self.aligned_chains
            ]
            return sum(amplitudes) + self.reference.chain_function(h0, lambdas, arguments, momenta)
        return f

    @property
    def combined_matrix(self) -> Callable:
        """Returns a function f(h0, arguments, momenta) that sums the aligned helicity matrices of all chains."""
        def matrix(h0, arguments: dict, momenta: dict) -> dict:
            matrices = [chain.aligned_matrix(h0, arguments, momenta) for chain in self.aligned_chains]
            matrices.append(self.reference.matrix(h0, arguments, momenta))
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

    def unpolarized_amplitude(self, ls_couplings: dict, complex_couplings=True) -> tuple[Callable, list[str]]:
        """
        Returns (func, param_names) where func(momenta, *params) -> per-event scalar intensity.
        Vectorize over events with jax.vmap(func, in_axes=({k: 0 for k in momenta}, None, ...)).
        """
        if self.root_resonance is None:
            raise ValueError(f"The root resonance must be the same for all chains! Root = {self.reference.topology.root}.")

        def f(arguments: dict):
            momenta = arguments.pop("momenta")
            return sum(
                abs(v)**2
                for h0 in self.root_resonance.quantum_numbers.angular.projections()
                for v in self.combined_matrix(h0, arguments, momenta).values()
            )

        names = ["momenta"] + self.resonance_params
        return _create_function(names, ls_couplings, f, complex_couplings=complex_couplings)

    def polarized_amplitude(self, ls_couplings: dict, complex_couplings: bool = True) -> tuple[Callable, list[str], list[str]]:
        """
        Returns (func, helicity_names, coupling_names) where func(momenta, h0, *h_finals, *params).
        """
        sorted_final_state_nodes = sorted([n.node.value for n in self.reference.final_state_nodes])
        final_state_lambdas = sorted([f"h_{n}" for n in sorted_final_state_nodes])

        def fun(arguments: dict):
            momenta = arguments.pop("momenta")
            h0 = arguments.pop("h0")
            lambdas = {n: arguments.pop(k) for k, n in zip(final_state_lambdas, sorted_final_state_nodes)}
            return self.combined_function(h0, lambdas, arguments, momenta)

        names = ["momenta", "h0", *final_state_lambdas]
        func, argnames = _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
        return func, ["h0", *final_state_lambdas], argnames[len(final_state_lambdas) + 2:]

    def matrix_function(self, ls_couplings: dict, complex_couplings: bool = True) -> tuple[Callable, list[str]]:
        """
        Returns (func, param_names) where func(momenta, h0, *params) -> helicity matrix dict.
        """
        def fun(arguments: dict):
            momenta = arguments.pop("momenta")
            h0 = arguments["h0"]
            return self.combined_matrix(h0, arguments, momenta)

        names = ["momenta", "h0"]
        return _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
