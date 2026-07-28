from decayamplitude.chain import DecayChain, AlignedChain, MultiChain, AlignedMultiChain
from decayangle.decay_topology import Topology
from typing import Union, Callable, Optional
from contextlib import contextmanager, ExitStack
from decayamplitude.resonance import LSTuple, Resonance
from decayamplitude.utils import _create_function

class ChainCombiner:
    """
    Class to automatically combine multiple decay chains into a single amplitude.
    The first chain is used as a reference for the topology.
    All other chains will be transformed into the reference basis.
    """

    def __init__(self, chains: list[Union[DecayChain, MultiChain]]) -> None:
        self.chains = chains
        self.reference = chains[0]
        self.aligned_chains = [
            AlignedMultiChain.from_multichain(
                chain,
                self.reference
            ) if isinstance(chain, MultiChain) else 
                AlignedChain(
                    chain.topology,
                    chain.resonances,
                    chain.momenta,
                    chain.final_state_qn,
                    self.reference
                )
            for chain in chains[1:]
        ]


    @property
    def root_resonance(self):
        if all(chain.root_resonance.quantum_numbers == self.reference.root_resonance.quantum_numbers for chain in self.chains):
            return self.reference.root_resonance
        return None

    @property
    def single_chains(self) -> list[DecayChain]:
        """
        Returns the single chains, by flattening the aligned chains and multi chains into a list of single chains.
        """
        chains = [self.reference] if not hasattr(self.reference, "chains") else [chain for chain in self.reference.chains]
        for aligned_chain in self.aligned_chains:
            if isinstance(aligned_chain, AlignedChain):
                chains.append(aligned_chain)
            elif isinstance(aligned_chain, AlignedMultiChain) :
                chains.extend(aligned_chain.chains)
        return chains

    @property
    def wigner_matrices(self) -> dict:
        """
        Returns a dict with all internal structures holding helicity angles and wigner matrices
        for the reference chain and all aligned chains. These are computed from the momenta.
        The dict can be passed to the amplitude functions (see the `external_wigner` options),
        which makes these values explicit function inputs instead of baked-in constants.
        This allows jax to trace them correctly.
        """
        return {
            "reference": self.reference.wigner_data,
            "aligned": [chain.wigner_data for chain in self.aligned_chains],
        }

    @contextmanager
    def _overridden_wigner_matrices(self, wigner_matrices: Optional[dict]):
        """
        Temporarily replaces the internal structures holding helicity angles and wigner matrices
        of all chains with the ones given in wigner_matrices. The structure has to match the one
        returned by `wigner_matrices`. If None is given, nothing is replaced.
        """
        if wigner_matrices is None:
            yield self
            return
        with ExitStack() as stack:
            if "reference" in wigner_matrices:
                stack.enter_context(self.reference._overridden_wigner_data(wigner_matrices["reference"]))
            for chain, data in zip(self.aligned_chains, wigner_matrices.get("aligned", [])):
                stack.enter_context(chain._overridden_wigner_data(data))
            yield self

    @property
    def combined_function(self):
        """
        Returns a function that combines the amplitudes of all chains.
        The optional wigner_matrices argument allows to pass the helicity angles and wigner
        matrices from the outside, as returned by the `wigner_matrices` property.
        """
        def f(h0, lambdas:dict, arguments:dict, wigner_matrices: Optional[dict]=None):
            with self._overridden_wigner_matrices(wigner_matrices):
                amplitudes = [
                    chain.aligned_matrix(h0, arguments)[tuple(lambdas[k] for k in sorted(lambdas.keys()))]
                    for chain in self.aligned_chains
                ]
                return sum(amplitudes) + self.reference.chain_function(h0, lambdas, arguments)

        return f

    def polarized_amplitude(self, ls_couplings:dict[int, dict[str: dict[LSTuple, float]]], external_wigner: bool=False) -> tuple[Callable, list[str], list[str]]:
        """
        Returns a function that combines the amplitudes of all chains.
        If external_wigner is True, the returned function accepts an additional
        `wigner_matrices` parameter (the internal helicity angles and wigner matrices),
        and the dict with these values is returned as an additional last element.
        """
        sorted_final_state_nodes = sorted([n.node.value for n in self.reference.final_state_nodes])
        final_state_lambdas = sorted([f"h_{n}" for n in sorted_final_state_nodes])
        def fun(arguments:dict):
            # build lambda dict, as it is used internally from plain parameters
            h0 = arguments.pop("h0")
            wigner_matrices = arguments.pop("wigner_matrices", None)
            lambdas = {n: arguments.pop(k) for k, n  in zip(final_state_lambdas, sorted_final_state_nodes)}
            return self.combined_function(h0, lambdas, arguments, wigner_matrices)
        names = ["h0", *final_state_lambdas]
        if external_wigner:
            names.append("wigner_matrices")
        polarized, argnames = _create_function(names + self.resonance_params, ls_couplings, fun)

        if external_wigner:
            return polarized, ["h0", *final_state_lambdas], argnames[len(final_state_lambdas)+1:], self.wigner_matrices
        return polarized, ["h0", *final_state_lambdas], argnames[len(final_state_lambdas)+1:]


    @property
    def combined_matrix(self) -> Callable:
        """
        Returns a function that combines the matrices of all chains.
        The final matrix will be a sum of all matrices, where the alignment is already performed.
        The optional wigner_matrices argument allows to pass the helicity angles and wigner
        matrices from the outside, as returned by the `wigner_matrices` property.
        """
        def matrix(h0, arguments:dict, wigner_matrices: Optional[dict]=None) -> dict:
            with self._overridden_wigner_matrices(wigner_matrices):
                matrices = [
                    chain.aligned_matrix(h0, arguments)
                    for chain in self.aligned_chains
                ]
                matrices.append(self.reference.matrix(h0, arguments))

                return {
                    key: sum(matrix[key] for matrix in matrices)
                    for key in matrices[0].keys()
                }
        return matrix

    def matrix_function(self, ls_couplings:dict[int, dict[str: dict[LSTuple, float]]], complex_couplings: bool=True, external_wigner: bool=False) -> tuple[Callable, list[str]]:
        """
        Returns a function that combines the matrices of all chains.
        The final matrix will be a sum of all matrices, where the alignment is already performed.
        If external_wigner is True, the returned function accepts an additional
        `wigner_matrices` parameter (the internal helicity angles and wigner matrices),
        and the dict with these values is returned as an additional last element.
        """
        if "h0" in self.resonance_params:
            raise ValueError("The parameter name 'h0' is reserved for the helicity quantum number of the mother particle. Please choose another name for the resonance parameter.")
        def fun(arguments:dict):
            h0 = arguments["h0"]
            wigner_matrices = arguments.pop("wigner_matrices", None)
            return self.combined_matrix(h0, arguments, wigner_matrices)
        names = ["h0"] + (["wigner_matrices"] if external_wigner else [])
        func, argnames = _create_function(names + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
        if external_wigner:
            return func, argnames, self.wigner_matrices
        return func, argnames
    
    def generate_couplings(self):
        """
        Generates the couplings for the ls basis.
        """
        couplings = {}
        for chain in self.chains:
            couplings.update(chain.generate_couplings())
        return couplings
    
    @property
    def resonance_params(self) -> list[str]:
        resonance_parameter_names = [name for chain in self.chains for name in chain.resonance_params]

        if len(set(resonance_parameter_names)) != len(resonance_parameter_names):
            from collections import Counter
            c = Counter(resonance_parameter_names)
            # raise ValueError(f"Parameter names are not unique: {', '.join([name for name, count in c.items() if count > 1])}")
        return list(set(resonance_parameter_names))

    def unpolarized_amplitude(self, ls_couplings: dict, complex_couplings=True, external_wigner: bool=False) -> tuple[Callable, list[str]]:
        """
        Returns a function that calculates the unpolarized amplitude summed over all chains.
        If external_wigner is True, the returned function accepts an additional
        `wigner_matrices` parameter (the internal helicity angles and wigner matrices),
        and the dict with these values is returned as an additional last element.
        """
        if self.root_resonance is None:
            raise ValueError(f"The root resonance must be the same for all chains! Root = {self.reference.topology.root}.")

        def f(arguments:dict):
            wigner_matrices = arguments.pop("wigner_matrices", None)
            with self._overridden_wigner_matrices(wigner_matrices):
                return sum(
                        abs(v)**2
                        for h0 in self.root_resonance.quantum_numbers.angular.projections()
                        for v in self.combined_matrix(h0, arguments).values()
                    )

        names = (["wigner_matrices"] if external_wigner else []) + self.resonance_params
        func, argnames = _create_function(names, ls_couplings, f, complex_couplings=complex_couplings)
        if external_wigner:
            return func, argnames, self.wigner_matrices
        return func, argnames
        

        

