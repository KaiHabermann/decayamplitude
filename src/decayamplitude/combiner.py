from decayamplitude.chain import DecayChain, AlignedChain, MultiChain, AlignedMultiChain
from decayangle.decay_topology import Topology
from typing import Union, Callable
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
    def combined_function(self):
        """
        Returns a function that combines the amplitudes of all chains
        """
        def f(h0, lambdas:dict, arguments:dict):

            amplitudes = [
                chain.aligned_matrix(h0, arguments)[tuple(lambdas[k] for k in sorted(lambdas.keys()))]
                for chain in self.aligned_chains
            ]
            return sum(amplitudes) + self.reference.chain_function(h0, lambdas, arguments)

        return f
    
    def polarized_amplitude(self, ls_couplings:dict[int, dict[str: dict[LSTuple, float]]]) -> tuple[Callable, list[str], list[str]]:
        """
        Returns a function that combines the amplitudes of all chains
        """
        sorted_final_state_nodes = sorted([n.node.value for n in self.reference.final_state_nodes])
        final_state_lambdas = sorted([f"h_{n}" for n in sorted_final_state_nodes]) 
        def fun(arguments:dict):
            # build lambda dict, as it is used internally from plain parameters
            h0 = arguments.pop("h0")
            lambdas = {n: arguments.pop(k) for k, n  in zip(final_state_lambdas, sorted_final_state_nodes)}
            return self.combined_function(h0, lambdas, arguments)
        polarized, argnames = _create_function(["h0", *final_state_lambdas] + self.resonance_params, ls_couplings, fun)

        return polarized, ["h0", *final_state_lambdas], argnames[len(final_state_lambdas)+1:]

    
    @property
    def combined_matrix(self) -> Callable:
        """
        Returns a function that combines the matrices of all chains.
        The final matrix will be a sum of all matrices, where the alignment is already performed.
        """
        def matrix(h0, arguments:dict) -> dict:
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
    
    def matrix_function(self, ls_couplings:dict[int, dict[str: dict[LSTuple, float]]], complex_couplings: bool=True) -> tuple[Callable, list[str]]:
        """
        Returns a function that combines the matrices of all chains.
        The final matrix will be a sum of all matrices, where the alignment is already performed.
        """
        if "h0" in self.resonance_params:
            raise ValueError("The parameter name 'h0' is reserved for the helicity quantum number of the mother particle. Please choose another name for the resonance parameter.")
        def fun(arguments:dict):
            h0 = arguments["h0"]
            return self.combined_matrix(h0, arguments)
        return _create_function(["h0"] + self.resonance_params, ls_couplings, fun, complex_couplings=complex_couplings)
    
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

    def unpolarized_amplitude(self, ls_couplings: dict, complex_couplings=True) -> tuple[Callable, list[str]]:

        if self.root_resonance is None:
            raise ValueError(f"The root resonance must be the same for all chains! Root = {self.reference.topology.root}.")

        def f(arguments:dict):
            return sum(
                    abs(v)**2 
                    for h0 in self.root_resonance.quantum_numbers.angular.projections()
                    for v in self.combined_matrix(h0, arguments).values()
                )

        return _create_function(self.resonance_params, ls_couplings, f, complex_couplings=complex_couplings)
        

        

