from decayamplitude.chain import DecayChain, AlignedChain, MultiChain, AlignedMultiChain
from decayangle.decay_topology import Topology
from typing import Union, Callable


class ChainCombiner:
    """
    Class to automatically combine multiple decay chains into a single amplitude.
    The first chain is used as a reference for the topology.
    All other chains will be transformed into the reference basis.
    """

    def __init__(self, chains: list[DecayChain | MultiChain]) -> None:
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
    
    @property
    def combined_matrix(self):
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
    
    def generate_ls_couplings(self):
        """
        Generates the couplings for the ls basis.
        """
        couplings = {}
        for chain in self.chains:
            couplings.update(chain.generate_ls_couplings())
        return couplings
    
    def unpolarized_amplitude(self, ls_couplings: dict) -> tuple[Callable, list[str]]:
        import inspect
        import types
        if self.root_resonance is None:
            raise ValueError(f"The root resonance must be the same for all chains! Root = {self.reference.topology.root}.")

        coupling_names = []
        coupling_structure = {}
        for resonance_id, coupling_dict in ls_couplings.items():
            coupling_structure[resonance_id] = {}
            for key, _ in coupling_dict["ls_couplings"].items():
                name = f"COUPLING_ID_{resonance_id}_LS_{'_'.join([str(k) for k in key])}"
                coupling_names.append(name) # we need only define a name 
                coupling_structure[resonance_id][key] = name

        def create_function(names):
            # Create a function signature dynamically
            parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names]
            sig = inspect.Signature(parameters)

            # Define a generic function that accepts *args
            def func(*args, **kwargs):
                named_map = {name: arg for name, arg in zip(names, args)}
                named_map.update(kwargs)
                couplings = {}
                for resonance_id, coupling_dict in coupling_structure.items():
                    couplings[resonance_id] = {"ls_couplings":{
                        key: named_map[coupling_dict[key]] for key in coupling_dict
                    }}
                kwargs = named_map.copy()
                kwargs.update(couplings)

                return sum(
                    abs(v)**2 
                    for h0 in self.root_resonance.quantum_numbers.angular.projections()
                    for v in self.combined_matrix(h0, kwargs).values()
                )

            # Assign the generated signature to the function
            func.__signature__ = sig
            return func
        
        resonances = [resonance for chain in self.chains for resonance in chain.resonance_list]
        resonance_parameter_names = [name for resonance in resonances for name in resonance.parameter_names]

        if len(set(resonance_parameter_names)) != len(resonance_parameter_names):
            from collections import Counter
            c = Counter(resonance_parameter_names)
            raise ValueError(f"Parameter names are not unique: {', '.join([name for name, count in c.items() if count > 1])}")

        total_names = coupling_names + resonance_parameter_names
        f = create_function(total_names)
        return f, total_names
        

        

