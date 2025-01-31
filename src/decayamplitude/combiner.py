from decayamplitude.chain import DecayChain, AlignedChain, MultiChain, AlignedMultiChain
from decayangle.decay_topology import Topology, HelicityAngles


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
    def combined_function(self):
        """
        Returns a function that combines the amplitudes of all chains
        """
        def f(h0, lambdas:dict, helicity_angles:dict[tuple, HelicityAngles], arguments:dict):

            amplitudes = [
                chain.aligned_matrix(h0, helicity_angles, arguments)[tuple(lambdas[k] for k in sorted(lambdas.keys()))]
                for chain in self.aligned_chains
            ]
            return sum(amplitudes) + self.reference.chain_function(h0, lambdas, helicity_angles, arguments)

        return f
    
    @property
    def combined_matrix(self):
        """
        Returns a function that combines the matrices of all chains.
        The final matrix will be a sum of all matrices, where the alignment is already performed.
        """
        def matrix(h0, helicity_angles:dict[tuple, HelicityAngles], arguments:dict) -> dict:
            matrices = [
                chain.aligned_matrix(h0, helicity_angles, arguments)
                for chain in self.aligned_chains
            ]
            matrices.append(self.reference.matrix(h0, helicity_angles, arguments))

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