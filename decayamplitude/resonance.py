from decayangle.decay_topology import Node
from decayamplitude.rotation import QN, Angular
from typing import Union, Callable

class Resonance:
    def __init__(self, node:Node, spin:Union[Angular, int] = None, parity:int = None, quantum_numbers:QN = None) -> None:
        self.node = node
        if quantum_numbers is None:
            if spin is None or parity is None:
                raise ValueError("Either quantum numbers or spin and parity must be provided")
            self.quantum_numbers = QN(spin, parity)
        else:
            self.quantum_numbers = quantum_numbers

    @property
    def lineshape(self) -> Callable:
        return self.__lineshape
    
    @lineshape.setter
    def lineshape(self, lineshape_function:Callable):
        self.__lineshape = lineshape_function