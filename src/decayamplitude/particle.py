from typing import Any
from decayamplitude.rotation import QN, Angular
from decayangle.decay_topology import Topology, TopologyCollection


class Particle(QN):
    """
    Particle class allows to store additional attributes to a set of quantum numbers.
    This is meant to be used to describe final state particles. Mainly to handle the case of indistignuishable particles
    in the final state. In this case the ordering of internal nodes is important, since the identical particles must allways be in the same relative order to the other particles in internal nodes.
    Otherwise one needs to apply a transformation to the couplings, wich would be non-trivial to ensure correctness on through the entire decay tree.
    """

    global_names = {}

    @classmethod
    def add_named_partice(cls, name):
        if name in cls.global_names:
            return cls.global_names[name]
        if len(cls.global_names) == 0:
            new_id = 0
        else:
            new_id = max(cls.global_names.values()) + 1
        cls.global_names[name] = new_id
        return new_id

    def __init__(self, spin: Angular | int = None, parity: int = None, quantum_numbers: QN = None, type_id: int = None, name=None):
        if quantum_numbers is None:
            if spin is None or parity is None:
                raise ValueError("Either quantum_numbers or spin and parity must be provided.")
            self.quantum_numbers = QN(angular_momentum=spin, parity=parity)
        else:
            self.quantum_numbers = quantum_numbers
        self.angular = self.quantum_numbers.angular
        self.parity = self.quantum_numbers.parity
        self.type_id = type_id
        self.name = name
        if self.type_id is None and self.name is not None:
            self.type_id = Particle.add_named_partice(self.name)

class DecaySetup:
    def __init__(self, final_state_particles: dict[int, Particle]):
        self.final_state_particles = final_state_particles

        if any(p.type_id for p in self.final_state_particles.values()):
            print(f"At least one particle has a given type_id.")
        for i, p in self.final_state_particles.items():
            if p.type_id is None: 
                p.type_id = i

        # self.sorting_function = lambda x: x if not isinstance(x, int) else self.final_state_particles[x].type_id
        def sorting_function(x):
            def key(x):
                if not isinstance(x, int):
                    return -len(x) * 10000 + key(x[0])
                else:
                    return self.final_state_particles[x].type_id
            if isinstance(x, tuple) or isinstance(x, list):
                return tuple(sorted(x, key=key))
            else:
                return key(x)

        self.sorting_function = sorting_function
        self.tc = TopologyCollection(0, list(self.final_state_particles), ordering_function=self.sorting_function)
    
    @property
    def topologies(self) -> Topology:
        return self.tc.topologies

    def filled_topologies(self, resonances: dict[tuple[int, ...], Any]):
        toplogies = self.topologies
        def flat(tpl):
            if isinstance(tpl, tuple) or isinstance(tpl, list):
                for item in tpl:
                    yield from flat(item)
            else:
                yield tpl

        def topology_filter(topo: Topology):
            resonances_internal = {
                tuple(sorted(flat(k))): v for k, v in resonances.items()
            }
            topo_nodes = [
                tuple(sorted(flat(node.value))) for node in topo.nodes.values() if not node.final_state
            ]

            return all(
                len(resonances_internal.get(node, [])) > 0 for node in topo_nodes
            )

        return list(
            filter(topology_filter,
                toplogies
            )
        )

