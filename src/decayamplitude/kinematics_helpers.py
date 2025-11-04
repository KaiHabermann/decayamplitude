from decayangle.decay_topology import Node
from decayangle.kinematics import mass
import numpy as np

def mass_from_node(node: Node, momenta):
    """
    Calculate the mass of a particle given its node and momenta.

    Parameters:
    - Node: The node representing the particle.
    - momenta: A dictionary containing the momenta of the particles.

    Returns:
    - The mass of the particle.
    """
    # copy the node
    node = Node(node.tuple)
    
    flat = tuple(np.array(node.tuple).flatten())

    return mass(
        sum(
            momenta[i] for i in flat
        )
    )
