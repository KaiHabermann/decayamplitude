from decayangle.decay_topology import Node
from decayangle.kinematics import mass


def flatten(t):
    if isinstance(t, tuple):
        for item in t:
            yield from flatten(item)
    else:
        yield t


def mass_from_node(node: Node, momenta):
    """
    Calculate the mass of a particle given its node and momenta.

    Parameters:
    - Node: The node representing the particle.
    - momenta: A dictionary containing the momenta of the particles.

    Returns:
    - The mass of the particle.
    """
    return mass(
        sum(
            momenta[i] for i in flatten(node.tuple)
        )
    )
