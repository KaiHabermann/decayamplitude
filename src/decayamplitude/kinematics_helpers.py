from decayangle.decay_topology import Node

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

    # construct the topology
    Node.construct_topology(node, node.tuple)

    return node.mass(momenta=momenta)
