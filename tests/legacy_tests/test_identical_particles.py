from decayamplitude.particle import Particle, DecaySetup


def replace_in_tupe(tpl, n1, n2):
    """
    Replace all occurrences of n1 with n2 in the nested tuple.
    """
    if not isinstance(tpl, tuple):
        return n2 if tpl == n1 else tpl
    return tuple(replace_in_tupe(x, n1, n2) for x in tpl)

def test_3_body():
    final_state = {
        1: Particle(spin=1, parity=-1, type_id=1),
        2: Particle(spin=0, parity=1),
        3: Particle(spin=1, parity=-1, type_id=1),
    }
    mother = Particle(spin=0, parity=1)

    final_state_no_type_id = {
        1: Particle(spin=1, parity=-1),
        2: Particle(spin=0, parity=1),
        3: Particle(spin=1, parity=-1),
    }

    setup = DecaySetup(final_state_particles=final_state)
    tuples = tuple(
        topo.tuple for topo in setup.topologies
    )
    # ab == ac
    assert len(set(tuples)) - len(set(replace_in_tupe(tuples, 3, 1))) == 1

def test_4_body():
        
    final_state = {
        1: Particle(spin=1, parity=-1, type_id=1),
        2: Particle(spin=0, parity=1),
        3: Particle(spin=1, parity=-1, type_id=1),
        4: Particle(spin=0, parity=1),
    }
    mother = Particle(spin=0, parity=1)

    setup = DecaySetup(final_state_particles=final_state)
    tuples = tuple(
        topo.tuple for topo in setup.topologies
    )
    
    # Removing one final state particle and replacing it with another one
    # Assume 0 -> abcd   and b = c
    # Reduces the amount of unique topologies by 3
    # ((ab), (cd)) == ((ac), (bd)),
    # (((ab) c) d) == (((ac) b) d)
    # ()
    assert len(set(tuples)) - len(set(replace_in_tupe(tuples, 3, 1))) == 5


def test_5_body():
    final_state = {
        1: Particle(spin=1, parity=-1, type_id=1),
        2: Particle(spin=0, parity=1),
        3: Particle(spin=1, parity=-1, type_id=1),
        4: Particle(spin=0, parity=1),
        5: Particle(spin=0, parity=1),
    }
    mother = Particle(spin=0, parity=1)

    setup = DecaySetup(final_state_particles=final_state)
    tuples = tuple(
        topo.tuple for topo in setup.topologies
    )
    
    # Removing one final state particle and replacing it with another one
    # Assume 0 -> abcde   and b = c
    assert len(set(tuples)) - len(set(replace_in_tupe(tuples, 3, 1))) == 42