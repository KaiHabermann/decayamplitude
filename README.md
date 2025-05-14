# Welcome to the decayamplitude software Project

[![PyPI - Version](https://img.shields.io/pypi/v/decayamplitude.svg)](https://pypi.org/project/decayamplitude/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayamplitude.svg)](https://pypi.org/project/decayamplitude/)
[![codecov](https://codecov.io/gh/KaiHabermann/decayamplitude/graph/badge.svg?token=KXBO8KEQ3V)](https://codecov.io/gh/KaiHabermann/decayamplitude)


---

**Table of Contents**

- [Installation](#installation)
- [Goal](#goal)
- [Related projects](#related-projects)
- [How to cite](#how-to-cite)
- [License](#license)

## Installation

```console
pip install decayamplitude
```

## Goal

The software project `decayamplitude` provides an amplitude package working in tandem with `decayangle` to build full cascade reaction amplitudes. 
The main goal is to provide a simple interface to build decay chains and their amplitudes, which can be used in amplitude analyses. 

## Online Decay Editor

There is a beta version of a web based decay editor, which is hosted [here](https://kaihabermann.github.io/DecaySelector/).
The Website allows for an easy selection of a decay and resonances. Once finished a script generating an amplitude with `decayamplitude` can be downloaded.
This is usually the easiest and quickest way to get a working amplitude. Explicit lineshapes need then to be defined by the user, where it is marked in the downloaded script.

## Usage
```python
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, Angular
from decayamplitude.chain import DecayChain, MultiChain
from decayamplitude.combiner import ChainCombiner




from decayangle.decay_topology import Topology, Node, TopologyCollection

# First we define the final state
final_state_qn = {
        1: QN(1, 1),
        2: QN(2, 1),
        3: QN(0, 1)
    }

# decayangles Topology class is used to generate the angular variables. Be carefull here, as fit results may depend on the ordering. 

topology1 = Topology(
    0,
    decay_topology=((2,3), 1)
)
topology2 = Topology(
    0,
    decay_topology=((1, 2), 3)
)

# alternatively we can use the topology collection to define all topologies automatically
tg = TopologyCollection(
    0,
    [1,2,3]
)
tg.topologies
```

We now have a basic definition of our decay scheme. But we need resonances to react. In general for a multi body decay we want to define a list of resonances for each isobar. Then let the code figure out all the possible decay chains.

```python
resonances1 = {
    # we need to define resonances for each internal node of the decay chain
    # A resonance needs a node, to which it couples, quantum numbers, a lineshape
    # and a list of argument names, which will be used to map the parameters of the resonance 
    # to the outside, once all our configurations are combined into an amplitude
    (2,3): Resonance(topology1.nodes[(2, 3)], 
                    quantum_numbers=QN(0, -1), 
                    lineshape=BW_lineshape(topology1.nodes[(2, 3)].mass(momenta)), 
                    argnames=["gamma1", "m01"]),

    # since we can not only model decays, but also transitions from 2 particle
        # to many particles
    # Thus we may have to model multiple initial resonances and their production
        # with a lineshape
    # For decays we can use a constant lineshape 
    # The preserve_parity flag controlls which L S couplings will be produced, 
        # if one uses the automated generation of the possible couplings
    # This has no effect if one plugs in couplings by hand
    0: Resonance(Node(0), quantum_numbers=QN(1, 1),
    lineshape=constant_lineshape, 
    argnames=[], 
    preserve_partity=False)

# we use the argnames, to use the same mass for both BW lineshapes, but different widths
resonances2 = {
    (1, 2): Resonance(nodes_2[(1, 2)], quantum_numbers=QN(3, -1), lineshape=BW_lineshape(nodes_2[(1, 2)].mass(momenta)), argnames=["gamma2", "m01"]),
    0: Resonance(Node(0), quantum_numbers=QN(1, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }
}
```

Now we can take the resonance dictionaries and combine them with a topology in order to produce a `DecayChain`. 

```python
chain1 = DecayChain(topology = topology1,
        resonances = resonances1,
        momenta = momenta,
        final_state_qn = final_state_qn)

chain2 = DecayChain(topology = topology2,
        resonances = resonances2,
        momenta = momenta,
        final_state_qn = final_state_qn)
```
These chains are already able to produce a chain function, which can be evaluated for a given set of parameters. 

```python
chain1.chain_function
```

We can now combine the chains into a `ChainCombiner` object. This object will internally take care of the alignmeht operations.

```python
combined = ChainCombiner([chain1, chain2])

# we can generate the unpolarized amplitude
unpolarized, argnames = combined.unpolarized_amplitude(combined.generate_couplings())
# unpolarized(*args) will return the amplitude for the given parameters

# we can generate the matix elements depending on the polarization of the particles
polarized, lambdas, argnames = combined.polarized_amplitude(combined.generate_couplings())
# polarized(*lambdas, *args) will return the amplitude for the given parameters and polarization

# here we get a list of aditional function parameters, which represent the polarization of the initial and final state particles
# lambdas = ["h_0", "h_1", "h_2", "h_3"]

# alternatively we can produce all matrix elements at once
matrx_function, matrix_argnames = combined.matrix_function(combined.generate_couplings())
# here we only have the initial state polarization as an additional parameter
# matrix_argnames = ["h_0"] + argnames
# matrx_function(h_0, *args) will return the amplitude for the given parameters and polarization

```

## Related projects

Amplitude analyses dealing with non-zero spin of final-state particles have to implement wigner rotations in some way.
However, there are a few projects addressing these rotations explicitly using analytic expressions in [DPD paper](https://inspirehep.net/literature/1758460), derived for three-body decays:

- [ThreeBodyDecays.jl](https://github.com/mmikhasenko/ThreeBodyDecays.jl),
- [SymbolicThreeBodyDecays.jl](https://github.com/mmikhasenko/SymbolicThreeBodyDecays.jl),
- [ComPWA/ampform-dpd](https://github.com/ComPWA/ampform-dpd).

## How to cite

This software package as well as the closely related [decayangle](https://github.com/KaiHabermann/decayangle) should be cited by the accompaning paper
[Wigner Rotations for Cascade Reactions](https://doi.org/10.1103/PhysRevD.111.056015)

## License

`decayamplitude` is distributed under the terms of the [MIT](https://mit-license.org/) license.
