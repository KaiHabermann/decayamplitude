# Welcome to the decayangle software Project

[![PyPI - Version](https://img.shields.io/pypi/v/decayamplitude.svg)](https://pypi.org/project/decayamplitude/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayamplitude.svg)](https://pypi.org/project/decayamplitude/)
[![codecov](https://codecov.io/gh/KaiHabermann/decayamplitude/graph/badge.svg?token=KXBO8KEQ3V)](https://codecov.io/gh/KaiHabermann/decayamplitude)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13122268.svg)](https://doi.org/10.5281/zenodo.13122268) -->

---

**Table of Contents**

- [Installation](#installation)
- [Goal](#goal)
- [Related projects](#related-projects)
- [License](#license)

## Installation

```console
pip install decayamplitude
```

## Goal

The software project `decayamplitude` provides an amplitude package working in tandem with `decayangle` to build full cascade reaction amplitudes. 

## Related projects

Amplitude analyses dealing with non-zero spin of final-state particles have to implement wigner rotations in some way.
However, there are a few projects addressing these rotations explicitly using analytic expressions in [DPD paper](https://inspirehep.net/literature/1758460), derived for three-body decays:

- [ThreeBodyDecays.jl](https://github.com/mmikhasenko/ThreeBodyDecays.jl),
- [SymbolicThreeBodyDecays.jl](https://github.com/mmikhasenko/SymbolicThreeBodyDecays.jl),
- [ComPWA/ampform-dpd](https://github.com/ComPWA/ampform-dpd).
  Consistency of the `decayangle` framework with these appoaches is validated in the tests.

## License

`decayamplitude` is distributed under the terms of the [MIT](https://mit-license.org/) license.
