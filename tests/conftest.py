"""
decayangle's decay_topology module snapshots `cb = cfg.backend` at import
time, so the backend must be selected before decayangle is first imported
anywhere in the process. Individual test modules set
`decayangle_config.backend` themselves, but that only has an effect if they
happen to be the first module pytest imports -- otherwise whichever test
file collects first silently wins for the entire session, regardless of
what later modules request.

Setting it here, in conftest.py, guarantees it happens before pytest
imports any test module, independent of collection order.
"""
from decayangle.config import config as decayangle_config

decayangle_config.backend = "jax"
decayangle_config.sorting = "value"
decayangle_config.use_rust = False
