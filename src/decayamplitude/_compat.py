"""
Backwards compatibility helpers for the decayamplitude <= 0.1.9 API.

Up to 0.1.9 the four-momenta were bound to a chain at construction time, e.g.

    DecayChain(topology, resonances, momenta, final_state_qn)

Since 0.2.0 momenta are passed at call time instead. The helpers in this module
allow the old style to keep working.
"""
import functools
import inspect
import warnings

from decayamplitude.rotation import QN

_construction_depth = 0


def _is_final_state_qn(obj) -> bool:
    """final_state_qn dicts map particle keys to QN (or Particle) objects, momenta dicts map to arrays."""
    return isinstance(obj, dict) and len(obj) > 0 and all(isinstance(v, QN) for v in obj.values())


def accepts_legacy_momenta(legacy_order: list[str]):
    """
    Decorator for ``__init__`` methods that accept ``momenta`` as an optional keyword argument.

    ``legacy_order`` is the positional parameter order of the <=0.1.9 signature (without ``self``).
    If a positional call follows that order, which is detected by the argument in the ``momenta``
    slot not being a final-state quantum-number dict, the positional arguments are re-mapped to keywords.
    A DeprecationWarning is emitted once per user-facing construction when momenta are bound.
    """
    momenta_index = legacy_order.index("momenta")

    def decorator(init):
        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            global _construction_depth
            if len(args) > momenta_index and not _is_final_state_qn(args[momenta_index]):
                if len(args) > len(legacy_order):
                    raise TypeError(f"{type(self).__name__}() got too many positional arguments")
                for name, value in zip(legacy_order, args):
                    if name in kwargs:
                        raise TypeError(f"{type(self).__name__}() got multiple values for argument '{name}'")
                    kwargs[name] = value
                args = ()
            if _construction_depth == 0 and kwargs.get("momenta") is not None:
                warnings.warn(
                    f"Passing momenta to {type(self).__name__}() at construction is deprecated since decayamplitude 0.2.0. "
                    "Pass the momenta when calling the amplitude functions instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _construction_depth += 1
            try:
                init(self, *args, **kwargs)
            finally:
                _construction_depth -= 1

        wrapper.__signature__ = inspect.signature(init)
        return wrapper

    return decorator
