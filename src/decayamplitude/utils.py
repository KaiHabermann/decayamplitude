from typing import Callable

def _create_function(names: list[str], ls_couplings: dict[int, dict[str, dict[tuple, float]]], f, complex_couplings=False) -> tuple[Callable, list[str]]:
    from decayamplitude.resonance import LSTuple, Resonance
    import inspect
    import types
    # Create a function signature dynamically
    
    coupling_names = []
    coupling_structure = {}
    for resonance_id, coupling_dict in ls_couplings.items():
        coupling_structure[resonance_id] = {}
        for key, _ in coupling_dict["couplings"].items():
            resonance = Resonance.get_instance(resonance_id)
            name = f"{resonance.descriptor}_{'LS' if resonance.scheme == 'ls' else 'H'}_{'_'.join([sanitize(str(k)) for k in key])}"
            if complex_couplings:
                name_real = f"{name}_real"
                name_imag = f"{name}_imaginary"
                coupling_names.append(name_real)
                coupling_names.append(name_imag)
            else:
                coupling_names.append(name) # we need only define a name 
            coupling_structure[resonance_id][key] = name
    full_names = names + coupling_names
    names_with_duplicates = full_names.copy()
    full_names = list(set(full_names)) # remove duplicates, since the same decay process can exist in multiple chains
    # Sort the names to ensure consistent ordering as given from the outside
    full_names.sort(key=lambda x: names_with_duplicates.index(x))
    # Define a generic function that accepts *args
    def func(*args, **kwargs):
        named_map = {name: arg for name, arg in zip(full_names, args)}
        named_map.update(kwargs)
        couplings = {}

        def compute_coupling(name):
            if complex_couplings:
                return named_map[f"{name}_real"] + 1j * named_map[f"{name}_imaginary"]
            else:
                return named_map[name]

        for resonance_id, coupling_dict in coupling_structure.items():

            couplings[resonance_id] = {"couplings":{
                key: compute_coupling(coupling_dict[key]) for key in coupling_dict
            }}
        arguments = named_map.copy()
        arguments.update(couplings)
        return f(arguments)

    # Assign the generated signature to the function
    # we use the set to remove duplicates. These can exist, if the same decay process exists in multiple chains
    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in full_names]
    sig = inspect.Signature(parameters)
    func.__signature__ = sig
    return func, full_names.copy()

def _no_momenta_guard(func):
    """Wrap a static-momenta creator function so misuse -- calling it as if it
    still needed momenta -- fails loudly instead of silently misassigning the
    first fit-parameter slot to a momenta dict. static_momenta mode drops
    "momenta" from the generated signature entirely (see unpolarized_amplitude
    etc.), so both a `momenta=` kwarg and a dict passed positionally first are
    always a caller mistake.
    """
    def wrapped(*args, **kwargs):
        if "momenta" in kwargs:
            raise TypeError(
                "This function was built with static_momenta and does not take a 'momenta' argument -- "
                "momenta is already baked in. Remove the 'momenta' keyword argument."
            )
        if args and isinstance(args[0], dict):
            raise TypeError(
                "This function was built with static_momenta and does not take momenta as its first "
                "argument -- momenta is already baked in. Pass only the fit parameters."
            )
        return func(*args, **kwargs)
    wrapped.__signature__ = func.__signature__
    wrapped.__wrapped__ = func  # keeps the underlying jax.jit object introspectable (e.g. .lower())
    return wrapped


def _warmup(func, argnames, overrides=None):
    """Force jax.jit to trace and compile `func` now, synchronously, instead
    of lazily on the first real call -- used by static_momenta mode so the
    (expensive, one-time) compile happens while building the function, not
    silently on whatever call happens to be first in a fit.

    Dummy values: 1.0 for everything by default (couplings and lineshape
    parameters). `overrides` supplies exact values for specific argument
    names -- required for h0/h_<n> helicity arguments, since those get used
    as dict keys against the set of physically valid helicity combinations
    (not just arithmetic inputs): an arbitrary placeholder like 1 is only
    valid for spin-1/2 and raises KeyError for any other spin (e.g. a spin-1
    particle's helicities are -2/0/2 in value2 convention, never 1). Callers
    with the relevant quantum numbers (see ChainCombiner.polarized_amplitude/
    matrix_function) must pass real valid projections via `overrides`.
    """
    import jax
    overrides = overrides or {}
    dummy_args = tuple(overrides.get(name, 1.0) for name in argnames)
    result = func(*dummy_args)
    jax.block_until_ready(result)


def sanitize(name: str) -> str:
    """
    Sanitize a name for use in python code
    """
    replacements = [
        ("*", "star"),
        ("(", ""),
        (")", ""),
        ("[", ""),
        ("]", ""),
        ("{", ""),
        ("}", ""),
    ] + [
        (a, "_") for a in " .,/\\^'\"~!?=>|&$#@%;:`´§°"
    ]
    for replacement in replacements:
        name = name.replace(*replacement)
    name = name.replace("+", "plus")
    name = name.replace("-", "minus")
    
    return name
    
    