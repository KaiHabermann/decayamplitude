from typing import Callable

def _create_function(names:list[str], ls_couplings:dict[int, dict[str: dict[tuple, float]]], f, complex_couplings=False) -> tuple[Callable, list[str]]:
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
            name = f"{resonance.descriptor}_{'LS' if resonance.scheme == 'ls' else 'H'}_{'_'.join([str(k) for k in key])}"
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
    return func, full_names

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
        (a, "_") for a in " -+.,/\\^'\"~!?=>|&$#@%;:`´§°"
    ]
    for replacement in replacements:
        name = name.replace(*replacement)
    
    return name
    
    