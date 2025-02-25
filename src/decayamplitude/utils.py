from typing import Callable
from decayamplitude.resonance import LSTuple, Resonance

def _create_function(names:list[set], ls_couplings:dict[int, dict[str: dict[LSTuple, float]]], f) -> Callable:
    import inspect
    import types
    # Create a function signature dynamically
    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names]
    sig = inspect.Signature(parameters)
    coupling_names = []
    coupling_structure = {}
    for resonance_id, coupling_dict in ls_couplings.items():
        coupling_structure[resonance_id] = {}
        for key, _ in coupling_dict["couplings"].items():
            resonance = Resonance.get_instance(resonance_id)
            if resonance.name is None:
                name = f"COUPLING_ID_{resonance_id}_{'LS' if resonance.scheme == 'ls' else 'H'}_{'_'.join([str(k) for k in key])}"
            else:
                name = f"{resonance.sanitized_name}_{'LS' if resonance.scheme == 'ls' else 'H'}_{'_'.join([str(k) for k in key])}"
            coupling_names.append(name) # we need only define a name 
            coupling_structure[resonance_id][key] = name
    full_names = names + coupling_names
    # Define a generic function that accepts *args
    def func(*args, **kwargs):
        named_map = {name: arg for name, arg in zip(full_names, args)}
        named_map.update(kwargs)
        couplings = {}
        for resonance_id, coupling_dict in coupling_structure.items():
            couplings[resonance_id] = {"couplings":{
                key: named_map[coupling_dict[key]] for key in coupling_dict
            }}
        arguments = named_map.copy()
        arguments.update(couplings)
        return f(arguments)

    # Assign the generated signature to the function
    func.__signature__ = sig
    return func, full_names
    