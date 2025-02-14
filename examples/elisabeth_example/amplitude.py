from generate_momenta import make_four_vectors_from_dict

from decayangle.decay_topology import Topology, TopologyCollection, HelicityAngles
from decayangle.lorentz import LorentzTrafo
from decayangle.config import config as decayangle_config
import numpy as np 

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, Angular
from decayamplitude.chain import DecayChain, MultiChain
from decayamplitude.combiner import ChainCombiner

from decayamplitude.backend import numpy as np



from decayangle.decay_topology import Topology, Node
from decayangle.kinematics import mass
from decayangle.config import config as decayangle_config


def constant_lineshape(*args):
    return 1

decayangle_config.sorting = "off"

tg = TopologyCollection(
    0,
    topologies=[
        Topology(0, decay_topology=((2, 3), 1)),
        Topology(0, decay_topology=((3, 1), 2)),
        Topology(0, decay_topology=((1, 2), 3)),
    ],
)


# Lc -> p K pi
# 0 -> 1 2 3

def read_helicity_angles_from_dict(dtc):
    mappings = {
        ((2, 3), 1): ("Kpi", "theta_Kst", "phi_Kst", "theta_K", "phi_K"),
        ((3, 1), 2): ("pip", "theta_D", "phi_D", "theta_pi", "phi_pi"),
        ((1, 2), 3): ("pK", "theta_L", "phi_L", "theta_p", "phi_p"),
    }

    topos = {}

    for tpl, (name, theta_hat, phi_hat, theta, phi) in mappings.items():
        topos[tpl] = {
            tpl: HelicityAngles(
                dtc[name][phi_hat],
                dtc[name][theta_hat],
            ),
            tpl[0]: HelicityAngles(
                dtc[name][phi],
                dtc[name][theta],
            ),
        }
    return topos


class Amplitude:
    def __init__(self, momenta):
        self.momenta = momenta
        L_1520 = Resonance(Node((1, 2)), quantum_numbers=QN(3, -1), lineshape=constant_lineshape, argnames=[], name = "L_1520", preserve_partity=True, scheme="helicity") # , scheme="helicity"
        Lc = Resonance(Node(0), quantum_numbers=QN(1, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Lc", scheme="helicity") # , scheme="helicity"
        self.resonances =   {
            (1, 2): L_1520,
            0: Lc,
        }
        self.final_state_qn = {
            1: QN(1, 1),
            2: QN(0, 1),
            3: QN(0, 1),
        }
        self.topology = Topology(0, decay_topology=((1, 2), 3))
        self.chain = DecayChain(
                topology = self.topology,
                resonances = self.resonances,
                momenta = momenta,
                final_state_qn = self.final_state_qn,
            )
        self.combiner = ChainCombiner([self.chain, ])
        
        for k,v in self.combiner.generate_couplings().items():
            print(k)
            for hel_tuple, val in v["couplings"].items():
                print(hel_tuple, val)
        matrix_function, matrix_argnames = self.combiner.matrix_function(self.combiner.generate_couplings())
        print(matrix_argnames)
        
        couplings = {
            "Lc_H_-3_0": 1,
            "Lc_H_-1_0": 1,
            "Lc_H_1_0": 1,
            "Lc_H_3_0": 1,
            "L_1520_H_1_0": -1,
            "L_1520_H_-1_0": 1,
        }

        # couplings = {

        # }

        for helicities, amplitude in matrix_function(1, **couplings).items():
            print(helicities, amplitude)
        for helicities, amplitude in matrix_function(-1, **couplings).items():
            print(helicities, amplitude)
        self.value = None


    


def test_elisabeth():
    import json

    path = "examples/test_data/Parsed_ccp_kinematics_100events.json"
    with open(path, "r") as f:
        data = json.load(f)
    for k, dtc in data.items():
        kwargs = {k: v for k, v in dtc["kinematic"].items() if k != "mkpisq" }
        momenta = make_four_vectors_from_dict(**dtc["chain_variables"]["Kpi"], **kwargs)
        amplitude = Amplitude(momenta)
        print(amplitude.value)
        exit(0)

if __name__ =="__main__":
    test_elisabeth()