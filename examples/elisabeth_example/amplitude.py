from generate_momenta import make_four_vectors_from_dict
from itertools import product
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
                convention="helicity",
            )
        self.combiner = ChainCombiner([self.chain, ])
        
        for k,v in self.combiner.generate_couplings().items():
            print(k)
            for hel_tuple, val in v["couplings"].items():
                print(hel_tuple, val)
        matrix_function, matrix_argnames = self.combiner.matrix_function(self.combiner.generate_couplings())
        print(matrix_argnames)
        
        couplings_m1_m1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 1,
            "Lc_H_1_0": 0,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0": 0,
            "L_1520_H_-1_0": -1,
        }
        couplings_1_m1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 0,
            "Lc_H_1_0": 1,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0": 0,
            "L_1520_H_-1_0": -1,
        }
        couplings_m1_1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 1,
            "Lc_H_1_0": 0,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0": -1,
            "L_1520_H_-1_0": 0,
        }
        couplings_1_1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 0,
            "Lc_H_1_0": 1,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0": -1,
            "L_1520_H_-1_0": 0,
        }
        # (hlc, l1520)
        self.value = {
            ( 1, -1, -1): matrix_function( 1, **couplings_m1_m1),
            ( 1, -1,  1): matrix_function( 1, **couplings_m1_1),
            ( 1,  1, -1): matrix_function( 1, **couplings_1_m1),
            ( 1,  1,  1): matrix_function( 1, **couplings_1_1),

            (-1, -1, -1): matrix_function(-1, **couplings_m1_m1),
            (-1, -1,  1): matrix_function(-1, **couplings_m1_1),
            (-1,  1, -1): matrix_function(-1, **couplings_1_m1),
            (-1,  1,  1): matrix_function(-1, **couplings_1_1)
        }



def parse_complex(s):
    try: 
        return complex(s)
    except Exception as e:
        print(s)
        raise e


def get_result_amplitude(dtc, a, b, c, d):
    key = f"L(1520)_{{{a}, {b}}}"
    dtc = dtc[key]

    dtc = {
        k: parse_complex(v.replace("im", "j").replace("+ -", "-").replace(" ", ""))
        for k, v in dtc.items()
    }
    return dtc[f"A[{c},{d}]"]


def r_phi(comp):
    return f"{abs(comp)} {float(np.angle(comp))}"

def test_elisabeth():
    import json

    path = "examples/test_data\Parsed_ccp_kinematics_100events.json"
    result_path = "examples/test_data/cpp_100_events_sign2pi_unmodified.json"
    with open(path, "r") as f:
        data = json.load(f)
    with open(result_path, "r") as f:
        result = json.load(f)

    
    for k, dtc in data.items():
        kwargs = {k: v for k, v in dtc["kinematic"].items() if k != "mkpisq" }
        momenta = make_four_vectors_from_dict(**dtc["chain_variables"]["Kpi"], **kwargs)
        amplitude = Amplitude(momenta)
        for hl1520, pi, hlc, hp in product([1, -1], [0], [1, -1], [1, -1]):
            print(hl1520, pi, hlc, hp)
            print(amplitude.value[(hlc, hl1520, hp)][(hp, 0, 0)] / 4**0.5, r_phi(amplitude.value[(hlc, hl1520, hp)][(hp, 0, 0)]/ 4**0.5))
            print(get_result_amplitude(result[k],  hl1520, pi, hlc, hp), r_phi(get_result_amplitude(result[k],  hl1520, pi, hlc, hp)))
            print(amplitude.value[(hlc, hl1520, hp)][(hp, 0, 0)] / 4**0.5 / get_result_amplitude(result[k],  hl1520, pi, hlc, hp))
            # print(get_result_amplitude(result[k], hlc , pi, hl1520, hp))


        
        # res1 = get_result_amplitude(result[k], 1, 0)
        # res2 = get_result_amplitude(result[k], -1, 0)

        exit(0)

if __name__ =="__main__":
    test_elisabeth()