from generate_momenta import make_four_vectors_from_dict
from itertools import product
from decayangle.decay_topology import Topology, TopologyCollection, HelicityAngles
from decayangle.config import config as decayangle_config
import numpy as np 

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import AlignedChain
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
        self.reference_topology = Topology(0, decay_topology=((2, 3), 1))
        self.helicity_angles = self.topology.helicity_angles(momenta=self.momenta)
        self.chain = AlignedChain(
                topology = self.topology,
                reference= self.reference_topology,
                resonances = self.resonances,
                momenta = momenta,
                final_state_qn = self.final_state_qn,
                convention="helicity",
            )
        
        self.combiner = ChainCombiner([self.chain, ])

        matrix_function, matrix_argnames = self.chain.aligned_matrix_function(self.chain.generate_couplings())
        
        couplings_m1_m1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 1,
            "Lc_H_1_0": 0,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0": 0,
            "L_1520_H_-1_0":  1/ (4)**0.5,
        }
        couplings_1_m1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 0,
            "Lc_H_1_0": 1,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0": 0,
            "L_1520_H_-1_0": 1/ (4)**0.5,
        }
        couplings_m1_1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 1,
            "Lc_H_1_0": 0,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0":  1/ (4)**0.5,
            "L_1520_H_-1_0": 0,
        }
        couplings_1_1 = {
            "Lc_H_-3_0": 0,
            "Lc_H_-1_0": 0,
            "Lc_H_1_0": 1,
            "Lc_H_3_0": 0,
            "L_1520_H_1_0":  1/ (4)**0.5,
            "L_1520_H_-1_0": 0,
        }

        arguments_1_1 = {
            L_1520.id: {
                    "couplings": {
                (1, 0) : 1/ (4)**0.5,
                (-1, 0) : 0,
            }},
            Lc.id: {
                    "couplings": {
                (1, 0) : 1,
                (-1, 0) : 0,
                (3, 0) : 0,
                (-3, 0) : 0,
            }
            }
        }
        arguments_m1_1 = {
            L_1520.id: {
                    "couplings": {
                (1, 0) : 1/ (4)**0.5,
                (-1, 0) : 0,
            }},
            Lc.id: {
                    "couplings": {
                (1, 0) : 0,
                (-1, 0) : 1,
                (3, 0) : 0,
                (-3, 0) : 0,
            }
            }
        }

        arguments_1_m1 = {
            L_1520.id: {
                    "couplings": {
                (1, 0) : 0,
                (-1, 0) : 1/ (4)**0.5,
            }
            },
            Lc.id: {
                    "couplings": {
                (1, 0) : 1,
                (-1, 0) : 0,
                (3, 0) : 0,
                (-3, 0) : 0,
            }
            }
        }

        arguments_m1_m1 = {
            L_1520.id: {
                    "couplings": {
                    (1, 0) : 0,
                    (-1, 0) : 1 / (4)**0.5,
                }
            },
            Lc.id:{
                "couplings":  {
                (1, 0) : 0,
                (-1, 0) : 1,
                (3, 0) : 0,
                (-3, 0) : 0,
            }
            }
        }




        aligned_matrix = self.chain.aligned_matrix
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
        # self.value = {
        #     (1, -1, -1): aligned_matrix(1, arguments_m1_m1),
        #     (1, -1, 1): aligned_matrix(1, arguments_m1_1),
        #     (1, 1, -1): aligned_matrix(1, arguments_1_m1),
        #     (1, 1, 1): aligned_matrix(1, arguments_1_1),

        #     (-1, -1, -1): aligned_matrix(-1, arguments_m1_m1),
        #     (-1, -1, 1): aligned_matrix(-1, arguments_m1_1),
        #     (-1, 1, -1): aligned_matrix(-1, arguments_1_m1),
        #     (-1, 1, 1): aligned_matrix(-1, arguments_1_1)
        # }

    @property
    def wig_d(self):
        Lc_decay_angle = self.helicity_angles[((1, 2), 3)]
        L1520_decay_angle = self.helicity_angles[(1,2)]
        Lc_J = 1
        L1520_J = 3

        return {
            (h_lc, h_l1520, h_p): wigner_capital_d(Lc_decay_angle.theta_rf, Lc_decay_angle.phi_rf, 0, Lc_J, h_lc, h_l1520) * wigner_capital_d(L1520_decay_angle.theta_rf, L1520_decay_angle.phi_rf, 0, L1520_J, h_l1520, h_p)
            for h_lc, h_l1520, h_p in product([1, -1], [1, -1], [1, -1])
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
    return dtc[f"A[{c},{d}]"], f"L(1520)_{{{a}, {b}}} A[{c},{d}]"


def r_phi(comp):
    return f"{abs(comp)} {float(np.angle(comp))}"

from decayamplitude.rotation import wigner_capital_d
def test_elisabeth():
    import json

    path = "examples/test_data/Parsed_ccp_kinematics_100events.json"
    result_path = "examples/test_data/cpp_100_events_sign2pi_unmodified.json"
    with open(path, "r") as f:
        data = json.load(f)
    with open(result_path, "r") as f:
        result = json.load(f)

    
    for k, dtc in list(data.items()):
        kwargs = {k: v for k, v in dtc["kinematic"].items() if k != "mkpisq" }
        momenta = make_four_vectors_from_dict(**dtc["chain_variables"]["Kpi"], **kwargs)
        amplitude = Amplitude(momenta)
        for hl1520, pi, hlc, hp in product([1, -1], [0], [1, -1], [1, -1]):
            el, string = get_result_amplitude(result[k],  hl1520, pi, hlc, hp)
            kai = amplitude.value[(hlc, hl1520, hp)][(hp, 0, 0)]
            kai = amplitude.wig_d[(hlc, hl1520, hp)]
            if not np.allclose(el, kai, atol=0.04):
                print(f"{hlc=}, {hl1520=},  {hp=} string = {string}")
                print("Rphi",r_phi(el), r_phi(kai))
                print("value",el, kai)
                print("Ratio", el / kai)
                print()
            else:
                # continue
                print("OK", f"{hl1520=}, {hlc=}, {hp=} string = {string}")

           
            # print(get_result_amplitude(result[k], hlc , pi, hl1520, hp))


        
        # res1 = get_result_amplitude(result[k], 1, 0)
        # res2 = get_result_amplitude(result[k], -1, 0)

        exit(0)


if __name__ =="__main__":
    test_elisabeth()