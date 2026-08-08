from typing import Callable, Literal
from itertools import product
from functools import cached_property
import warnings as warnings
import jax

from decayangle.decay_topology import Topology, Node, HelicityAngles
from decayamplitude.particle import Particle
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, wigner_capital_d, Angular, convert_angular
from decayamplitude.resonance import ResonanceDict
from decayamplitude.backend import numpy as np

from decayamplitude.utils import _create_function, sanitize, _no_momenta_guard, _warmup
from decayamplitude.kinematics_helpers import mass_from_node


def _stack_sum(terms):
    """Sum a list of same-shape JAX-array terms via stack + jnp.sum instead of
    a Python-unrolled chain of `+` operators.

    Mathematically equivalent up to floating-point summation order, but for
    resonances/chains with several terms this collapses what would otherwise
    be O(len(terms)) separate traced add ops into a single reduction, which is
    what actually drives XLA compile time down (see benchmarks/benchmark_compile_time.py).
    """
    if len(terms) == 1:
        return terms[0]
    return np.sum(np.stack(terms, axis=0), axis=0)


def _per_particle_alignment_factors(final_state_qn, final_state_keys, wigner_rotation):
    """Precompute, for each final-state particle, conj(wigner_capital_d(...))
    for every (lambda_, lambda) pair of that particle's OWN helicity states.

    aligned_matrix's alignment sum runs over all (lambdas, lambdas_) helicity
    TUPLE pairs, but the Wigner-D factor for a given final_state_key only
    depends on that one particle's own two helicity values, not on the other
    particles'. Naively recomputing wigner_capital_d inside the
    O(len(helicities)^2) nested sum therefore does len(helicities)^2 * N
    calls for N final-state particles, almost all of it identical repeated
    work; looking values up in this small per-particle table instead needs
    only sum((2j+1)^2) distinct calls, independent of how many OTHER
    final-state particles there are.
    """
    return {
        key: {
            (m1, m2): np.conj(wigner_capital_d(*wigner_rotation[key], final_state_qn[key].angular.value2, m1, m2))
            for m1 in final_state_qn[key].angular.projections(return_int=True)
            for m2 in final_state_qn[key].angular.projections(return_int=True)
        }
        for key in final_state_keys
    }


def _alignment_rotation_matrix(helicities, final_state_keys, alignment_factors):
    """Build the (n_helicities, n_helicities) alignment matrix
    R[i, j] = prod_key alignment_factors[key][(helicities[j][key], helicities[i][key])]

    so aligned_matrix's combination can be a single contraction (einsum)
    against the per-lambda_ amplitude vector instead of a Python double loop
    over all (lambdas, lambdas_) pairs. For spin-full final states
    len(helicities) grows fast (product of each particle's 2j+1), and the
    naive nested loop creates O(len(helicities)^2) small, disconnected
    multiply/reduce ops -- empirically this, not the per-chain amplitude
    recursion, is what makes XLA compile time blow up (see
    benchmarks/benchmark_compile_time.py): going from 1 to 2 topologies only
    grew the jaxpr by ~2.7x but grew compile time by ~14x, and compile time
    barely changed when the number of resonances feeding the same
    2-topology alignment was reduced from 4 to 1. Collapsing the alignment
    into one matrix and one einsum turns those O(n^2) disconnected ops into
    O(n^2) *connected* ones that XLA can fuse as a single kernel.
    """
    rows = []
    for lambdas in helicities:
        entries = []
        for lambdas_ in helicities:
            factor = None
            for key in final_state_keys:
                term = alignment_factors[key][(lambdas_[key], lambdas[key])]
                factor = term if factor is None else factor * term
            entries.append(factor)
        rows.append(np.stack(entries, axis=0))
    return np.stack(rows, axis=0)


def _cached(cache, key, compute):
    """Look up `key` in `cache` (a dict, or None to disable caching), computing
    and storing it via `compute()` on a miss. Used for values that do not
    depend on h0 -- e.g. inter-topology Wigner-rotation angles -- but would
    otherwise be recomputed on every h0 iteration of the outer h0-loop in
    ChainCombiner.unpolarized_amplitude."""
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    return cache[key]


class MomentaCache:
    """Object-level cache for momenta-only quantities (helicity angles,
    per-node masses, inter-topology alignment rotation data). Shared by
    reference across an entire chain's node tree, and (once combined) across
    every chain in a ChainCombiner -- see DecayChain.momenta_cache and
    ChainCombiner.momenta_cache -- so populating/enabling it once makes the
    data visible everywhere without threading it through every function call.

    Deliberately separate from the `cache` parameter still threaded through
    matrix/chain_function/aligned_matrix/amplitude: that one memoizes
    fit-parameter-dependent terms (h0_independent_terms) and MUST stay
    freshly created per call, or stale coupling values could leak across fit
    iterations with different parameters. See DecayChain.enable_static_momenta
    and _momenta_cached for how the two coexist.
    """
    def __init__(self):
        self.data = {}
        self.enabled = False

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


def _momenta_cached(momenta_cache, cache, key, compute):
    """Look up a momenta-only value (helicity angles, masses, alignment
    rotation data): prefer the object-level `momenta_cache` if enabled (valid
    across many calls -- e.g. static_momenta mode), otherwise fall back to
    the per-call `cache` parameter (valid only within the current call --
    the existing, unchanged default-mode dedup from earlier optimization
    cycles).
    """
    if momenta_cache.enabled:
        return _cached(momenta_cache, key, compute)
    return _cached(cache, key, compute)


def _node_mass_cache_entries(nodes, momenta):
    """Build {(id(node), "masses"): (mass, d1_mass, d2_mass)} entries for every
    non-final-state node in `nodes`, matching the cache key DecayChainNode.amplitude
    looks up. Shared by DecayChain/MultiChain._static_cache to eagerly
    precompute masses for a fixed momenta set, so mass_from_node is never
    called at trace time in static-momenta mode."""
    return {
        (id(node), "masses"): (
            mass_from_node(node.node, momenta),
            mass_from_node(node.daughters[0].node, momenta),
            mass_from_node(node.daughters[1].node, momenta),
        )
        for node in nodes
        if not node.final_state
    }


def _node_wigner_d_cache_entries(nodes, helicity_angles):
    """Build {(id(node), "wigner_d", h0, m_diff): conj(wigner_capital_d(...))}
    entries for every non-final-state node and every (h0, m_diff) pair its own
    amplitude recursion can call wigner_capital_d with (see the tail of
    DecayChainNode.amplitude). h0, m_diff and J2 are always plain Python ints
    at trace time regardless of mode (driven by Python-level loops /
    static_argnums, never JAX tracers) -- the angles are the only piece that
    need momenta to be resolved, so once helicity_angles is a concrete value
    this whole per-node Wigner-D factor can be precomputed eagerly instead of
    relying on XLA to fold it during compilation. Shared by
    DecayChain/MultiChain._static_cache."""
    entries = {}
    for node in nodes:
        if node.final_state:
            continue
        angles = helicity_angles[node.decay_tuple]
        phi, theta = angles.phi_rf, angles.theta_rf
        psi = 0 if node.convention == "helicity" else -phi
        J2 = node.quantum_numbers.angular.value2
        d1, d2 = node.daughters
        m_diffs = {
            h1 - h2
            for h1 in d1.quantum_numbers.projections(return_int=True)
            for h2 in d2.quantum_numbers.projections(return_int=True)
        }
        for h0 in node.quantum_numbers.projections(return_int=True):
            for m_diff in m_diffs:
                entries[(id(node), "wigner_d", h0, m_diff)] = np.conj(wigner_capital_d(phi, theta, psi, J2, h0, m_diff))
    return entries

class DecayChainNode:
    """
    Class to represent a node in the decay chain. This utilizes the Node class from decayangle. 
    A Node has a resonance, and a topology, to make senese of its position in the decay chain. The node value only has a meaning in the context of the topology.
    """


    def __init__(self, node: Node, resonances: dict[tuple, Resonance] | ResonanceDict, final_state_qn: dict[int, QN | Particle], topology: Topology, convention: Literal["helicity", "minus_phi"] = "helicity", momenta_cache: "MomentaCache | None" = None) -> None:
        """
        Initializes a DecayChainNode object. The object will contain a resonance and a topology.

        Parameters:
        node: Node
            The node of the decay chain. This is a node of the decay topology as defined in `decayangle`
        resonances: dict[tuple, Resonance]
            A dictionary with the resonances of the decay chain. The keys are the tuples of the nodes, the values are the resonances
        final_state_qn: dict[tuple, QN]
            A dictionary with the quantum numbers of the final state particles. The keys are the tuples of the nodes, the values are the quantum numbers
        topology: Topology
            The topology of the decay chain. This is a topology as defined in `decayangle`
        convention: str
            The convention of the decay chain. This is either "helicity" or "minus_phi". The default is "helicity"
        momenta_cache: MomentaCache | None
            Object-level cache for momenta-only quantities (see MomentaCache).
            If not given, this node creates its own (disabled) one. Threaded
            into daughter construction below so the whole recursive tree
            shares a single instance from the moment it's built -- see
            DecayChain.momenta_cache for how that instance later gets
            populated/enabled.
        """
        self.momenta_cache = momenta_cache if momenta_cache is not None else MomentaCache()
        # this check needs to happen first to avoid errors

        if node.value not in topology.nodes:
            # someone may have initialized the root with a tuple instead of 0
            if node.tuple == topology.root.tuple:
                self.node = topology.root
            else:
                raise ValueError(f"Node {node} not in topology {topology}")
        else:
            self.node = topology.nodes[node.value]
        
        self.tuple = self.node.tuple
        self.__is_root = self.node == topology.root
            
        if not isinstance(resonances, ResonanceDict):
            # if the resonances are not a ResonanceDict, we convert them to one
            resonances = ResonanceDict(resonances)
        self.resonance: Resonance | None = resonances.get(self.tuple, resonances.get(self.node.value, None))
        if self.resonance is None and not self.final_state:
            raise ValueError(f"Resonance for {self.node.tuple} not found. Every internal node must have a resonance to describe its behaviour!")

        self.resonances = resonances
        self.topology = topology
        self.final_state_qn = final_state_qn
        self.convention = convention
            
        self.daughters = [
                    DecayChainNode(daughter, resonances, self.final_state_qn, topology, convention=self.convention, momenta_cache=self.momenta_cache)
                    for daughter in self.node.daughters
            ]
        
        if not self.final_state:
            if self.resonance is None:
                raise ValueError(f"Resonance for {self.tuple} not found. Every internal node must have a resonance to describe its behaviour!")
            # set the daughters of the resonance
            self.quantum_numbers = self.resonance.quantum_numbers
            self.resonance.daughters = [daughter for daughter in self.daughters]
        else:
            self.quantum_numbers = self.final_state_qn[self.tuple]

    @property
    def final_state(self):
        """
        Returns:
        bool
            True if the node is a final state particle, False otherwise
        """
        return self.node.final_state
    
    @property
    def name(self) -> str:
        """
        Returns:
        str
            The name of the node
        """
        if self.resonance is not None:
            return self.resonance.name
        if isinstance(self.quantum_numbers, Particle):
            if self.quantum_numbers.name is not None:
                return f"{sanitize(self.quantum_numbers.name)}"
            return f"particle_type_{self.quantum_numbers.type_id}"
        return f"particle_{self.node.value}"
    
    @property
    def sanitized_name(self) -> str:
        """
        Returns:
        str
            The sanitized name of the node
        """
        if self.resonance is not None:
            return self.resonance.sanitized_name
        return sanitize(self.name)
    
    @property
    def is_root(self):
        """
        Returns:
        bool
            True if the node is the root of the decay chain, False otherwise
        """
        return self.__is_root

    @property
    def quantum_numbers(self) -> QN:
        """
        Returns:
        QN
            The quantum numbers of the node
        """
        return self.__qn 
    
    @property
    def decay_tuple(self) -> tuple:
        """
        Returns:
        tuple
            The decay tuple of the node. This is the tuple of the nodes daughters values.
        """
        return tuple([daughter.node.value for daughter in self.daughters])
    
    @quantum_numbers.setter
    def quantum_numbers(self, qn: QN):
        """
        Sets the quantum numbers of the node. This is used to set the quantum numbers of the resonance.
        """
        self.__qn = qn

    def __helicity_angles(self, angles: HelicityAngles) -> tuple:
        """
        Returns the helicity angles of the node. This is used to calculate the amplitude of the decay chain.

        Parameters:
            angles: HelicityAngles
                The helicity angles of the node. This is used to calculate the amplitude of the decay chain. Helicity angles are defined in the `decayangle` library.
        Returns:
            tuple
                The helicity angles of the node. This is used to calculate the amplitude of the decay chain.
        """
        if self.convention == "helicity":
            return (angles.phi_rf, angles.theta_rf, 0)
        if self.convention == "minus_phi":
            return (angles.phi_rf, angles.theta_rf, -angles.phi_rf)
        raise ValueError(f"Convention {self.convention} not known")

    @convert_angular
    def amplitude(self, h0: Angular | int, lambdas: dict, arguments: dict, momenta: dict, helicity_angles: dict, cache: dict | None = None):
        """
        The amplitude of a single node given the helicity of the decaying particle.
        Recursively computes daughter amplitudes.

        parameters:
        h0: int
            Helicity of the decaying particle
        lambdas: dict
            Helicities of the final-state particles
        arguments: dict
            Couplings and resonance parameters
        momenta: dict
            Per-event four-momenta, keyed by particle index
        helicity_angles: dict
            Per-event helicity angles for every internal node, keyed by decay_tuple.
            Pre-computed once at chain_function level from momenta.
        cache: dict | None
            Optional cache shared across calls that only differ in h0 (e.g. the
            h0-loop in ChainCombiner.unpolarized_amplitude). Everything at this
            node except the final Wigner-D rotation factor -- the daughter
            sub-amplitudes and this node's own resonance coupling -- is
            independent of h0, so it is computed once per (node, lambdas) and
            reused across h0 values instead of being recomputed from scratch on
            every h0 iteration. Pass None (the default) to disable and get the
            original per-call behaviour.
        """
        if self.final_state:
            yield 1.
        else:
            angles = helicity_angles[self.decay_tuple]
            phi, theta = angles.phi_rf, angles.theta_rf
            psi = 0 if self.convention == "helicity" else -phi
            J2 = self.quantum_numbers.angular.value2

            cache_key = (id(self), tuple(sorted(lambdas.items()))) if cache is not None else None
            if cache_key is not None and cache_key in cache:
                h0_independent_terms = cache[cache_key]
            else:
                d1, d2 = self.daughters
                d1_helicities = [lambdas[d1.tuple]] if d1.final_state else d1.quantum_numbers.projections(return_int=True)
                d2_helicities = [lambdas[d2.tuple]] if d2.final_state else d2.quantum_numbers.projections(return_int=True)

                # Masses depend only on momenta (never on arguments/h0), so they
                # get their own cache slot separate from h0_independent_terms:
                # self.momenta_cache, if enabled (static_momenta mode), lets
                # this be prepopulated so mass_from_node is never called at
                # trace time at all -- see _momenta_cached. h0_independent_terms
                # itself must still always be recomputed fresh whenever
                # arguments change, hence the separate `cache` parameter below.
                mass, d1_mass, d2_mass = _momenta_cached(
                    self.momenta_cache, cache, (id(self), "masses"),
                    lambda: (mass_from_node(self.node, momenta), mass_from_node(d1.node, momenta), mass_from_node(d2.node, momenta)),
                )

                h0_independent_terms = []
                for h1 in d1_helicities:
                    for h2 in d2_helicities:
                        for A_1 in d1.amplitude(h1, lambdas, arguments, momenta, helicity_angles, cache=cache):
                            for A_2 in d2.amplitude(h2, lambdas, arguments, momenta, helicity_angles, cache=cache):
                                coupling = self.resonance.amplitude(h1, h2, arguments, mass, d1_mass, d2_mass)
                                h0_independent_terms.append((h1 - h2, A_1 * A_2 * coupling * (J2 + 1)**0.5))
                if cache_key is not None:
                    cache[cache_key] = h0_independent_terms

            for m_diff, term in h0_independent_terms:
                d_factor = _momenta_cached(
                    self.momenta_cache, cache, (id(self), "wigner_d", h0, m_diff),
                    lambda: np.conj(wigner_capital_d(phi, theta, psi, J2, h0, m_diff)),
                )
                yield term * d_factor

class DecayChain:
    """
    Class to represent a decay chain. This is a topology in connection with a set of resonances. One resonance for each internal node in the topology.
    """

    def __init__(self, topology: Topology, resonances: dict[tuple, Resonance] | ResonanceDict, final_state_qn: dict[int, QN | Particle], convention: Literal["helicity", "minus_phi"] = "helicity") -> None:
        self.topology = topology
        if not isinstance(resonances, ResonanceDict):
            resonances = ResonanceDict(resonances)
        self.resonances = resonances
        self.final_state_qn = final_state_qn
        self.convention = convention

        self.root_resonance = self.resonances.get(self.topology.root.value)
        if self.root_resonance is None:
            self.root_resonance = self.resonances.get(self.topology.root.tuple)
        if self.root_resonance is None:
            raise ValueError(f"No root resonance found! The root resonance should be the decaying particle. The lineshape is irrelelevant for decay processes, but the quantum numbers are crucial! Define a root resonance under the key {self.topology.root.value} or {self.topology.root.tuple}. Available resonances: {list(self.resonances.keys())}")

        # we need a sorted version of the particle keys to map matrix elements to the correct particle helicities later
        self.final_state_keys = sorted(final_state_qn.keys())
        helicities = Angular.generate_helicities(*[final_state_qn[key] for key in self.final_state_keys])
        self.helicities = [
            {key: helicity[i] for i, key in enumerate(self.final_state_keys)}
            for helicity in helicities
        ]
        self.helicity_tuples = helicities
        self.resonance_list = list(resonances.values())
        # Created once, threaded into the node tree below (nodes/root), so
        # every node in this chain shares one instance from construction --
        # see the `momenta_cache` property for how it later gets
        # populated/enabled without needing to touch the nodes again.
        self._momenta_cache = MomentaCache()

    @cached_property
    def nodes(self):
        # The node tree only depends on (topology, resonances, final_state_qn,
        # convention), all fixed at construction time, so it is safe -- and,
        # since amplitude() rebuilds it from scratch on every access otherwise,
        # important for trace time -- to build it once and reuse it.
        return list(
            DecayChainNode(node, self.resonances, self.final_state_qn, self.topology, self.convention, momenta_cache=self._momenta_cache)
            for node in self.topology.nodes.values()
        )

    @property
    def final_state_nodes(self) -> list[DecayChainNode]:
        return [node for node in self.nodes if node.final_state]

    @property
    def momenta_cache(self) -> MomentaCache:
        return self._momenta_cache

    @momenta_cache.setter
    def momenta_cache(self, value: MomentaCache):
        # Mutate the existing shared object's contents rather than replacing
        # the reference: every node in self.nodes/self.root already holds a
        # reference to self._momenta_cache (threaded in at construction), so
        # this update is instantly visible throughout the tree with no
        # separate per-node cascade needed.
        self._momenta_cache.data = value.data
        self._momenta_cache.enabled = value.enabled

    @cached_property
    def root(self):
        return DecayChainNode(self.topology.root, self.resonances, self.final_state_qn, self.topology, self.convention, momenta_cache=self._momenta_cache)

    def _amplitude_for_lambdas(self, h0, lambdas: dict, arguments: dict, momenta: dict, helicity_angles: dict, cache: dict | None = None):
        """Amplitude for one set of final-state helicities, given precomputed helicity_angles."""
        amplitudes = list(self.root.amplitude(h0, lambdas, arguments, momenta, helicity_angles, cache=cache))
        prefactor = 1/(self.root.resonance.quantum_numbers.angular.value2 + 1)**0.5
        return prefactor * _stack_sum(amplitudes)

    def _matrix_for_angles(self, h0, arguments: dict, momenta: dict, helicity_angles: dict, cache: dict | None = None) -> dict:
        """Full helicity matrix given precomputed helicity_angles.

        Used internally by MultiChain to share one helicity_angles computation
        across all sibling resonance hypotheses for the same topology, instead
        of every sibling recomputing it from momenta independently.
        """
        return {
            tuple([lambdas[key] for key in self.final_state_keys]): self._amplitude_for_lambdas(h0, lambdas, arguments, momenta, helicity_angles, cache=cache)
            for lambdas in self.helicities
        }

    def _static_cache(self, momenta) -> dict:
        """Eagerly (outside of any jax.jit trace) compute every momenta-only
        cache entry this chain's amplitude computation can use: helicity_angles
        for its topology, masses for every non-final-state node, and each
        such node's own per-(h0, m_diff) Wigner-D factors. Used by the
        `static_momenta` mode of the creator functions (unpolarized_amplitude
        etc.) so those quantities are baked in as concrete values instead of
        being recomputed (or, for the Wigner-D factors, XLA-folded) from a
        traced momenta argument on every call -- see _cached,
        _node_mass_cache_entries and _node_wigner_d_cache_entries for the
        matching lookup keys.
        """
        helicity_angles = self.topology.helicity_angles(momenta=momenta, convention=self.convention)
        return {
            (id(self.topology), "helicity_angles"): helicity_angles,
            **_node_mass_cache_entries(self.nodes, momenta),
            **_node_wigner_d_cache_entries(self.nodes, helicity_angles),
        }

    def enable_static_momenta(self, momenta):
        """Precompute and enable this chain's object-level momenta_cache for
        a fixed momenta set (see MomentaCache and the momenta_cache property).
        Used by the `static_momenta` mode of unpolarized_amplitude.

        NOTE: mutates shared state. Calling this again on the same chain
        overwrites the previous momenta's cached values -- safe only because
        creator functions warm up (fully trace + compile) before returning,
        so an earlier build's compiled function no longer reads this cache
        by the time a later build starts.
        """
        cache = MomentaCache()
        cache.data = self._static_cache(momenta)
        cache.enabled = True
        self.momenta_cache = cache

    @property
    def chain_function(self):
        """
        Returns a function f(h0, lambdas, arguments, momenta, cache=None) -> complex amplitude
        for a single set of helicities. See DecayChainNode.amplitude for `cache`.
        """
        def f(h0, lambdas: dict, arguments: dict, momenta: dict, cache: dict | None = None):
            helicity_angles = _momenta_cached(
                self.momenta_cache, cache, (id(self.topology), "helicity_angles"),
                lambda: self.topology.helicity_angles(momenta=momenta, convention=self.convention),
            )
            return self._amplitude_for_lambdas(h0, lambdas, arguments, momenta, helicity_angles, cache=cache)
        return f

    @property
    def matrix(self):
        """
        Returns a function f(h0, arguments, momenta, cache=None) -> dict mapping
        final-state helicity tuples to their amplitude, covering all helicity
        combinations. See DecayChainNode.amplitude for `cache`.
        """
        def matrix(h0, arguments: dict, momenta: dict, cache: dict | None = None):
            helicity_angles = _momenta_cached(
                self.momenta_cache, cache, (id(self.topology), "helicity_angles"),
                lambda: self.topology.helicity_angles(momenta=momenta, convention=self.convention),
            )
            return self._matrix_for_angles(h0, arguments, momenta, helicity_angles, cache=cache)
        return matrix
    
    def generate_couplings(self):
        """
        Returns all LS couplings for the decay chain
        """
        return {
            node.resonance.id: node.resonance.generate_couplings(node.resonance.preserve_partity)
            for node in self.nodes
            if not node.final_state
        }
    
    @property
    def resonance_params(self):
        resonances = [resonance for resonance in self.resonance_list]
        resonance_parameter_names = [name for resonance in resonances for name in resonance.parameter_names]

        if len(set(resonance_parameter_names)) != len(resonance_parameter_names):
            from collections import Counter
            c = Counter(resonance_parameter_names)
            # raise ValueError(f"Parameter names are not unique: {', '.join([name for name, count in c.items() if count > 1])}")
        return list(set(resonance_parameter_names))

    def unpolarized_amplitude(self, ls_couplings: dict, complex_couplings=True, static_momenta=None) -> tuple[Callable, list[str]]:
        """
        Returns a per-event function f(momenta, *coupling_args) -> scalar intensity.
        Vectorize with jax.vmap over the momenta argument.

        If `static_momenta` is given, momenta is treated as a fixed dataset:
        every momenta-only computation (helicity angles, masses) is
        precomputed once, eagerly, right now, instead of being retraced from
        a momenta argument on every call. The returned func(*params) does not
        take momenta at all (calling it with momenta raises TypeError) and is
        pre-compiled (warmed up) before being returned. See
        ChainCombiner.unpolarized_amplitude for the full rationale.
        """
        if static_momenta is not None:
            self.enable_static_momenta(static_momenta)

        def f(arguments: dict):
            # Momenta-only lookups (helicity_angles, masses) are served by
            # self.momenta_cache when static_momenta is enabled, or
            # recomputed per call otherwise. `cache` is only for
            # h0_independent_terms, which must stay scoped to this call since
            # it bakes in `arguments`.
            momenta = static_momenta if static_momenta is not None else arguments.pop("momenta")
            cache: dict = {}
            return sum(
                abs(v)**2
                for h0 in self.root_resonance.quantum_numbers.angular.projections()
                for v in self.matrix(h0, arguments, momenta, cache=cache).values()
            )

        if static_momenta is not None:
            func, argnames = _create_function(self.resonance_params, ls_couplings, f, complex_couplings=complex_couplings)
            # h0 is only ever a Python-level loop variable here, never a
            # function argument, so no static_argnums are needed for jit-safety.
            func = jax.jit(func)
            func = _no_momenta_guard(func)
            _warmup(func, argnames)
            return func, argnames

        return _create_function(["momenta"] + self.resonance_params, ls_couplings, f, complex_couplings=complex_couplings)

class AlignedChain(DecayChain):
    """
    The aligned version of the decay chain. This is used to calculate the aligned amplitude, which is the amplitude in the final state helicity frame as defined by a reference topology or reference chain.
    """

    def __init__(self, topology: Topology, resonances: dict[tuple, Resonance], final_state_qn: dict[int, QN | Particle], reference: Topology | DecayChain, convention: Literal["helicity", "minus_phi"] = "helicity") -> None:
        if isinstance(reference, DecayChain):
            if reference.convention != convention:
                raise ValueError(f"Reference and chain must have the same convention. Found reference: {reference.convention} and self: {convention}!")
        self.reference: Topology = reference if isinstance(reference, Topology) else reference.topology
        super().__init__(topology, resonances, final_state_qn, convention)

    def to_tuple(self, lambdas: dict):
        """Maps a {particle_key: helicity} dict to the sorted-key tuple used as a matrix key."""
        return tuple([lambdas[key] for key in self.final_state_keys])

    def _static_cache(self, momenta) -> dict:
        """Extends the parent's _static_cache with the alignment-specific
        entries aligned_matrix looks up: relative_wigner_angles, the
        per-particle alignment factor table, and the full alignment rotation
        matrix -- mirrors aligned_matrix's own computation exactly.
        """
        cache = super()._static_cache(momenta)
        wigner_rotation = self.reference.relative_wigner_angles(self.topology, momenta, convention=self.convention)
        cache[(id(self), "wigner_rotation")] = wigner_rotation
        alignment_factors = _per_particle_alignment_factors(self.final_state_qn, self.final_state_keys, wigner_rotation)
        cache[(id(self), "alignment_factors")] = alignment_factors
        cache[(id(self), "rotation_matrix")] = _alignment_rotation_matrix(self.helicities, self.final_state_keys, alignment_factors)
        return cache

    @property
    def aligned_matrix(self):
        """
        Returns a function f(h0, arguments, momenta) -> dict mapping final-state
        helicity tuples to their amplitude, rotated into the reference frame via
        Wigner D-matrices computed fresh from momenta on every call.
        """
        m = self.matrix
        def f(h0, arguments: dict, momenta: dict, cache: dict | None = None):
            matrix = m(h0, arguments, momenta, cache=cache)
            # relative_wigner_angles depends only on (topology, momenta), not on
            # h0 -- cache it too so it isn't recomputed on every h0 iteration.
            wigner_rotation = _momenta_cached(
                self.momenta_cache, cache, (id(self), "wigner_rotation"),
                lambda: self.reference.relative_wigner_angles(self.topology, momenta, convention=self.convention),
            )
            # Per-particle Wigner-D lookup table instead of recomputing
            # wigner_capital_d fresh for every (lambdas, lambdas_) tuple pair
            # below -- see _per_particle_alignment_factors.
            alignment_factors = _momenta_cached(
                self.momenta_cache, cache, (id(self), "alignment_factors"),
                lambda: _per_particle_alignment_factors(self.final_state_qn, self.final_state_keys, wigner_rotation),
            )
            # The full (n_helicities, n_helicities) alignment matrix, h0-independent
            # like alignment_factors -- see _alignment_rotation_matrix for why this
            # is built once as a matrix and applied via one contraction, rather
            # than combined per (lambdas, lambdas_) pair with a Python loop.
            rotation_matrix = _momenta_cached(
                self.momenta_cache, cache, (id(self), "rotation_matrix"),
                lambda: _alignment_rotation_matrix(self.helicities, self.final_state_keys, alignment_factors),
            )
            matrix_vec = np.stack([matrix[self.to_tuple(lambdas_)] for lambdas_ in self.helicities], axis=0)
            result_vec = np.einsum("ij...,j...->i...", rotation_matrix, matrix_vec)
            return {
                self.to_tuple(lambdas): result_vec[i]
                for i, lambdas in enumerate(self.helicities)
            }
        return f


class MultiChain(DecayChain):
    @classmethod
    def create_chains(cls, resonances: dict[tuple, tuple[Resonance]] | ResonanceDict, topology: Topology) -> list[dict[tuple, Resonance]]:
        """
        Creates all possible chains from a dictionary with lists of reonances for each isobar
        """
        if not isinstance(resonances, ResonanceDict):
            # if the resonances are not a ResonanceDict, we convert them to one
            resonances = ResonanceDict(resonances)

        if topology is None:
            raise ValueError("Topology must be provided to create chains from resonances")
        # with a given topology we can restrict the resonances to the nodes in the topology
        # this is usefull, if we only have one global dict of resonances
        filtered_resonances = resonances.filter_by_topology(topology)
        chains =list( product(*[filtered_resonances[key] for key in filtered_resonances.keys() ]) )
        return [
            ResonanceDict({
                key: chain[i].copy()
                for i, key in enumerate(filtered_resonances.keys())
            })
            for chain in chains
        ]

    @classmethod
    def from_chains(cls, chains: list[DecayChain]) -> "MultiChain":
        if any(chain.topology != chains[0].topology for chain in chains):
            raise ValueError("All chains must have the same topology")
        return cls(chains[0].topology, chains[0].final_state_qn, chains=chains)

    def __init__(self, topology: Topology, final_state_qn: dict[int, QN | Particle], resonances: dict[tuple, tuple[Resonance]] | None = None, chains: list[DecayChain] | None = None, convention: Literal["minus_phi", "helicity"] | None = "helicity") -> None:
        """
        Initializes a MultiChain object. The object will contain a list of DecayChain objects.

        Parameters:
        topology: Topology
            The topology of the decay chain
        momenta: dict
            The momenta of the decay chain
        final_state_qn: dict[tuple, QN]
            The quantum numbers of the final state particles
        resonances: dict[tuple, tuple[Resonance]] | ResonanceDict
            A dictionary with a list of resonances for each isobar
        chains: list[DecayChain]
            A list of DecayChain objects
        convention: str
            The convention of the decay chain. Default is "helicity"
        """

        if chains is not None:
            self.chains = chains
            # I will stick with a default value for the convention. It will have no effect if chains are provided
            # if convention is not None:
            #     raise ValueError("Convention must not be set if chains are provided directly")
            if not all(chain.convention == chains[0].convention for chain in chains):
                raise ValueError("All chains must have the same convention")
            self.convention = chains[0].convention
        elif resonances is not None:
            if not isinstance(resonances, ResonanceDict):
                print(resonances.keys())
                # if the resonances are not a ResonanceDict, we convert them to one
                resonances = ResonanceDict(resonances)
            resonant_nodes = [node for node in topology.nodes.values() if not node.final_state]
            if any(node.value not in resonances and node.tuple not in resonances for node in resonant_nodes):
                warnings.warn(f"Not all nodes have a resonance assigned: {resonances.keys()}, {list(map(lambda x: x.value,resonant_nodes))}")
            self.chains = [
                DecayChain(topology, chain_definition, final_state_qn, convention)
                for chain_definition in type(self).create_chains(resonances, topology)
            ]
            def chain_filter(chain: DecayChain) -> bool:
                try:
                    chain.generate_couplings()
                except ValueError as e:
                    warnings.warn(f"Chain {chain} is not valid: {e}")
                    return False
                return True
            self.chains = [chain for chain in self.chains if chain_filter(chain)]
            if not self.chains:
                raise ValueError("There are no valid chains in the provided resonances! Check the resonances and the topology! Or check the quantum numbers!")
            self.convention = convention
        else:
            raise ValueError("Either resonances or chains must be provided")
        if chains is not None and resonances is not None:
            raise ValueError("Either resonances or chains must be provided")
        self.final_state_qn = final_state_qn

    def _static_cache(self, momenta) -> dict:
        """Like DecayChain._static_cache, but masses and per-node Wigner-D
        factors are precomputed for every non-final-state node across EVERY
        sibling chain in self.chains (each resonance hypothesis has its own
        node tree), not just self.chains[0]'s. helicity_angles is still
        computed once, since all siblings share the same topology.
        """
        helicity_angles = self.topology.helicity_angles(momenta=momenta, convention=self.convention)
        cache = {(id(self.topology), "helicity_angles"): helicity_angles}
        for chain in self.chains:
            cache.update(_node_mass_cache_entries(chain.nodes, momenta))
            cache.update(_node_wigner_d_cache_entries(chain.nodes, helicity_angles))
        return cache

    @property
    def momenta_cache(self) -> MomentaCache:
        # MultiChain has no node tree of its own (like topology/nodes/root,
        # it delegates); each sibling in self.chains owns and shares its own
        # cache with its own node tree (see DecayChain.momenta_cache), so
        # "the" MultiChain-level cache is whichever the first sibling has.
        return self.chains[0].momenta_cache

    @momenta_cache.setter
    def momenta_cache(self, value: MomentaCache):
        # Siblings were built independently (each with its own momenta_cache
        # instance), so unifying them needs an explicit cascade here -- each
        # sibling's own setter then mutates its own shared instance in place,
        # reaching that sibling's whole node tree with no further cascade.
        for chain in self.chains:
            chain.momenta_cache = value

    @property
    def chain_function(self) -> Callable:
        """Returns f(h0, lambdas, arguments, momenta, cache=None), summed over all constituent chains."""
        def f(h0, lambdas: dict, arguments: dict, momenta: dict, cache: dict | None = None):
            # All chains share the same topology (enforced in __init__), so
            # helicity_angles(momenta) is identical for every one of them --
            # compute it once instead of once per resonance hypothesis, and
            # (via _cached) once per call rather than once per h0 too.
            helicity_angles = _momenta_cached(
                self.momenta_cache, cache, (id(self.topology), "helicity_angles"),
                lambda: self.topology.helicity_angles(momenta=momenta, convention=self.convention),
            )
            return _stack_sum([
                chain._amplitude_for_lambdas(h0, lambdas, arguments, momenta, helicity_angles, cache=cache)
                for chain in self.chains
            ])
        return f
    
    @property
    def resonance_list(self) -> list[Resonance]:
        return [
            resonance for chain in self.chains
            for resonance in chain.resonance_list
        ]
    
    @property
    def final_state_keys(self) -> list[tuple | int]:
        return self.chains[0].final_state_keys
    
    @property
    def topology(self):
        return self.chains[0].topology

    @property
    def matrix(self):
        def dict_sum(*dtcs):
            if len(dtcs) == 1:
                return dtcs[0]
            if len(dtcs) == 0:
                raise ValueError("No dicts to sum")
            if any(set(dtcs[0].keys()) != set(dtc.keys()) for dtc in dtcs):
                raise ValueError("Keys of the dicts do not match")
            return {
                key: _stack_sum([dtc[key] for dtc in dtcs])
                for key in dtcs[0].keys()
            }

        def matrix(h0, arguments: dict, momenta: dict, cache: dict | None = None):
            # All chains share the same topology (enforced in __init__), so
            # helicity_angles(momenta) is identical for every one of them --
            # compute it once instead of once per resonance hypothesis, and
            # (via _cached) once per call rather than once per h0 too.
            helicity_angles = _momenta_cached(
                self.momenta_cache, cache, (id(self.topology), "helicity_angles"),
                lambda: self.topology.helicity_angles(momenta=momenta, convention=self.convention),
            )
            return dict_sum(
                *[chain._matrix_for_angles(h0, arguments, momenta, helicity_angles, cache=cache)
                for chain in self.chains]
            )
        return matrix
    
    @property
    def root(self):
        return self.chains[0].root
    
    @property
    def nodes(self):
        return self.chains[0].nodes
    
    @property
    def helicities(self):
        return self.chains[0].helicities
    
    @property
    def helicity_tuples(self):
        return self.chains[0].helicity_tuples

    def generate_couplings(self):
        """
        Returns all LS couplings for the decay chain
        """
        coupling_dict = {}
        for chain in self.chains:
            coupling_dict.update(chain.generate_couplings())
        return coupling_dict
    
    @property
    def root_resonance(self) -> Resonance | None:
        if all(chain.root_resonance.quantum_numbers == self.chains[0].root_resonance.quantum_numbers for chain in self.chains):
            return self.chains[0].root_resonance
        return None
    
class AlignedMultiChain(MultiChain):
    @classmethod
    def from_chains(cls, chains: list[DecayChain], reference: Topology | DecayChain) -> "AlignedMultiChain":
        return cls(chains[0].topology, chains[0].final_state_qn, reference, chains=chains)

    @classmethod
    def from_multichain(cls, multichain: MultiChain, reference: Topology | DecayChain) -> "AlignedMultiChain":
        return cls.from_chains(multichain.chains, reference)

    def __init__(self, topology: Topology, final_state_qn: dict[int, QN | Particle], reference: Topology | DecayChain, resonances: dict[tuple, tuple[Resonance]] | ResonanceDict | None = None, chains: list[DecayChain] | None = None, convention: Literal["helicity", "minus_phi"] = "helicity") -> None:
        if isinstance(reference, DecayChain):
            if reference.convention != convention:
                raise ValueError(f"Reference and chain must have the same convention. Found reference: {reference.convention} and self: {convention}!")
        super().__init__(topology, final_state_qn, resonances=resonances, chains=chains, convention=convention)
        self.reference: Topology = reference if isinstance(reference, Topology) else reference.topology

    def to_tuple(self, lambdas: dict):
        """Maps a {particle_key: helicity} dict to the sorted-key tuple used as a matrix key."""
        return tuple([lambdas[key] for key in self.final_state_keys])

    def _static_cache(self, momenta) -> dict:
        """Extends the parent's _static_cache with the alignment-specific
        entries aligned_matrix looks up: relative_wigner_angles, the
        per-particle alignment factor table, and the full alignment rotation
        matrix -- mirrors aligned_matrix's own computation exactly.
        """
        cache = super()._static_cache(momenta)
        wigner_rotation = self.reference.relative_wigner_angles(self.topology, momenta, convention=self.convention)
        cache[(id(self), "wigner_rotation")] = wigner_rotation
        alignment_factors = _per_particle_alignment_factors(self.final_state_qn, self.final_state_keys, wigner_rotation)
        cache[(id(self), "alignment_factors")] = alignment_factors
        cache[(id(self), "rotation_matrix")] = _alignment_rotation_matrix(self.helicities, self.final_state_keys, alignment_factors)
        return cache

    @property
    def aligned_matrix(self):
        """
        Returns a function f(h0, arguments, momenta) -> dict mapping final-state
        helicity tuples to their amplitude, rotated into the reference frame via
        Wigner D-matrices computed fresh from momenta on every call.
        """
        m = self.matrix
        def f(h0, arguments: dict, momenta: dict, cache: dict | None = None):
            matrix = m(h0, arguments, momenta, cache=cache)
            # relative_wigner_angles depends only on (topology, momenta), not on
            # h0 -- cache it too so it isn't recomputed on every h0 iteration.
            wigner_rotation = _momenta_cached(
                self.momenta_cache, cache, (id(self), "wigner_rotation"),
                lambda: self.reference.relative_wigner_angles(self.topology, momenta, convention=self.convention),
            )
            # Per-particle Wigner-D lookup table instead of recomputing
            # wigner_capital_d fresh for every (lambdas, lambdas_) tuple pair
            # below -- see _per_particle_alignment_factors.
            alignment_factors = _momenta_cached(
                self.momenta_cache, cache, (id(self), "alignment_factors"),
                lambda: _per_particle_alignment_factors(self.final_state_qn, self.final_state_keys, wigner_rotation),
            )
            # The full (n_helicities, n_helicities) alignment matrix, h0-independent
            # like alignment_factors -- see _alignment_rotation_matrix for why this
            # is built once as a matrix and applied via one contraction, rather
            # than combined per (lambdas, lambdas_) pair with a Python loop.
            rotation_matrix = _momenta_cached(
                self.momenta_cache, cache, (id(self), "rotation_matrix"),
                lambda: _alignment_rotation_matrix(self.helicities, self.final_state_keys, alignment_factors),
            )
            matrix_vec = np.stack([matrix[self.to_tuple(lambdas_)] for lambdas_ in self.helicities], axis=0)
            result_vec = np.einsum("ij...,j...->i...", rotation_matrix, matrix_vec)
            return {
                self.to_tuple(lambdas): result_vec[i]
                for i, lambdas in enumerate(self.helicities)
            }
        return f





