from scipy.stats import rv_continuous
import numpy as np
import pennylane as qml
from numpy.linalg import norm

dev = qml.device("default.qubit", wires=2) 

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi)

@qml.qnode(dev)
def haar_random_unitary():
    phi1, omega1 = 2 * np.pi * np.random.uniform(size=2)
    theta1 = sin_sampler.rvs(size=1)
    
    phi2, omega2 = 2 * np.pi * np.random.uniform(size=2)
    theta2 = sin_sampler.rvs(size=1)
    
    qml.Rot(phi1, theta1, omega1, wires=0)
    qml.Rot(phi2, theta2, omega2, wires=1)
    
    return qml.state()

def apply_haar_scrambling(encoded_data, num_samples, seed=None):
    scrambled_vectors = []
    new_dim = encoded_data.shape[1]

    for sample in range(num_samples):
        scrambled_vector = []
        for _ in range(new_dim):
            channels = []
            for _ in range(new_dim):
                if seed is not None:
                    np.random.seed(seed)

                # Haar random unitary for 4D vector with 2 qubits
                scrambled_state = haar_random_unitary()

                scrambled_state = np.reshape(scrambled_state, (4,))
                scrambled_state /= np.linalg.norm(scrambled_state)

                channels.append(scrambled_state)

                if seed is not None:
                    seed += 1
            scrambled_vector.append(channels)
        scrambled_vectors.append(scrambled_vector)

    return np.array(scrambled_vectors)

import numpy as np
from numpy.linalg import qr, det, norm

def _sample_haar_unitary(dim, seed=None, force_su=False):
    """
    Sample a Haar-random unitary of size (dim x dim) using the QR method.
    If force_su=True, scale to make determinant 1 (in SU(dim)).
    """
    rng = np.random.default_rng(seed)
    # complex gaussian entries
    X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    # QR decomposition
    Q, R = qr(X)
    # make Q Haar distributed by normalizing diagonal phases of R
    diag_R = np.diag(R)
    lambda_phase = diag_R / np.abs(diag_R)
    Q = Q * (1.0 / lambda_phase)
    if force_su:
        # adjust global phase to make determinant 1
        ph = np.exp(-1j * np.angle(det(Q)) / dim)
        Q = Q * ph
    return Q

def _to_statevector(x):
    """
    Accepts:
    - already a 1D complex or real vector (treated as state amplitudes)
    - multi-d array (image) -> flattens
    Returns normalized complex statevector (1D numpy array).
    """
    v = np.ravel(x).astype(np.complex128)
    nrm = norm(v)
    if nrm == 0:
        # if zero vector, return basis |0...0>
        v = np.zeros_like(v)
        v[0] = 1.0 + 0j
        return v
    return v / nrm

def apply_global_haar_scrambling(encoded_data, seed=None, force_su=False, approx_threshold_qubits=8, approx_depth=6):
    """
    Apply a single global Haar-random unitary to each sample in encoded_data.

    Parameters
    ----------
    encoded_data : array-like with shape (N, ...) or (N_samples, state_dim)
        If shape is (N_samples, H, W, ...) each sample will be flattened into a statevector.
    seed : int or None
        Base RNG seed for reproducibility. Each sample will get a different derived seed.
    force_su : bool
        If True, scale the unitary to have determinant 1 (SU(dim)). Not usually required.
    approx_threshold_qubits : int
        If log2(state_dim) >= approx_threshold_qubits, the function raises an error recommending
        an approximate random-circuit approach (to avoid huge matrices). You can increase this if you have memory.
    approx_depth : int
        Depth used if you later choose to use the random-circuit approximation helper (not used in exact Haar).
    Returns
    -------
    scrambled_states : ndarray, shape (N_samples, 2**n)
        Each sample returned as a normalized complex statevector of length 2**n (n chosen per-run).
        Also returns the list of unitaries used so you can store them if you want invertibility.
    unitaries : list of ndarray
        The (dense) Haar unitaries applied to each sample (useful if you want to invert later).
    """
    data = np.asarray(encoded_data)
    # detect samples axis
    if data.ndim == 1:
        samples = data[np.newaxis, ...]
    elif data.ndim >= 2 and data.shape[0] == 1:
        samples = data
    else:
        samples = data

    N = samples.shape[0]
    # prepare statevectors
    statevecs = []
    for i in range(N):
        v = _to_statevector(samples[i])
        statevecs.append(v)
    # find required dimension and pad/truncate to nearest power of two
    lengths = [v.size for v in statevecs]
    maxlen = max(lengths)
    # choose dim = next power of two >= maxlen
    n_qubits = int(np.ceil(np.log2(maxlen)))
    dim = 2 ** n_qubits
    if n_qubits >= approx_threshold_qubits:
        raise MemoryError(
            f"Requested global Haar on {n_qubits} qubits (dim={dim}). This is large and will allocate {dim**2} complex numbers.\n"
            "Consider increasing approx_threshold_qubits or giving up. "
        )

    # pad/truncate statevectors to length dim
    padded_states = np.zeros((N, dim), dtype=np.complex128)
    for i, v in enumerate(statevecs):
        L = v.size
        if L <= dim:
            padded_states[i, :L] = v
            if L < dim:
                # remaining entries are zero; re-normalize to unit norm
                padded_states[i] /= norm(padded_states[i])
        else:
            # truncate (rare) and renormalize
            padded_states[i] = v[:dim] / norm(v[:dim])

    # prepare RNG
    base_rng = np.random.default_rng(seed)

    unitaries = []
    scrambled = np.zeros_like(padded_states, dtype=np.complex128)
    for i in range(N):
        # derive a per-sample seed for reproducibility
        sample_seed = int(base_rng.integers(0, 2**31)) if seed is not None else None
        U = _sample_haar_unitary(dim, seed=sample_seed, force_su=force_su)
        unitaries.append(U)
        scrambled[i] = U @ padded_states[i]
        # normalize numerically
        scrambled[i] /= norm(scrambled[i])

    return scrambled, unitaries


# SIMULATING HAAR VIA RANDOM CIRCUITS FOR MEMORY EFFICIENCY
def _make_random_circuit_params(n_qubits, depth, rng):
    """
    Return a params array of shape (depth, n_qubits, 3) for qml.Rot per-qubit.
    Each gate uses 3 Euler angles.
    """
    # shape: (depth, n_qubits, 3)
    return rng.random(size=(depth, n_qubits, 3)) * 2 * np.pi

def _apply_entangling_layer(n_qubits, layer_idx):
    """
    Brickwork entangling pattern:
      - even layer: CNOTs between (0,1), (2,3), ...
      - odd  layer: CNOTs between (1,2), (3,4), ...
    We use CNOT for speed and universality.
    """
    if n_qubits < 2:
        return
    if layer_idx % 2 == 0:
        # even layer: pairs (0,1), (2,3), ...
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    else:
        # odd layer: pairs (1,2), (3,4), ...
        for i in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

def build_scrambler_qnode(n_qubits, depth, params, device=None):
    """
    Build and return a QNode that:
      - Prepares an input state via QubitStateVector (argument `state`)
      - Applies random single-qubit Rot gates using `params` (shape depth x n_qubits x 3)
      - Applies brickwork CNOT entangling layers between Rot layers
      - Returns qml.state()
    """
    if device is None:
        device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="autograd")  # interface doesn't matter if not differentiating
    def qnode(state):
        # state is a 1D numpy array of length 2**n_qubits (complex)
        qml.QubitStateVector(state, wires=range(n_qubits))

        # apply the parameterized random circuit (params already fixed)
        for d in range(params.shape[0]):
            for w in range(n_qubits):
                a, b, c = params[d, w]
                qml.Rot(a, b, c, wires=w)
            _apply_entangling_layer(n_qubits, d)

        return qml.state()

    return qnode

def circuit_based_haar_scramble(encoded_states,
                                n_qubits,
                                depth=8,
                                seed=42,
                                shared_unitary=True,
                                device=None):
    """
    Apply a single-shot circuit-based global scramble to each state in encoded_states.

    Parameters
    ----------
    encoded_states : np.ndarray, shape (N, 2**n_qubits) OR (N, ...)
        If not flat, will be flattened. Must be length <= 2**n_qubits per sample;
        if shorter, samples are zero-padded and re-normalized.
    n_qubits : int
        Number of qubits the circuit uses. dim = 2**n_qubits.
    depth : int
        Number of random layers (controls scrambling strength).
    seed : int or None
        RNG seed for reproducibility.
    shared_unitary : bool
        If True, the same random circuit is applied to every sample (fast).
        If False, a new random circuit is sampled per-sample (slower).
    device : qml.device or None
        Optional PennyLane device to reuse.

    Returns
    -------
    scrambled_states : np.ndarray, shape (N, 2**n_qubits), complex128
    used_params : list of params arrays used (length 1 if shared_unitary else N)
    """
    rng = np.random.default_rng(seed)
    encoded = np.asarray(encoded_states)
    N = encoded.shape[0]

    dim = 2 ** n_qubits

    # Flatten and pad/truncate each sample to length dim
    statevecs = np.zeros((N, dim), dtype=np.complex128)
    for i in range(N):
        v = np.ravel(encoded[i]).astype(np.complex128)
        L = v.size
        if L == 0:
            statevecs[i, 0] = 1.0 + 0j
        elif L <= dim:
            statevecs[i, :L] = v
            # renormalize to unit norm
            statevecs[i] /= norm(statevecs[i])
        else:
            # truncate (rare)
            statevecs[i] = v[:dim] / norm(v[:dim])

    used_params = []

    # build one QNode for shared unitary (or reuse for each per-sample if not shared)
    if shared_unitary:
        params = _make_random_circuit_params(n_qubits, depth, rng)
        used_params.append(params)
        qnode = build_scrambler_qnode(n_qubits, depth, params, device=device)
        scrambled = np.zeros_like(statevecs, dtype=np.complex128)
        for i in range(N):
            out = qnode(statevecs[i])
            scrambled[i] = np.array(out, dtype=np.complex128)
            scrambled[i] /= norm(scrambled[i])
        return scrambled, used_params
    else:
        scrambled = np.zeros_like(statevecs, dtype=np.complex128)
        for i in range(N):
            params_i = _make_random_circuit_params(n_qubits, depth, rng)
            used_params.append(params_i)
            qnode_i = build_scrambler_qnode(n_qubits, depth, params_i, device=device)
            out = qnode_i(statevecs[i])
            scrambled[i] = np.array(out, dtype=np.complex128)
            scrambled[i] /= norm(scrambled[i])
        return scrambled, used_params
