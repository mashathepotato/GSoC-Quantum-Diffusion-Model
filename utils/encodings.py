# encoding_methods.py
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

######################################
# Utility: get sparse pixel locations
######################################
def get_sparse_indices(image, threshold=1e-6):
    """
    Returns indices and values of non-zero (or significant) pixels.
    threshold: cutoff below which values are ignored as zero.
    """
    flat_img = image.flatten()
    nonzero_indices = np.where(np.abs(flat_img) > threshold)[0]
    nonzero_values = flat_img[nonzero_indices]
    return nonzero_indices, nonzero_values


######################################
# 1. Angle Encoding
######################################
def angle_encoding(image, wires, threshold=1e-6):
    """
    Encode pixel values as rotation angles on corresponding qubits.
    Only encodes nonzero pixels.
    """
    idx, vals = get_sparse_indices(image, threshold)
    n_wires = len(wires)
    for i, v in zip(idx, vals):
        qml.RY(v, wires=wires[i % n_wires])


######################################
# 2. Amplitude Encoding
######################################
def amplitude_encoding(image, wires, threshold=1e-6):
    """
    Encode the image into amplitudes of a quantum state.
    Sparse-aware: normalizes only nonzero pixels.
    """
    flat_img = image.flatten()
    idx, vals = get_sparse_indices(flat_img, threshold)

    state = np.zeros(len(flat_img))
    state[idx] = vals
    norm = np.linalg.norm(state)

    if norm > 0:
        state = state / norm

    qml.AmplitudeEmbedding(state, wires=wires, normalize=True)


######################################
# 3. Sinusoidal (Trigonometric) Encoding
######################################
def sinusoidal_encoding(image, wires, threshold=1e-6):
    """
    Encode values using sin/cos features across qubits.
    Sparse-aware: only applies encoding for significant pixels.
    """
    idx, vals = get_sparse_indices(image, threshold)
    for i, v in zip(idx, vals):
        qml.RY(np.sin(v), wires=wires[i % len(wires)])
        qml.RZ(np.cos(v), wires=wires[i % len(wires)])


######################################
# 4. IQP Encoding
######################################
def iqp_encoding(image, wires, threshold=1e-6):
    """
    Implements an IQP embedding circuit.
    Sparse-aware: uses only significant pixels.
    """
    idx, vals = get_sparse_indices(image, threshold)

    # Step 1: Hadamards
    for w in wires:
        qml.Hadamard(wires=w)

    # Step 2: Parameterized Z-rotations (sparse)
    for i, v in zip(idx, vals):
        qml.RZ(v, wires=wires[i % len(wires)])

    # Step 3: Entangling layer
    for i in range(len(wires)):
        for j in range(i+1, len(wires)):
            qml.CZ(wires=[wires[i], wires[j]])


######################################
# Example usage (standalone test)
######################################
if __name__ == "__main__":
    n_qubits = 12  # enough for 64x64 amplitude encoding (2^12 = 4096)
    dev = qml.device("default.qubit", wires=n_qubits)

    image = np.random.rand(64, 64)  # fake test image

    @qml.qnode(dev)
    def circuit():
        # Try one encoding at a time:
        amplitude_encoding(image, wires=range(n_qubits))
        return qml.state()

    state = circuit()
    print("Encoded state shape:", state.shape)
