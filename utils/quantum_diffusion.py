# quantum_diffusion.py
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import scipy
from numpy.linalg import norm
from skimage.metrics import structural_similarity as ssim

# Quantum model definitions
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch')
        def quantum_circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)


class QuantumDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_qubits, n_layers):
        super(QuantumDiffusionModel, self).__init__()
        self.n_qubits = n_qubits
        self.num_patches = input_dim // n_qubits
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)

    def forward(self, x):
        x_patched = x.view(-1, self.num_patches, self.n_qubits)
        outputs = [self.quantum_layer(x_patched[:, p]) for p in range(self.num_patches)]
        return torch.cat(outputs, dim=1)


# --- Utility functions ---
def decode(encoded_data, dim=32):
    num_samples, h, w, c = encoded_data.shape
    decoded = np.zeros((num_samples, dim*2, dim*2))
    for n in range(num_samples):
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    if k == 0: decoded[n, 2*i, 2*j]   = encoded_data[n, i, j, k]
                    if k == 1: decoded[n, 2*i, 2*j+1] = encoded_data[n, i, j, k]
                    if k == 2: decoded[n, 2*i+1, 2*j] = encoded_data[n, i, j, k]
                    if k == 3: decoded[n, 2*i+1, 2*j+1] = encoded_data[n, i, j, k]
    return decoded

def flip(decoded_data): return 1 - decoded_data

def calculate_statistics(data):
    data = data.reshape(data.shape[0], -1)
    return np.mean(data, axis=0), np.cov(data, rowvar=False)

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def calculate_ssim(real, recon):
    scores = [ssim(real[i], recon[i], data_range=recon[i].max()-recon[i].min())
              for i in range(min(len(real), len(recon)))]
    return np.mean(scores)


def train_model(train_encoded, val_encoded, train_scrambled, val_scrambled,
                n_qubits=8, n_layers=8, input_dim=32*32*4, hidden_dim=128, output_dim=32*32*4,
                num_epochs=50, lr=0.01):
    model = QuantumDiffusionModel(input_dim, hidden_dim, output_dim, n_qubits, n_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_values, val_loss_values = [], []

    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(train_scrambled.view(len(train_scrambled), -1))
        loss = criterion(outputs, train_encoded.view(len(train_encoded), -1))
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_scrambled.view(len(val_scrambled), -1))
            val_loss = criterion(val_outputs, val_encoded.view(len(val_encoded), -1))
            val_loss_values.append(val_loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Final reconstruction
    model.eval()
    with torch.no_grad():
        denoised = model(val_scrambled.view(len(val_scrambled), -1))
        denoised = denoised.view(len(val_scrambled), 32, 32, 4).detach().numpy()
        decoded = flip(decode(denoised))

    return model, decoded, loss_values, val_loss_values
