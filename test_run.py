import torch
from utils.quantum_diffusion import QuantumDiffusionModel

def main():
    
    n_qubits = 4
    n_layers = 2
    input_dim = 4 * 4 * 4    
    hidden_dim = 16
    output_dim = input_dim

    model = QuantumDiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_qubits=n_qubits,
        n_layers=n_layers
    )


    x = torch.randn(2, input_dim)

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Smoke test passed!")

if __name__ == "__main__":
    main()
