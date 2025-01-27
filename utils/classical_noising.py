import torch 
import numpy as np

# Old functions (not great for noising but possibly usable)
# def scramble_vectors(encoded_data, seed):
#     np.random.seed(seed)
#     scrambled_vectors = []
#     timesteps = 1
    
#     for _ in range(timesteps):
#         for i in range(len(encoded_data)):
#             gaussian_noise = np.random.normal(0, 0.1, encoded_data[i].shape)
#             scrambled_state = encoded_data[i] + gaussian_noise
            
#             scrambled_vectors.append(scrambled_state)
    
#     return np.array(scrambled_vectors)

# def scramble_state_vectors(encoded_data, seed):
#     np.random.seed(seed)
#     scrambled_vectors = []
#     for i in range(len(encoded_data)):
#         gaussian_matrix = np.random.normal(0, 0.1, (16, 16))
#         scrambled_state = np.multiply(gaussian_matrix, encoded_data[i])
#         scrambled_vectors.append(scrambled_state)
#     return np.array(scrambled_vectors).reshape(len(encoded_data), 8, 8, 4)

def scramble_vectors(vectors, num_steps=1000, beta_min=0.0001, beta_max=0.02, seed=42):
    """
    Applies a forward diffusion process by gradually adding Gaussian noise to the input 16x16 vectors.
    This is an implementation of a linear scheduler with set optional parameters.

    Args:
        vectors (np.ndarray): Input array of shape (num_samples, 16, 16).
        num_steps (int): Number of diffusion steps.
        beta_min (float): Minimum noise variance.
        beta_max (float): Maximum noise variance.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Noised version of the input vectors of shape (num_samples, 16, 16).
    """
    rng = np.random.default_rng(seed)
    num_samples, height, width = vectors.shape

    betas = np.linspace(beta_min, beta_max, num_steps)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    t = rng.integers(0, num_steps, size=num_samples)
    alpha_bar_t = alpha_bars[t].reshape(-1, 1, 1)

    noise = rng.normal(size=(num_samples, height, width))
    scrambled_vectors = np.sqrt(alpha_bar_t) * vectors + np.sqrt(1 - alpha_bar_t) * noise

    return scrambled_vectors