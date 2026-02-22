import numpy as np

def decode(encoded_data):

    if not isinstance(encoded_data, np.ndarray):
        raise TypeError("encoded_data must be a NumPy array")

    if encoded_data.ndim != 4:
        raise ValueError(
            "encoded_data must have shape (num_samples, height, width, channels)"
        )

    n, h, w, c = encoded_data.shape

    if c <= 0:
        raise ValueError("Number of channels must be positive")

    k = int(np.sqrt(c))

    if k * k != c:
        raise ValueError(
            f"Number of channels ({c}) must be a perfect square"
        )

    decoded = (
        encoded_data
        .reshape(n, h, w, k, k)
        .transpose(0, 1, 3, 2, 4)
        .reshape(n, h * k, w * k)
    )

    return decoded
