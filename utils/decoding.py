import numpy as np


def flip(decoded_data):
    return 1 - decoded_data


def decode(encoded_data):
    num_samples, encoded_height, encoded_width, num_channels = encoded_data.shape
    decoded_data = np.zeros((num_samples, encoded_height * 2, encoded_width * 2))

    for sample in range(num_samples):
        for i in range(encoded_height):
            for j in range(encoded_width):
                for c in range(num_channels):
                    if c == 0:
                        decoded_data[sample, 2 * i, 2 * j] = encoded_data[sample, i, j, c]
                    elif c == 1:
                        decoded_data[sample, 2 * i, 2 * j + 1] = encoded_data[sample, i, j, c]
                    elif c == 2:
                        decoded_data[sample, 2 * i + 1, 2 * j] = encoded_data[sample, i, j, c]
                    elif c == 3:
                        decoded_data[sample, 2 * i + 1, 2 * j + 1] = encoded_data[sample, i, j, c]

    return decoded_data
