import numpy as np
import h5py
import torch
from angle_encoding_script import angle_encoding

channel = 1
filename = f"C:/Users/realc/OneDrive/Documents/GSOC/data/QG{channel}_normalized_16x16_100k"
data_X, data_y = np.array(h5py.File(filename, "r")['X']), np.array(h5py.File(filename, "r")['y'])

num_samples = 100000

encoded_data = [angle_encoding(data_X, sample) for sample in range(num_samples)]
encoded_data = torch.tensor(np.array(encoded_data), dtype=torch.float32)
print(encoded_data.shape)


torch.save(encoded_data, f"C:/Users/realc/OneDrive/Documents/GSOC/data/Q{channel}_16x16_100k_encoded.pt")