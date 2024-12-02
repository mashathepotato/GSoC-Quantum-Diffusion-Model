import torch 
import numpy as np

def flip(decoded_data):
    return 1 - decoded_data

def decode(encoded_data):
    num_samples, encoded_height, encoded_width, num_channels = encoded_data.shape
    decoded_data = np.zeros((num_samples, 16, 16))

    for sample in range(num_samples):
        for i in range(encoded_height):
            for j in range(encoded_width):
                for c in range(num_channels):
                    if c == 0:
                        decoded_data[sample, 2*i, 2*j] = encoded_data[sample, i, j, c]
                    elif c == 1:
                        decoded_data[sample, 2*i, 2*j+1] = encoded_data[sample, i, j, c]
                    elif c == 2:
                        decoded_data[sample, 2*i+1, 2*j] = encoded_data[sample, i, j, c]
                    elif c == 3:
                        decoded_data[sample, 2*i+1, 2*j+1] = encoded_data[sample, i, j, c]

    return decoded_data

def generate_new_images(model, num_images, input_dim=8*8*4):
    model.eval()  
    with torch.no_grad():
        random_noise = torch.randn(num_images, input_dim)
        generated_data = model(random_noise)
        generated_data = flip(generated_data.view(num_images, 8, 8, 4).detach().numpy())
        decoded_images = decode(generated_data) 

        # for i in range(num_images):
        #     fig, axes = plt.subplots(1, 5, figsize=(10, 2))

        #     for qubit in range(4):
        #         im = axes[qubit].imshow(generated_data[i, :, :, qubit], cmap='viridis')
        #         axes[qubit].set_title(f"Encoded {qubit+1}")
        #         fig.colorbar(im, ax=axes[qubit])

        #     im = axes[4].imshow(decoded_images[i], cmap='viridis')
        #     axes[4].set_title("Decoded")
        #     fig.colorbar(im, ax=axes[4])
            
        #     plt.tight_layout()
        #     plt.show()
        
    return decoded_images