import torch 
import numpy as np
from utils.decoding import decode, flip

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
