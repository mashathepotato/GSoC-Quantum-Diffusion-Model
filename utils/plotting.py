import numpy as np
import matplotlib.pyplot as plt

def plot_mean_decoded_images(decoded_images):
    mean_image = np.mean(decoded_images, axis=0)

    plt.figure(figsize=(5, 5))
    plt.imshow(mean_image, cmap='viridis')
    plt.title("Mean of All Generated Decoded Samples")
    plt.colorbar()
    plt.show()

def plot_all_decoded_images(decoded_images, grid_size=(10, 10)):
    num_images = decoded_images.shape[0]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 20))

    image_idx = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if image_idx < num_images:
                axes[i, j].imshow(decoded_images[image_idx], cmap='viridis')
                axes[i, j].axis('off')
                image_idx += 1
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_decoded_images_without_noise(decoded_images, grid_size=(4, 4), threshold_percent=90):
    num_images = decoded_images.shape[0]
    denoised_images = []

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 20))

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i * grid_size[1] + j) < num_images:
                image = decoded_images[i * grid_size[1] + j]

                threshold = np.percentile(image, threshold_percent)

                image_without_noise = np.where(image > threshold, image, 0.0)

                denoised_images.append(image_without_noise)

                axes[i, j].imshow(image_without_noise, cmap='viridis')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    denoised_images = np.array(denoised_images)
    mean_denoised_image = np.mean(denoised_images, axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_denoised_image, cmap='viridis')
    plt.title(f"Mean of All Denoised Images (Top {threshold_percent}% Values)")
    plt.colorbar()
    plt.show()

    return denoised_images