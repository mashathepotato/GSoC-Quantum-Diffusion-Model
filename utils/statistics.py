from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

import numpy as np

def calculate_statistics(data):
    data = data.reshape(data.shape[0], -1)
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)
    return mean, covariance

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

def calculate_wasserstein(real_data, generated_data):
    num_samples = real_data.shape[0]
    wasserstein_scores = []
    for i in range(num_samples):
        w_score = wasserstein_distance(real_data[i].ravel(), generated_data[i].ravel())
        wasserstein_scores.append(w_score)
    return np.mean(wasserstein_scores)

# Same thing for FID 
def calculate_fid_stable(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    
    sigma1_eps = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2_eps = sigma2 + np.eye(sigma2.shape[0]) * eps

    covmean, _ = sqrtm(sigma1_eps @ sigma2_eps, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid