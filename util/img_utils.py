import numpy as np
import torch
import scipy as sp
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def compute_last_diff_step(t, t_start, N_bi):
    if t < N_bi:
        t_stop = int(t_start* 0.7)
    else:
        t_stop = 0
    return t_stop

def interpolate_image_efficient(Y, H, search_radius=5):
    channels, height, width = Y.shape
    interpolated = Y.clone()

    # Create a meshgrid for coordinates
    xx, yy = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mask = H[0]
    for c in range(channels):
        channel = Y[c]

        # Identify masked pixels
        masked_y, masked_x = torch.where(~mask)

        for i, j in zip(masked_y, masked_x):
            # Define a search window
            xmin, xmax = max(0, j - search_radius), min(width, j + search_radius)
            ymin, ymax = max(0, i - search_radius), min(height, i + search_radius)

            # Extract the search window
            window = channel[ymin:ymax, xmin:xmax]
            window_mask = mask[ymin:ymax, xmin:xmax]

            # Find valid (non-masked) pixels in the window
            valid_y, valid_x = torch.where(window_mask)

            if len(valid_y) > 0:
                # Compute distances to valid pixels in the window
                distances = torch.sqrt((valid_x - (j - xmin))**2 + (valid_y - (i - ymin))**2)

                # Find the nearest valid pixel
                nearest_idx = torch.argmin(distances)
                interpolated[c, i, j] = window[valid_y[nearest_idx], valid_x[nearest_idx]]

    return interpolated

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


"""
Helper functions for new types of inverse problems
"""

def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)


inv_transform = transforms.Compose([
        transforms.Normalize((-1), (2)),
        transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    ])

def clear_color(X):
    return inv_transform(X).squeeze().permute(1, 2, 0).cpu().numpy()

#def clear_color(x):
#    if torch.is_complex(x):
#        x = torch.abs(x)
#    x = x.detach().cpu().squeeze().numpy()
#    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


def unnormalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img / scaling


def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling


def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)
