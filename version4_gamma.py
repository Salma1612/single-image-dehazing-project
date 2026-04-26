import numpy as np
from skimage.morphology import disk, erosion

def dehaze_v4(img):
    # Version 4: Gamma Correction after Dehazing
    min_c = np.min(img, axis=2)
    flat_min_c = min_c.flatten()
    flat_img = img.reshape(-1, 3)
    num_pixels = len(flat_min_c)
    num_bright_pixels = int(num_pixels * 0.001)
    top_indices = np.argsort(flat_min_c)[-num_bright_pixels:]
    A = np.mean(flat_img[top_indices], axis=0)

    t_initial = 1 - 0.95 * min_c
    selem = disk(3)
    t = erosion(t_initial, selem)
    t = np.clip(t, 0.1, 1)

    out = (img - A) / t[:, :, None] + A

    # Apply gamma correction
    gamma = 1.2 # A typical value, can be tuned
    out_gamma_corrected = np.power(out, 1/gamma)

    return np.clip(out_gamma_corrected, 0, 1)
