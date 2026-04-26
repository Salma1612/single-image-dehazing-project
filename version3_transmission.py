import numpy as np
from skimage.morphology import disk, erosion

def dehaze_v3(img):
    # Version 3: Refined Transmission Map using soft matting (or simpler, erosion for smoothing)
    min_c = np.min(img, axis=2)
    flat_min_c = min_c.flatten()
    flat_img = img.reshape(-1, 3)
    num_pixels = len(flat_min_c)
    num_bright_pixels = int(num_pixels * 0.001)
    top_indices = np.argsort(flat_min_c)[-num_bright_pixels:]
    A = np.mean(flat_img[top_indices], axis=0)

    # Basic dark channel for transmission
    t_initial = 1 - 0.95 * min_c

    # Apply a simple erosion for smoothing the transmission map
    selem = disk(3) # Use a disk-shaped structuring element
    t = erosion(t_initial, selem)
    t = np.clip(t, 0.1, 1)

    out = (img - A) / t[:, :, None] + A
    return np.clip(out, 0, 1)
