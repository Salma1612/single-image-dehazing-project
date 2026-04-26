import numpy as np

def dehaze_v2(img):
    # Version 2: Improved Airlight Estimation
    # Uses average of top 0.1% brightest pixels in the dark channel for Airlight
    min_c = np.min(img, axis=2)
    flat_min_c = min_c.flatten()
    flat_img = img.reshape(-1, 3)
    # Get the indices of the top 0.1% brightest pixels
    num_pixels = len(flat_min_c)
    num_bright_pixels = int(num_pixels * 0.001)
    top_indices = np.argsort(flat_min_c)[-num_bright_pixels:]
    A = np.mean(flat_img[top_indices], axis=0) # Average RGB values of these pixels

    t = 1 - 0.95 * min_c
    t = np.clip(t, 0.1, 1)

    out = (img - A) / t[:, :, None] + A
    return np.clip(out, 0, 1)
