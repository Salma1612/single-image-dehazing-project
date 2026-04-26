import numpy as np

def dehaze_v1(img):
    # Version 1: Simple dark channel prior baseline (same as base_code.py)
    min_c = np.min(img, axis=2)
    t = 1 - 0.95 * min_c
    t = np.clip(t, 0.1, 1)
    A = np.max(img, axis=(0,1))
    out = (img - A) / t[:, :, None] + A
    return np.clip(out, 0, 1)
