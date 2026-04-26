import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

# Import dehazing functions from separate files
from base_code import base_dehaze
from improved_code import improved_dehaze

def show(img, title):
    img = np.clip(img, 0, 1)
    plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

# ================= LOAD =================
hazy = cv2.imread("IMG.jpg").astype(np.float32) / 255.0
gt   = cv2.imread("IMG1.jpg").astype(np.float32) / 255.0

# ================= DEHAZE =================
base = base_dehaze(hazy)
improved = improved_dehaze(hazy)

# ==========================================================
# METRICS CALCULATION
# ==========================================================
psnr_base = psnr_fn(gt, base, data_range=1)
psnr_imp  = psnr_fn(gt, improved, data_range=1)

ssim_base = ssim_fn(gt, base, data_range=1, channel_axis=2)
ssim_imp  = ssim_fn(gt, improved, data_range=1, channel_axis=2)

# ================= SAVE RESULTS =================
# Create output directory
output_dir = "03_Results_Output_Images"
os.makedirs(output_dir, exist_ok=True)

# Save metrics to file
metrics_path = os.path.join(output_dir, "metrics_results.txt")
with open(metrics_path, "w") as f:
    f.write("========== FINAL RESULTS ==========\n")
    f.write(f"Base PSNR      : {psnr_base:.2f}\n")
    f.write(f"Improved PSNR  : {psnr_imp:.2f}\n")
    f.write(f"Base SSIM      : {ssim_base:.4f}\n")
    f.write(f"Improved SSIM  : {ssim_imp:.4f}\n")

# Save images
cv2.imwrite(os.path.join(output_dir, "hazy_input.png"), (hazy * 255).astype(np.uint8)[:, :, ::-1]) # OpenCV uses BGR, convert to RGB for saving
cv2.imwrite(os.path.join(output_dir, "ground_truth.png"), (gt * 255).astype(np.uint8)[:, :, ::-1]) # OpenCV uses BGR, convert to RGB for saving
cv2.imwrite(os.path.join(output_dir, "base_output.png"), (base * 255).astype(np.uint8)[:, :, ::-1]) # OpenCV uses BGR, convert to RGB for saving
cv2.imwrite(os.path.join(output_dir, "improved_output.png"), (improved * 255).astype(np.uint8)[:, :, ::-1]) # OpenCV uses BGR, convert to RGB for saving

print("========== FINAL RESULTS ==========")
print(f"Base PSNR      : {psnr_base:.2f}")
print(f"Improved PSNR  : {psnr_imp:.2f}")
print(f"Base SSIM      : {ssim_base:.4f}")
print(f"Improved SSIM  : {ssim_imp:.4f}")

# ================= PLOT =================
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
show(hazy, "Hazy Input")

plt.subplot(1,4,2)
show(base, "Base Output")

plt.subplot(1,4,3)
show(improved, "Improved Output")

plt.subplot(1,4,4)
show(gt, "Ground Truth")

plt.tight_layout()
plt.show()
