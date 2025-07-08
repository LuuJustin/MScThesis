import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

oai_files = ['OAI_part00.h5', ]

check_files = ['CHECK_part01.h5', ]

def adjust_brightness(img, factor):
    """img: NumPy array (3, H, W), factor >1 brightens, <1 darkens"""
    img = img.astype(np.float32) * factor
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)
def add_blur(image, sigma):
    return gaussian_filter(image, sigma=sigma)
def add_poisson_noise(img):
    noisy = np.random.poisson(img.astype(np.float32))
    return np.clip(noisy, 0, 255).astype(np.float32)

for path in oai_files:
    with h5py.File(path, "r") as f:
        for side in ["left_hip", "right_hip"]:
            imgs = f[f"{side}/images"][:]
            scores = f[f"{side}/scores"][:]
            sids = f[f"{side}/subject_ids"][:].astype(str)

            print(f"\nFile: {path}, Side: {side}")
            print(f"Total images: {len(imgs)}")

            for i, (img, score, sid) in enumerate(zip(imgs, scores, sids)):

                # Normalize
                img_display = np.transpose(img, (1, 2, 0))  # (224, 224, 3)
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                img_display = np.array(add_blur(adjust_brightness(img_display, 1.05), 1.5), np.float32)
                plt.imshow(img_display)

                plt.show()

                # Optional: break after first few images per file/side
                if i >= 0:  # Show only 3 images per side
                    break