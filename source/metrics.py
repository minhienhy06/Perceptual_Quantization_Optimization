import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original_frame, compressed_frame):
    score, _ = ssim(original_frame, compressed_frame, data_range=255, full=True)
    return score

def calculate_psnr(original_frame, compressed_frame):
    return cv2.PSNR(original_frame, compressed_frame)

def calculate_bitrate_savings(baseline_zeros, adaptive_zeros):
    if baseline_zeros == 0: return 0.0
    savings = (adaptive_zeros - baseline_zeros) / baseline_zeros
    return savings * 100