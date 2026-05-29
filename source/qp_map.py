import cv2
import numpy as np

def generate_qp_map(frame, base_qp=60):
    """
    Analyzes an image block by block to determine the optimal Quantization Parameter (QP).
    Uses Spatial Variance and Luminance Masking.
    """
    # Convert to grayscale to analyze the structure and brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Create an empty grid to hold QP numbers
    qp_map = np.zeros((height // 8, width // 8), dtype=np.int32)
    
    # Find the average variance of the whole frame to use as our baseline comparison
    frame_avg_std = np.std(gray)
    if frame_avg_std == 0: 
        frame_avg_std = 1.0 # Prevent crashing by dividing by zero
    
    # Loop through the image in 8x8 blocks
    for y in range(height // 8):
        for x in range(width // 8):
            
            # Extract the exact 8x8 block of pixels
            y_start = y * 8
            y_end = y_start + 8
            x_start = x * 8
            x_end = x_start + 8
            block = gray[y_start:y_end, x_start:x_end]
            
            # spatial variance
            block_standard_deviation = np.std(block)
            variance_ratio = block_standard_deviation / frame_avg_std
            
            # luminance masking
            block_average_brightness = np.mean(block)
            
            # normalize
            brightness_difference = block_average_brightness - 128.0
            luminance_multiplier = 1.0 + ((brightness_difference / 128.0) ** 2)
            
            # qp
            raw_calculated_qp = base_qp * variance_ratio * luminance_multiplier
            
            # set limit (> 0.85 is good for ssim so need to set limit for qp)
            final_qp_integer = int(np.clip(np.round(raw_calculated_qp), 10, 100))
            
            qp_map[y, x] = final_qp_integer
            
    return qp_map

def create_heatmap(qp_map, original_shape):
    """Transforms the matrix of numbers into a colorful heatmap for display."""
    height, width = original_shape[:2]
    resized = cv2.resize(qp_map.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)