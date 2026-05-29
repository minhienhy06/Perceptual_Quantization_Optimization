import cv2
import numpy as np

def compress_frame(frame, qp_map):
    # dct -> divide -> inverse dct
    height, width, channels = frame.shape
    
    # Create an empty 3D image to hold our newly compressed pixels
    reconstructed_image = np.zeros_like(frame, dtype=np.float32)
    
    total_zeros = 0
    total_coefficients = 0
    
    # Loop through the 3 color layers (0=Blue, 1=Green, 2=Red)
    for color_index in range(channels):
        
        # Extract just one color layer. (DCT math strictly requires float32 decimal numbers)
        single_color_layer = np.float32(frame[:, :, color_index])
        
        for y in range(height // 8):
            for x in range(width // 8):
                
                # Define block boundaries clearly
                y_start = y * 8
                y_end = y_start + 8
                x_start = x * 8
                x_end = x_start + 8
                
                # Extract the 8x8 block for this color
                block = single_color_layer[y_start:y_end, x_start:x_end]
                
                # Get the specific QP number for this block from our Brain
                block_qp = qp_map[y, x]
                
                #pixels to frequencies domain
                dct_frequencies = cv2.dct(block)
                
                # quantization
                quantized_frequencies = np.round(dct_frequencies / block_qp)
                
                # track 0s for bit savings
                total_zeros += np.sum(quantized_frequencies == 0)
                total_coefficients += 64
                
                # 4. De-quantize: Multiply back up (Zeros stay zero!)
                dequantized_frequencies = quantized_frequencies * block_qp
                
                # 5. Inverse Transform: Turn frequencies back into pixels
                reconstructed_block = cv2.idct(dequantized_frequencies)
                
                # reconstruct
                reconstructed_image[y_start:y_end, x_start:x_end, color_index] = reconstructed_block

    # force pixel from 0 to 255
    final_clean_image = np.uint8(np.clip(reconstructed_image, 0, 255))
    
    return final_clean_image, total_zeros, total_coefficients