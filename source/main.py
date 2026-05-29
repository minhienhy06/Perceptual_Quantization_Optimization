import cv2
import numpy as np
from qp_map import generate_qp_map, create_heatmap
from quantization import compress_frame
from metrics import calculate_ssim, calculate_bitrate_savings, calculate_psnr

VIDEO_PATH = "../dataset/foreman_cif.y4m"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    is_paused = False 
    
    # frame counting
    frame_counter = 0
    last_ssim_score = 1.0 
    
    print("Controls: Spacebar (Pause/Play), 'q' (Quit)")

    while True:
        if not is_paused:
            success, raw_frame = cap.read()
            if not success:
                break
            
            # size control
            raw_frame = cv2.resize(raw_frame, (0, 0), fx=0.6, fy=0.6)
            frame_counter += 1
                
            my_qp_map = generate_qp_map(raw_frame)
            compressed_frame, adaptive_zeros, total_coeffs = compress_frame(raw_frame, my_qp_map)
            
            # THE METRICS (Fast Metrics calculated every frame)
            baseline_zeros = int(total_coeffs * 0.40)
            bit_savings = calculate_bitrate_savings(baseline_zeros, adaptive_zeros)
            
            # PSNR is ultra-fast, so we calculate it every frame!
            psnr_score = calculate_psnr(raw_frame, compressed_frame)
            
            # THE HEAVY METRIC (Calculated 1 out of 20 frames)
            if frame_counter % 20 == 1:
                gray_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                gray_compressed = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
                last_ssim_score = calculate_ssim(gray_raw, gray_compressed)
            
            # Print all 3 metrics to the terminal
            print(f"PSNR: {psnr_score:.2f} dB | SSIM: {last_ssim_score:.4f} | Bitrate Savings: {bit_savings:.1f}%")
            
            # VISUALS
            heatmap = create_heatmap(my_qp_map, raw_frame.shape)
            cv2.imshow("1. Original Raw Video", raw_frame)
            cv2.imshow("2. Adaptive QP Heatmap", heatmap)
            cv2.imshow("3. Compressed Output", compressed_frame)

        # wait
        key = cv2.waitKey(0 if is_paused else 15) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused
            print("Paused" if is_paused else "Playing")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()