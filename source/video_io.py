import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Import your toolboxes
from qp_map import generate_qp_map, create_heatmap
from quantization import compress_frame
from metrics import calculate_ssim, calculate_bitrate_savings, calculate_psnr

# ==========================================
# 1. BRIGHT / POSITIVE MINIMALISM CSS
# ==========================================
st.set_page_config(page_title="Quantization Optimization", layout="wide")

st.markdown("""
    <style>
    /* High-end minimalist typography (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Pure white, uplifting, bright background */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Inter', -apple-system, sans-serif;
        color: #1A1A1A;
    }
    
    /* Clean, thin, dark typography for maximum readability */
    h1, h2, h3 {
        color: #000000 !important;
        font-weight: 500;
        letter-spacing: -0.03em;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* Professional, light-themed buttons */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 6px !important;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .stButton>button:hover {
        background-color: #F9FAFB !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04) !important;
        transform: translateY(-1px);
    }

    /* Stripped-down Metric Cards */
    div[data-testid="metric-container"] {
        background-color: transparent;
        border: none;
        padding: 10px 0px;
    }
    div[data-testid="metric-container"] label {
        color: #6B7280 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
    }
    div[data-testid="metric-container"] div {
        color: #111827 !important;
        font-weight: 400 !important;
    }

    /* Crisp images with very soft, elegant borders */
    img {
        border-radius: 6px;
        border: 1px solid #F3F4F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Clean layout */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False

# ==========================================
# 3. PURE FUNCTIONAL HEADER
# ==========================================
st.write("<br>", unsafe_allow_html=True)
st.title("Perceptual Quantization Optimization")

# File Uploader
uploaded_file = st.file_uploader("", type=['y4m', 'mp4'], label_visibility="collapsed")

st.write("<br>", unsafe_allow_html=True)

# Control Buttons
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    if st.button("Start Processing", use_container_width=True):
        st.session_state.is_processing = True
        st.session_state.stop_requested = False
with col_btn2:
    if st.button("Stop & Save", use_container_width=True):
        st.session_state.stop_requested = True

st.markdown("---")

# ==========================================
# 4. THE PROCESSING LOOP
# ==========================================
if uploaded_file is not None and st.session_state.is_processing:
    
    with st.spinner('Initializing processing engine...'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        original_name = uploaded_file.name.split('.')[0]
        output_filename = f"{original_name}_output.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- THEATER ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("RAW INPUT")
        raw_screen = st.empty()
    with col2:
        st.caption("ADAPTIVE QP MAP")
        map_screen = st.empty()
    with col3:
        st.caption("COMPRESSED OUTPUT")
        compressed_screen = st.empty()
        
    st.write("<br>", unsafe_allow_html=True)
    
    # --- METRICS ---
    # Upgraded to 3 columns to include PSNR
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    psnr_metric = metric_col1.empty()
    ssim_metric = metric_col2.empty()
    bitrate_metric = metric_col3.empty()

    frame_counter = 0
    last_ssim = 1.0
    last_bit_savings = 0.0
    
    # --- LIVE LOOP ---
    while cap.isOpened():
        if st.session_state.stop_requested:
            break
            
        success, raw_frame = cap.read()
        if not success:
            break
            
        # 0.75 scale for excellent quality and good performance
        small_frame = cv2.resize(raw_frame, (0, 0), fx=0.75, fy=0.75)
        frame_counter += 1
        
        # Engine Math
        my_qp_map = generate_qp_map(small_frame)
        heatmap = create_heatmap(my_qp_map, small_frame.shape)
        compressed_frame, adaptive_zeros, total_coeffs = compress_frame(small_frame, my_qp_map)
        
        # Restore to Original Size for saving to MP4
        out_frame = cv2.resize(compressed_frame, (width, height))
        out_video.write(out_frame)
        
        # PSNR is fast enough to calculate on EVERY frame
        current_psnr = calculate_psnr(small_frame, compressed_frame)
        
        # Heavy Math calculates only 1 in 20 frames!
        if frame_counter % 20 == 1:
            baseline_zeros = int(total_coeffs * 0.40)
            last_bit_savings = calculate_bitrate_savings(baseline_zeros, adaptive_zeros)
            
            gray_raw = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray_compressed = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
            last_ssim = calculate_ssim(gray_raw, gray_compressed)
            
        # Update Screens
        raw_screen.image(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        map_screen.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), channels="RGB")
        compressed_screen.image(cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Update Metric Cards
        psnr_metric.metric("PSNR (QUALITY)", f"{current_psnr:.2f} dB")
        ssim_metric.metric("STRUCTURAL SIMILARITY (SSIM)", f"{last_ssim:.4f}")
        bitrate_metric.metric("BITRATE SAVINGS", f"+{last_bit_savings:.1f} %")

    # Cleanup
    cap.release()
    out_video.release()
    st.session_state.is_processing = False
    
    st.write("<br>", unsafe_allow_html=True)
    
    st.success("Processing complete. The output file is ready.")
    
    with open(output_path, 'rb') as f:
        st.download_button(
            label="Download Output .mp4",
            data=f,
            file_name=output_filename,
            mime='video/mp4'
        )