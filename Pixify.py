import streamlit as st
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image, ImageOps
import concurrent.futures
# For side-by-side comparison slider
from streamlit_image_comparison import image_comparison

##########################################################
# 1) LOADING & CROPPING
##########################################################
def load_single_target(file) -> np.ndarray:
    """
    Load a single target image using OpenCV.
    """
    file.seek(0)
    arr = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def process_file_with_orientation(i, f, target_size):
    """
    Process a single part image file:
      - Re-read from file,
      - Correct orientation using EXIF data,
      - Use thumbnail to scale to target_size,
      - Convert to RGB and then to a NumPy array.
    Returns a tuple of (index, image or None).
    """
    f.seek(0)
    try:
        im = Image.open(f)
        # Correct orientation based on EXIF (this can be expensive)
        im = ImageOps.exif_transpose(im)
        im.thumbnail(target_size, Image.Resampling.LANCZOS)
        im_rgb = im.convert("RGB")
        return i, np.array(im_rgb)
    except Exception:
        return i, None

@st.cache_data(show_spinner=False)
def load_part_images(files):
    """
    Load part images by applying orientation correction, thumbnailing, and converting to NumPy arrays.
    This function processes files concurrently and shows a progress bar.
    All part images are resized to 118 x 118.
    """
    target_size = (118, 118)
    images = [None] * len(files)
    progress_bar = st.progress(0)
    total = len(files)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file_with_orientation, i, f, target_size): i for i, f in enumerate(files)}
        for future in concurrent.futures.as_completed(futures):
            result_index, result_img = future.result()
            images[result_index] = result_img
            completed += 1
            progress_bar.progress(completed / total)
    return [img for img in images if img is not None]

def fractional_crop(image: np.ndarray, top_frac: float, bottom_frac: float,
                    left_frac: float, right_frac: float) -> np.ndarray:
    """
    Crop an image based on fractional percentages for top, bottom, left, and right.
    """
    h, w, _ = image.shape
    t_px = int(h * top_frac)
    b_px = int(h * (1 - bottom_frac))
    l_px = int(w * left_frac)
    r_px = int(w * (1 - right_frac))
    t_px = max(0, min(t_px, h))
    b_px = max(0, min(b_px, h))
    l_px = max(0, min(l_px, w))
    r_px = max(0, min(r_px, w))
    if t_px >= b_px or l_px >= r_px:
        return image
    return image[t_px:b_px, l_px:r_px]

##########################################################
# 2) UTILS
##########################################################
def alpha_blend(original: np.ndarray, mosaic: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two images using a given alpha value.
    If sizes differ, resize the original to match the mosaic.
    """
    if original.shape != mosaic.shape:
        original = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(original, alpha, mosaic, 1 - alpha, 0)

def compute_avg_color(img: np.ndarray) -> np.ndarray:
    """
    Compute the average color of an image.
    """
    return img.mean(axis=(0, 1))

##########################################################
# 3) NAIVE K-MEANS (using one cluster per part image)
##########################################################
def naive_kmeans_average_color(parts, max_iters=10, analysis_size=(30, 30)):
    """
    For each part image, compute its average color (used as its descriptor).
    """
    n_parts = len(parts)
    if n_parts <= 0:
        return None, None, {}, {}
    descriptors = []
    part_analysis = {}
    part_full = {}
    for i, img in enumerate(parts):
        part_full[i] = img
        small = cv2.resize(img, analysis_size, interpolation=cv2.INTER_AREA)
        part_analysis[i] = small
        desc = compute_avg_color(small)
        descriptors.append(desc)
    descriptors_arr = np.array(descriptors, dtype=np.float32)
    cluster_centers = descriptors_arr.copy()
    cluster_assignments = np.arange(n_parts, dtype=np.int32)
    return cluster_centers, cluster_assignments, part_analysis, part_full

##########################################################
# 4) BUILD FINAL MOSAIC
##########################################################
def build_final_mosaic_kmeans(scaled_target: np.ndarray,
                              cluster_centers: np.ndarray,
                              cluster_assignments: np.ndarray,
                              part_analysis: dict,
                              part_full: dict,
                              tile_side_px: int,
                              progress_bar=None):
    """
    For each tile in the target image, choose the best matching part image
    (based on average color) and construct the final mosaic.
    An optional progress_bar (st.progress) is updated as tiles are processed.
    """
    h, w, _ = scaled_target.shape
    squares_w = w // tile_side_px
    squares_h = h // tile_side_px
    used_w = squares_w * tile_side_px
    used_h = squares_h * tile_side_px
    sub_img = scaled_target[:used_h, :used_w]

    usage_count = {pid: 0 for pid in part_full.keys()}
    mosaic = np.zeros((used_h, used_w, 3), dtype=np.uint8)
    tile_info = []
    total_tiles = squares_w * squares_h
    counter = 0
    update_interval = max(1, total_tiles // 100)
    for r in range(squares_h):
        for c in range(squares_w):
            y1 = r * tile_side_px
            x1 = c * tile_side_px
            tile = sub_img[y1:y1 + tile_side_px, x1:x1 + tile_side_px]
            tile_desc = compute_avg_color(tile)
            diff = cluster_centers - tile_desc
            dist2 = np.sum(diff * diff, axis=1)
            best_idx = np.argmin(dist2)
            anal_size = part_analysis[best_idx].shape[:2][::-1]
            tile_small = cv2.resize(tile, anal_size, interpolation=cv2.INTER_AREA)
            chosen_full = part_full[best_idx]
            resized = cv2.resize(chosen_full, (tile_side_px, tile_side_px), interpolation=cv2.INTER_AREA)
            mosaic[y1:y1 + tile_side_px, x1:x1 + tile_side_px] = resized
            usage_count[best_idx] += 1
            tile_info.append((r, c, best_idx, tile_small))
            counter += 1
            if progress_bar is not None and counter % update_interval == 0:
                progress_bar.progress(min(counter / total_tiles, 1.0))
    if progress_bar is not None:
        progress_bar.progress(1.0)
    return mosaic, tile_info, usage_count

##########################################################
# 5) STREAMLIT APP
##########################################################
def main():
    st.set_page_config(page_title="Pixify", layout="wide")
    
    # Welcome message (outside expander)
    st.markdown(
        """
        # Welcome to Pixify!
        #### **A picture is worth a thousand words—but what about a picture made of a thousand memories?**
        With Pixify, turn a single cherished photo into a breathtaking mosaic crafted from the moments that matter most.
        Whether it’s family, friends, or a lifetime of adventures, watch your story come to life in a whole new way.
        """
    )
    
    st.image("pictures/Example.png", use_column_width=True)

    # Collapsible 5 Steps Explanation
    with st.expander("Guide: 5 Steps to Create Your Mosaic", expanded=False):
        st.markdown(
            """     
            1. **Upload a Target Image:**  
               Upload exactly one target image in the left sidebar.
            """
        )
        st.image("pictures/Target_example.png", caption="Exemplary target image.", width=300)
        st.markdown(
            """  
            2. **Crop the Target:**  
               Use the cropping sliders in the left sidebar to adjust your target.
            """
        )
        st.image("pictures/Cropping_example.png", caption="Exemplary cropped target image.", width=300)
        st.markdown(
            """  
            3. **Upload Part Images:**  
               Upload multiple part images in the left sidebar. They will be resized to 118×118 pixels. These will be later used for recreating the target image as mosaic.
            """
        )
        st.image("pictures/Parts_example.png", caption="Exemplary part images.", width=700)
        st.markdown(
            """  
            4. **Set Mosaic Settings:**  
               Adjust the tile side (side length of each mosaic tile in centimeters), mosaic resolution (quality or resolution of the final mosaic in DPI), 
               target image scale-up (for increasing the size of the final mosaic image), and alpha blending (to control the mix of mosaic and target image) in the left sidebar.
            """
        )
        st.image("pictures/DPI_example.png", caption="Exemplary resolutions based on DPI values.", width=400)
        st.image("pictures/Alpha_example.png", caption="Exemplary mosaics with different alpha values.", width=800)
        st.markdown(
            """    
            5. **Build and Download:**  
               Click *Build Final Mosaic* in the sidebar to generate your mosaic and then download it as PNG or PDF.
            """
        )
    
    # Initialize session state flags if not already set
    if "raw_mosaic" not in st.session_state:
        st.session_state.raw_mosaic = None
    if "build_trigger" not in st.session_state:
        st.session_state.build_trigger = False
    
    # Sidebar controls with guidance and tooltips
    with st.sidebar:
        st.header("1) Target Image Upload")
        tgt_files = st.file_uploader("Upload 1 Target Image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        st.header("2) Cropping of Target Image")
        top_frac = st.slider("Crop Top (%)", 0, 90, 0, 1, help="Adjust the top crop percentage of the target image.") / 100
        bottom_frac = st.slider("Crop Bottom (%)", 0, 90, 0, 1, help="Adjust the bottom crop percentage of the target image.") / 100
        left_frac = st.slider("Crop Left (%)", 0, 90, 0, 1, help="Adjust the left crop percentage of the target image.") / 100
        right_frac = st.slider("Crop Right (%)", 0, 90, 0, 1, help="Adjust the right crop percentage of the target image.") / 100

        st.header("3) Part Images Upload")
        part_files = st.file_uploader("Upload Part Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        # Conditional CSS for Build Final Mosaic button based on part image upload status
        if part_files and tgt_files:
            button_bg = "#4EA72E"
            button_hover_bg = "#5AC537"
        else:
            button_bg = "#7F7F7F"
            button_hover_bg = "#A6A6A6"
        st.markdown(
            f"""
            <style>
            div.stButton > button {{
                background-color: {button_bg} !important;
                color: white !important;
                font-weight: bold;
                margin: 0 auto;
                display: block;
                border: 1px solid {button_bg} !important;
            }}
            div.stButton > button:hover {{
                background-color: {button_hover_bg} !important;
                color: white !important;
                border: 1px solid {button_hover_bg} !important;
            }}
            </style>
            """, unsafe_allow_html=True
        )

        st.header("4) Mosaic Settings")
        cm_tile = st.slider("Tile side (cm)", 1.0, 5.0, 1.5, 0.5, help="The physical length of each mosaic tile side (cm).")
        mosaic_dpi = st.slider("Mosaic resolution (DPI)", 50, 300, 200, 50, help="The resolution of the final mosaic image.")
        scale_factor = st.slider("Target image scale-up", 1.0, 20.0, 1.0, 0.5, help="Increase the size of the final mosaic image.")
        alpha_val = st.slider("Alpha blending", 0.0, 0.5, 0.4, 0.05, help="Blend between the target and mosaic images.")
        
        # Placeholder for resulting dimensions
        dims_placeholder = st.empty()

        st.header("5) Mosaic Creation")
        if st.button("Build Final Mosaic"):
            st.session_state.build_trigger = True
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center;">
                <a href="https://www.linkedin.com/in/konradschulte/" target="_blank" style="text-decoration: none; margin-right: 24px; color: inherit;">
                    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="22px" style="margin-right:8px; filter: brightness(0.1) invert(1)"/>
                    LinkedIn
                </a>
                <a href="https://github.com/konradschulte" target="_blank" style="text-decoration: none; color: inherit;">
                    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="22px" style="margin-right:8px; filter: brightness(0.1) invert(1)"/>
                    GitHub
                </a>
            </div>
            <div style="text-align: center; margin-top: 0.5rem;">
                <a href="mailto:konrad.schulte3@gmx.de" style="text-decoration: none; color: inherit;">Feedback</a>
            </div>
            <div style="text-align: center; margin-top: 0.5rem;">
                © Konrad Schulte
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Process the target image if available
    target_img = None
    if not tgt_files or len(tgt_files) == 0:
        st.info("**Steps 1 & 2:** Please upload exactly one target image and crop it (if necessary) in the left sidebar.")
    elif len(tgt_files) > 1:
        st.warning("Only one target image is allowed.")
    else:
        target_img = load_single_target(tgt_files[0])
        if target_img is None:
            st.error("Could not read target image.")
        else:
            st.session_state.target_uploaded = True

    if target_img is not None:
        cropped = fractional_crop(target_img, top_frac, bottom_frac, left_frac, right_frac)
        st.subheader("Target")
        st.image(cropped, use_column_width=True)
        # Compute mosaic dimensions based on current settings
        new_w = int(cropped.shape[1] * scale_factor)
        new_h = int(cropped.shape[0] * scale_factor)
        scaled_target = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tile_side_px = round((cm_tile / 2.54) * mosaic_dpi)
        tile_side_px = max(1, tile_side_px)
        squares_w = new_w // tile_side_px
        squares_h = new_h // tile_side_px
        total_sq = squares_w * squares_h

        mosaic_width_px = squares_w * tile_side_px
        mosaic_height_px = squares_h * tile_side_px
        mosaic_width_in = mosaic_width_px / mosaic_dpi
        mosaic_height_in = mosaic_height_px / mosaic_dpi
        mosaic_width_cm = mosaic_width_in * 2.54
        mosaic_height_cm = mosaic_height_in * 2.54

        dims_placeholder.markdown(f"### Resulting Dimensions:\n **Mosaic dimension:** {squares_w} x {squares_h} = {total_sq} squares \n \n **Real-world size:** {mosaic_width_cm:.1f} x {mosaic_height_cm:.1f} cm")
    else:
        scaled_target = None
        squares_w = squares_h = None

    # Load part images
    parts_list = []
    if part_files:
        st.session_state.part_uploaded = True
        loading_placeholder = st.empty()
        loading_placeholder.info("Loading and processing part images...")
        parts_list = load_part_images(part_files)
        loading_placeholder.empty()
        st.write(f"Loaded {len(parts_list)} part images.")
        info_placeholder = st.empty()
        # Display info for Steps 4 & 5 until mosaic build is triggered
        if st.session_state.get("raw_mosaic") is None and tgt_files:
            info_placeholder.info("**Steps 4 & 5:** Adjust mosaic settings and create mosaic in the left sidebar.")
    elif tgt_files:
        st.info("**Steps 2 & 3:** Crop the target image (if necessary) and upload part images in the left sidebar.")

    # Build mosaic only when triggered
    if st.session_state.get("build_trigger", False):
        if target_img is None:
            st.warning("No target image provided. Please upload one before building mosaic (in the left sidebar).")
            st.session_state.build_trigger = False
        elif len(parts_list) == 0:
            st.warning("No part images found. Please upload some (in the left sidebar).")
            st.session_state.build_trigger = False
        elif squares_w is None or squares_h is None or squares_w < 1 or squares_h < 1:
            st.warning("No squares fit. Adjust tile or scale factor.")
            st.session_state.build_trigger = False
        else:
            info_placeholder.empty()
            # Store current mosaic settings to freeze the mosaic
            st.session_state.mosaic_settings = {
                "cropped": cropped,
                "scaled_target": scaled_target,
                "tile_side_px": tile_side_px,
                "squares_w": squares_w,
                "squares_h": squares_h,
                "cm_tile": cm_tile,
                "mosaic_dpi": mosaic_dpi,
                "scale_factor": scale_factor,
                "alpha_val": alpha_val,
            }
            building_placeholder = st.empty()
            building_placeholder.info("Building mosaic. Please wait...")
            mosaic_progress = st.progress(0)
            cluster_centers, cluster_assignments, part_analysis, part_full = naive_kmeans_average_color(parts_list, max_iters=10)
            raw_mosaic, tile_info, usage_count = build_final_mosaic_kmeans(
                scaled_target,
                cluster_centers,
                cluster_assignments,
                part_analysis,
                part_full,
                tile_side_px,
                progress_bar=mosaic_progress
            )
            st.session_state.raw_mosaic = raw_mosaic
            st.session_state.build_trigger = False
            used_parts = [pid for pid, cnt in usage_count.items() if cnt > 0]
            st.write(f"**Used:** {len(used_parts)}/{len(parts_list)} part images ({len(used_parts) / len(parts_list) * 100:.1f}%)")
            building_placeholder.empty()

    # If a mosaic has been built, show preview and prepare downloads.
    if st.session_state.raw_mosaic is not None and target_img is not None:
        # Use stored mosaic settings if available to prevent automatic updates
        if "mosaic_settings" in st.session_state:
            settings = st.session_state.mosaic_settings
            scaled_target = settings.get("scaled_target", scaled_target)
            cropped = settings.get("cropped", cropped)
            tile_side_px = settings.get("tile_side_px", tile_side_px)
            squares_w = settings.get("squares_w", squares_w)
            squares_h = settings.get("squares_h", squares_h)
            alpha_val_stored = settings.get("alpha_val", alpha_val)
            cm_tile = settings.get("cm_tile", cm_tile)
            mosaic_dpi = settings.get("mosaic_dpi", mosaic_dpi)
        else:
            alpha_val_stored = alpha_val
        st.subheader("Mosaic Preview")
        raw_mosaic = st.session_state.raw_mosaic
        final_out = alpha_blend(scaled_target, raw_mosaic, alpha_val_stored)
        preview = cv2.resize(final_out, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_AREA)
        image_comparison(
            Image.fromarray(cropped),
            Image.fromarray(preview),
            label1="Target",
            label2="Mosaic",
            width=800
        )
        download_placeholder = st.empty()
        download_placeholder.info("Preparing downloads. Please wait...")
        final_img = Image.fromarray(final_out)
        
        # Save PNG using optimize and compress_level=1.
        png_buf = BytesIO()
        final_img.save(png_buf, format="PNG",compress_level=1)
        dl_name_png = f"{squares_h}x{squares_w}_{cm_tile}cm_{mosaic_dpi}dpi_{alpha_val_stored}a.png"
        st.download_button("Download Mosaic (PNG)",
                           data=png_buf.getvalue(),
                           file_name=dl_name_png,
                           mime="image/png")
        
        # Save PDF (usually smaller)
        pdf_buf = BytesIO()
        final_img.save(pdf_buf, format="PDF")
        pdf_buf.seek(0)
        dl_name_pdf = f"{squares_h}x{squares_w}_{cm_tile}cm_{mosaic_dpi}dpi_{alpha_val_stored}a.pdf"
        st.download_button("Download Mosaic (PDF)",
                           data=pdf_buf.getvalue(),
                           file_name=dl_name_pdf,
                           mime="application/pdf")
        download_placeholder.empty()

if __name__ == "__main__":
    main()