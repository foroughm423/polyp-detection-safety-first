# Polyp Detection - Streamlit Web Application
#
# Model: YOLOv8m-seg (aug + negative samples), hosted on Hugging Face Hub
# Model weights (best.pt) are excluded from git due to file size.
# results/metrics/final_summary.json is committed directly to the repo.

import json
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

import subprocess
import imageio_ffmpeg

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Polyp Detection",
    page_icon="🔬",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

load_dotenv()
MODEL_PATH = hf_hub_download(
    repo_id="foroughm423/polyp-detection-safety-first",
    filename="best.pt",
    token=os.environ.get("HF_TOKEN"),
)

CONF_DEFAULT = 0.30
RESULTS_PATH = "results/metrics/final_summary.json"
FIGURES_PATH = "results/figures"



# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path):
    return YOLO(path)


@st.cache_data
def load_summary(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# ── Helper functions ──────────────────────────────────────────────────────────

def run_inference(model, image_array, conf):
    """Run detection on a numpy RGB image and return the result."""
    result = model.predict(image_array, conf=conf, verbose=False)[0]
    return result


def draw_result(result):
    """
    Draw bounding boxes and segmentation masks on the image.
    result.plot() always returns BGR (per Ultralytics docs),
    so we convert to RGB for correct display in Streamlit.
    """
    annotated = result.plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def reencode_to_browser_mp4(input_path):
        """
        Re-encode any video file to H.264/mp4 so it plays in browser
        <video> tags. Works regardless of the input's original codec
        or container (mp4, avi, mov, mkv, etc.) because ffmpeg itself
        handles the decoding - we only control the *output* codec.

        Returns the path to the new, browser-compatible file.
        """
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        output_path = input_path.rsplit(".", 1)[0] + "_h264.mp4"

        cmd = [
            ffmpeg_exe,
            "-y",                  # overwrite output if it exists
            "-i", input_path,
            "-c:v", "libx264",     # H.264 video codec - browser standard
            "-pix_fmt", "yuv420p", # required for compatibility with
                                    # some browsers (notably Safari/iOS)
            "-c:a", "aac",         # re-encode audio too, in case the
                                    # source has an unusual audio codec
            "-movflags", "+faststart",  # lets browser start playback
                                          # before the full file downloads
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # ffmpeg failed - surface the error instead of silently
            # falling through with a broken video
            raise RuntimeError(f"ffmpeg re-encode failed:\n{result.stderr}")

        return output_path

def result_to_dict(result):
    """Extract detection info as a list of dicts for display."""
    detections = []
    for i, box in enumerate(result.boxes):
        entry = {
            "polyp_id":   i + 1,
            "confidence": float(box.conf[0]),
            "x1": int(box.xyxy[0][0]),
            "y1": int(box.xyxy[0][1]),
            "x2": int(box.xyxy[0][2]),
            "y2": int(box.xyxy[0][3]),
        }
        detections.append(entry)
    return detections

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=CONF_DEFAULT,
        step=0.05,
        help="Lower = more detections, higher = fewer but more certain",
    )

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("YOLOv8m-seg")
    st.markdown("Aug + Negative Samples")

    st.markdown("---")
    st.markdown("**Key Results**")
    summary = load_summary(RESULTS_PATH)
    if summary:
        st.metric("Kvasir-SEG Recall",
                  f"{summary['results']['kvasir_test']['recall']:.1%}")
        st.metric("CVC-ClinicDB Recall",
                  f"{summary['results']['cvc_cross_dataset']['recall']:.1%}")
        st.metric("Inference Speed",
                  f"{summary['inference_speed']['fps']:.0f} FPS")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Tracking", "Model Results"])

# ── Tab 1: Image Detection ────────────────────────────────────────────────────

with tab1:
    st.header("Polyp Detection in Colonoscopy Images")
    st.markdown(
        "Upload a colonoscopy image to detect polyps. "
        "The model returns both a bounding box and a pixel-level segmentation mask."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_file is not None:
        image       = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(image, use_container_width=True)

        model = load_model(MODEL_PATH)

        start      = time.time()
        result     = run_inference(model, image, conf_threshold)
        elapsed_ms = (time.time() - start) * 1000

        annotated  = draw_result(result)
        detections = result_to_dict(result)

        with col2:
            st.subheader(f"Detection ({len(detections)} polyp(s) found)")
            st.image(annotated, use_container_width=True)

        st.markdown(f"Inference time: **{elapsed_ms:.1f} ms**")

        if detections:
            st.subheader("Detection Details")
            for d in detections:
                st.markdown(
                    f"Polyp {d['polyp_id']}: "
                    f"confidence **{d['confidence']:.2f}** | "
                    f"box [{d['x1']}, {d['y1']}, {d['x2']}, {d['y2']}]"
                )
        else:
            st.info(
                "No polyps detected at this confidence threshold. "
                "Try lowering the threshold in the sidebar."
            )

# ── Tab 2: Video Tracking ─────────────────────────────────────────────────────

with tab2:
    st.header("Polyp Tracking in Colonoscopy Video")
    st.markdown(
        "Upload a short colonoscopy video clip. "
        "The model detects polyps in each frame and ByteTrack assigns "
        "consistent IDs across frames."
    )

    uploaded_video = st.file_uploader(
        "Choose a video (mp4, max ~50MB recommended)",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            raw_tmp_path = tmp.name

        # Normalize whatever format/codec the user uploaded into a
        # browser-playable H.264 file BEFORE doing anything else with it
        with st.spinner("Preparing video..."):
            tmp_path = reencode_to_browser_mp4(raw_tmp_path)

        model = load_model(MODEL_PATH)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            st.video(tmp_path)   

        with st.spinner("Running detection + tracking..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                track_results = model.track(
                    source=tmp_path,
                    conf=conf_threshold,
                    tracker="bytetrack.yaml",
                    persist=True,
                    imgsz=640,
                    save=True,
                    project=tmp_dir,
                    name="tracked",
                    exist_ok=True,
                    verbose=False,
                )

                out_video_path = os.path.join(
                    tmp_dir, "tracked", os.path.basename(tmp_path)
                )

                # Ultralytics sometimes saves with a different container/codec
                # than the input - if the exact name isn't found, grab whatever
                # video file landed in that output folder instead.
                if not os.path.exists(out_video_path):
                    tracked_dir = os.path.join(tmp_dir, "tracked")
                    if os.path.isdir(tracked_dir):
                        candidates = [
                            f for f in os.listdir(tracked_dir)
                            if f.lower().endswith((".mp4", ".avi", ".mov"))
                        ]
                        if candidates:
                            out_video_path = os.path.join(tracked_dir, candidates[0])

                frame_counts = [len(r.boxes) for r in track_results]
                all_ids      = set()
                for r in track_results:
                    if r.boxes.id is not None:
                        all_ids.update(
                            r.boxes.id.cpu().numpy().astype(int).tolist()
                        )

                # Read the output video bytes BEFORE the TemporaryDirectory
                # context closes and deletes the file - st.video() and the
                # download button both need the bytes to persist afterward.
                output_video_bytes = None
                if out_video_path and os.path.exists(out_video_path):
                    try:
                        out_video_path = reencode_to_browser_mp4(out_video_path)
                    except RuntimeError:
                            pass  # if re-encode fails, fall back to the original file
                                  # rather than crashing the whole app
                                  
                    with open(out_video_path, "rb") as f:
                        output_video_bytes = f.read()

        with col2:
            st.subheader("Detection + Tracking")
            if output_video_bytes:
                st.video(output_video_bytes)
            else:
                st.warning("Tracked video file not found - showing stats only.")

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Frames processed",    len(frame_counts))
        col_b.metric("Frames with detection",
                     sum(1 for n in frame_counts if n > 0))
        col_c.metric("Unique track IDs",     len(all_ids))

        if output_video_bytes:
            st.download_button(
                "Download tracked video",
                data=output_video_bytes,
                file_name="polyp_tracked.mp4",
                mime="video/mp4",
            )

        os.unlink(tmp_path)

# ── Tab 3: Model Results ──────────────────────────────────────────────────────

with tab3:
    st.header("Model Performance Summary")

    st.markdown(
        "This project trained and evaluated four configurations. "
        "The table below shows recall on two test sets: "
        "Kvasir-SEG (same distribution as training) and "
        "CVC-ClinicDB (never seen during training — cross-dataset generalization)."
    )

    data = {
        "Approach": [
            "Baseline (conf=0.50)",
            "Threshold Tuning (conf=0.30)",
            "Higher Resolution (imgsz=960)",
            "Aug + Negatives — Final",
        ],
        "Kvasir Recall": ["0.867", "0.867", "0.835", "**0.898**"],
        "CVC Recall":    ["0.752", "0.752", "0.751", "**0.814**"],
        "Notes": [
            "Standard YOLOv8m-seg",
            "Same model, lower threshold",
            "Retrain at 960px — did not improve",
            "Stronger augmentation + 140 negative frames",
        ],
    }

    st.table(data)

    # Figures from notebook 05
    st.markdown("---")
    st.subheader("Visual Comparison")

    fig_comparison  = os.path.join(FIGURES_PATH, "final_comparison.png")
    fig_progression = os.path.join(FIGURES_PATH, "recall_progression.png")
    fig_error       = os.path.join(FIGURES_PATH, "error_analysis_summary.png")

    if os.path.exists(fig_comparison) and os.path.exists(fig_progression):
        col1, col2 = st.columns(2)
        col1.image(fig_comparison,  caption="Recall Comparison",     use_container_width=True)
        col2.image(fig_progression, caption="Improvement Journey",   use_container_width=True)

    if os.path.exists(fig_error):
        st.image(fig_error, caption="Why Polyps Were Missed: Size Analysis",
                 use_container_width=True)

    st.markdown("---")
    st.markdown("### Engineering Journey")
    st.markdown(
        "1. **Root cause analysis:** missed polyps were 59% smaller (median) "
        "than detected ones — a small-object detection problem.\n"
        "2. **Hypothesis 1:** higher resolution (imgsz=960) → no improvement "
        "(smaller batch hurt training stability more than resolution helped).\n"
        "3. **Hypothesis 2:** stronger augmentation + negative samples "
        "(per YOLO-LAN 2025) → **+6.2% CVC recall**.\n\n"
        "Cross-dataset recall improved from 75.2% to 81.4%, "
        "with inference speed unchanged at ~45 FPS."
    )

    st.markdown("---")
    st.markdown(
        "**Dataset:** Kvasir-SEG + CVC-ClinicDB  \n"
        "**Model:** YOLOv8m-seg (Ultralytics)  \n"
        "**Tracking:** ByteTrack  \n"
        "**Framework:** PyTorch  "
    )