import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import yaml
import os
import tempfile
import numpy as np

# === 📄 Load config.yaml
with open("config.yaml", "r") as fh:
    C = yaml.safe_load(fh)

CLF_PATH = C.get("cut_classifier", "pizza_cut_classifier_model.pt")
FRAME_STEP = C.get("frame_step", 2)
YOLO_CONF = C.get("yolo_conf", 0.5)
YOLO_IOU = C.get("yolo_iou", 0.5)
TRACKER_CFG = C.get("tracker_cfg", "bytetrack.yaml")
BOX_CONF_FILTER = C.get("box_conf_filter", 0.4)
MIN_BOX_SIZE = C.get("min_box_size", 20)

# === 📦 Load models
yolo_model = YOLO('yolov10n.pt')
clf_model = YOLO(CLF_PATH)

# === 🍕 Find pizza class id
pizza_id = next((i for i, name in yolo_model.names.items() if name.lower() == 'pizza'), None)
if pizza_id is None:
    st.error("❌ Pizza class not found in YOLO model.")
    st.stop()

# === 🎛 Streamlit UI
st.title("🍕 Pizza QC Video Stream App")
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    temp_video_path = tfile.name

    run = st.checkbox("▶ Start video processing")

    frame_placeholder = st.empty()
    label_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_STEP == 0:
                results = yolo_model.track(
                    source=frame,
                    persist=True,
                    classes=[pizza_id],
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    verbose=False,
                    tracker=TRACKER_CFG
                )

                label_status = ""
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        cls = int(box.cls[0])
                        track_id = int(box.id[0]) if box.id is not None else None
                        conf = float(box.conf[0])

                        if cls != pizza_id or conf < BOX_CONF_FILTER or track_id is None:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        if (x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE):
                            continue

                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size == 0:
                            continue

                        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        clf_result = clf_model(pil_img, verbose=False)[0]
                        pred_class = clf_result.names[clf_result.probs.top1]
                        confidence = clf_result.probs.top1conf

                        label = f"{pred_class.upper()} ({confidence:.0%})"
                        color = (0, 255, 0) if pred_class == 'cut' else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        label_status = f"Track ID: {track_id}, Status: {pred_class.upper()}, Confidence: {confidence:.2%}"

            # Show frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB")
            if label_status:
                label_placeholder.info(label_status)

            frame_count += 1
 
        cap.release()
        st.success("✅ Video processing completed.")
