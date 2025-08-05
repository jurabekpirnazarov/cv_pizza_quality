import cv2, yaml, os
from PIL import Image
from ultralytics import YOLO


with open("config.yaml", "r") as fh:
    C = yaml.safe_load(fh)

VIDEO_PATH = "/Users/jawlon/Downloads/20250605114550304_L17050358_narezka_11_video.mov"
CLF_PATH = C.get("cut_classifier", "pizza_cut_classifier_model.pt")
SAVE_DIR = C.get("save_dir", "output")
os.makedirs(SAVE_DIR, exist_ok=True)

yolo_model = YOLO('yolov10n.pt')
clf_model = YOLO(CLF_PATH)

# === 🍕 Find Pizza Class ID ===
pizza_id = next((i for i, name in yolo_model.names.items() if name.lower() == 'pizza'), None)
if pizza_id is None:
    raise ValueError("❌ Pizza class not found in YOLO model. Please check the model and class names.")


cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
saved_best_preds = {}  # 🔑 track_id: (confidence, cropped_img, save_path)
FRAME_STEP = C.get("frame_step", 2)  # Default to 1 if not specified in config

print("🚀 Video processing started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_STEP == 0:
        results = yolo_model.track(
            source=frame,
            persist=True,
            classes=[pizza_id],
            conf=C.get("yolo_conf", 0.5),  # Default confidence threshold
            iou=C.get("yolo_iou", 0.5),  # Default IoU threshold
            verbose=False,
            tracker=C.get("tracker_cfg", "bytetrack.yaml")  # Default tracker config
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None
                conf = float(box.conf[0])

                if cls != pizza_id or conf < C.get("box_conf_filter", 0.4) or track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1 < C.get("min_box_size", 20) or y2 - y1 < C.get("min_box_size", 20)):
                    continue

                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                result = clf_model(pil_img, verbose=False)[0]
                pred_class = result.names[result.probs.top1]
                confidence = result.probs.top1conf

                if pred_class == 'cut':
                    if (track_id not in saved_best_preds) or (confidence > saved_best_preds[track_id][0]):
                        save_path = os.path.join(SAVE_DIR, f"pizzaID_{track_id}_frame{frame_count}_cut.jpg")
                        saved_best_preds[track_id] = (confidence, cropped.copy(), save_path)

    frame_count += 1

cap.release()

# 📸 Save the best predictions
saved_count = 0
for track_id, (conf, cropped_img, save_path) in saved_best_preds.items():
    cv2.imwrite(save_path, cropped_img)
    print(f"Pizza ID {track_id} ✅ CUT ({conf:.2%}) ➤ Saved: {save_path}")
    saved_count += 1

# === 🍕 Video Processing Completed ===
print(f"🎥 Video processing completed successfully! {saved_count} images saved in '{SAVE_DIR}' directory.")
