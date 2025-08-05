import cv2, os, yaml
from PIL import Image
from ultralytics import YOLO

with open("config.yaml", "r") as fh:
    C = yaml.safe_load(fh)

os.makedirs(C["save_dir"], exist_ok=True)

yolo_model = YOLO(C["yolo_model"]).to(C["device"])
clf_model = YOLO(C["cut_classifier"]).to(C["device"])

pizza_id = next((i for i, n in yolo_model.names.items() if n.lower() == "pizza"), None)
if pizza_id is None:
    raise RuntimeError("Pizza class not found in detector model.")

# CAP_FFMPEG ensures OpenCV hands the stream to FFmpeg (works on most builds)
cap = cv2.VideoCapture(C["video_source"], cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Could not open RTSP stream. Check URL/credentials.")

print("🚀 RTSP stream opened …")

frame_cnt, saved_best = 0, {}    # track_id ➜ (best_conf, img, path)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    if frame_cnt % C["frame_step"] == 0:
        results = yolo_model.track(
            source   = frame,
            persist  = True,
            classes  = [pizza_id],
            conf     = C["yolo_conf"],
            iou      = C["yolo_iou"],
            tracker  = C["tracker_cfg"],
            verbose  = False
        )

        for r in results:
            for box in (r.boxes or []):
                cls   = int(box.cls[0]);   conf = float(box.conf[0])
                tid   = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # quick filters
                if (cls != pizza_id or conf < C["box_conf_filter"] or tid is None):
                    continue
                if (x2 - x1 < C["min_box_size"] or y2 - y1 < C["min_box_size"]):
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # classifier
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                out = clf_model(pil, verbose=False)[0]
                label       = out.names[out.probs.top1]       # 'cut' / 'uncut'
                cut_conf    = out.probs.top1conf

                # keep best "cut" snapshot per pizza
                if label == "cut" and (
                        tid not in saved_best or cut_conf > saved_best[tid][0]):
                    path = os.path.join(C["save_dir"],
                                        f"pizza_{tid}_frame{frame_cnt}.jpg")
                    saved_best[tid] = (cut_conf, crop.copy(), path)

    frame_cnt += 1

cap.release()

# ── 7. dump crops to disk ────────────────────────────────────────
for tid, (conf, img, path) in saved_best.items():
    cv2.imwrite(path, img)
    print(f"Pizza {tid} ✅ CUT {conf:.1%} → {path}")

print("🎬 Stream processing ended.")
