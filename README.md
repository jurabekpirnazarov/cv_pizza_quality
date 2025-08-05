 
# 🍕 Pizza Quality Control – Cut vs Uncut Classification

This project uses **YOLOv8 classification** and **YOLOv10 detection** to:
- Train a pizza classifier (`cut` vs `uncut`)
- Analyze a video and **extract only cut pizzas**

---

## 📦 Project Structure

.
├── data_set/ # Raw images (cut & uncut)
├── aug_data/ # Augmented images
├── datasets/ # Train/val split for YOLO, it will shows while trainig
├── output/ # Saved cut pizzas from video, it will shows while trainig
├── runs/ # YOLO training outputs
├── training_yolo_model_binary.py # Contains training pipeline
├── video_predictor.py # Detect & classify from video
├── requirements.txt
└── README.md

Always show details


---

## 🔧 Setup

Install required packages:

```bash
pip install -r requirements.txt
Or manually:

Always show details

pip install opencv-python pillow torchvision ultralytics tqdm
🧠 Training Classifier

All training logic is inside PizzaClassifierTrainer class.

Always show details

from classifier import PizzaClassifierTrainer

trainer = PizzaClassifierTrainer()

trainer.augment_images(trainer.cut_dir, trainer.aug_cut_dir, times=2)
trainer.augment_images(trainer.uncut_dir, trainer.aug_uncut_dir, times=1)

trainer.prepare_dataset()
trainer.train_model()
trainer.export_model()
📁 Outputs a trained model at:
runs/classify/train/weights/best.pt

🧪 Classify a Single Image

Always show details

from classifier import classify_image

classify_image("some_pizza.jpg")
Outputs:

Always show details

Image: some_pizza.jpg
Predicted class: cut (97.41%)
🎥 Run on Video

Script: video_predictor.py

Always show details

python video_predictor.py
This:

Detects pizzas in a video
Tracks them
Crops each new pizza
Classifies it as cut or uncut
Saves cut pizzas to /cutted
📝 Set video path and model path at top of script:

Always show details

VIDEO_PATH = "path/to/video.mov"
CLF_PATH = "runs/classify/train/weights/best.pt"
🧾 Notes

Detection: YOLOv10 with tracker (ByteTrack)
Classification: YOLOv8 classification model
Images resized to 224x224 before classification
Only first occurrence of each pizza ID is saved
✅ Output Example

For streamlit running on terminal:

streamlit run stream_video_predictior.py


Always show details

[104] Pizza ID 7 ✅ CUT (95.23%) ➤ Saqlandi: cutted/pizzaID_7_frame104_cut.jpg
✅ Tugadi. Jami 37 ta kesilgan pizza saqlandi: cutted/
📜 License

MIT License.

✨ Author

Developed by ML/AI team for Belissimo AI QC Task 🍕