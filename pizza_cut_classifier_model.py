import os
import random
import shutil
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from ultralytics import YOLO


class PizzaClassifierTrainer:
    def __init__(self,
                 cut_dir='data_set/cut',
                 uncut_dir='data_set/uncut',
                 aug_cut_dir='aug_data/cut',
                 aug_uncut_dir='aug_data/uncut',
                 dataset_root='datasets',
                 model_arch='yolov8n-cls.pt',
                 epochs=2,
                 imgsz=224,
                 batch=32):
        self.cut_dir = cut_dir
        self.uncut_dir = uncut_dir
        self.aug_cut_dir = aug_cut_dir
        self.aug_uncut_dir = aug_uncut_dir
        self.dataset_root = dataset_root
        self.model_arch = model_arch
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch

        os.makedirs(self.aug_cut_dir, exist_ok=True)
        os.makedirs(self.aug_uncut_dir, exist_ok=True)

        self.augment_transform = T.Compose([
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def augment_images(self, input_dir, output_dir, times=2):
        for img_name in tqdm(os.listdir(input_dir), desc=f'Augmenting {input_dir}'):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(input_dir, img_name)
            image = Image.open(path).convert('RGB')
            base_name = os.path.splitext(img_name)[0]
            image.save(os.path.join(output_dir, f"{base_name}_orig.jpg"))
            for i in range(times):
                transformed = self.augment_transform(image)
                transformed.save(os.path.join(output_dir, f"{base_name}_aug{i}.jpg"))

    def prepare_dataset(self, split_ratio=0.8):
        for label, src_dir in [('cut', self.aug_cut_dir), ('uncut', self.aug_uncut_dir)]:
            imgs = os.listdir(src_dir)
            random.shuffle(imgs)
            split = int(len(imgs) * split_ratio)
            train_imgs = imgs[:split]
            val_imgs = imgs[split:]

            for subset, files in [('train', train_imgs), ('val', val_imgs)]:
                out_dir = os.path.join(self.dataset_root, subset, label)
                os.makedirs(out_dir, exist_ok=True)
                for f in files:
                    shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))

    def train_model(self):
        print("📚 Training started...")
        model = YOLO(self.model_arch)
        model.train(
            data=self.dataset_root,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch
        )
        print("✅ Training finished!")

    def export_model(self, trained_model_path='runs/classify/train/weights/best.pt', export_format='onnx'):
        model = YOLO(trained_model_path)
        model.export(format=export_format)
        print(f"✅ Model exported as {export_format}.")


def classify_image(image_path, model_path='runs/classify/train/weights/best.pt'):
    model = YOLO(model_path)
    result = model(image_path)
    pred_class = result[0].names[result[0].probs.top1]
    confidence = result[0].probs[result[0].probs.top1].item()
    print(f"Image: {image_path}")
    print(f"Predicted class: {pred_class} ({confidence * 100:.2f}%)")
    return pred_class, confidence


if __name__ == "__main__":
    trainer = PizzaClassifierTrainer()

    # Step 1: Augment images
    trainer.augment_images(trainer.cut_dir, trainer.aug_cut_dir, times=2)
    trainer.augment_images(trainer.uncut_dir, trainer.aug_uncut_dir, times=1)

    # Step 2: Prepare dataset
    trainer.prepare_dataset()

    # Step 3: Train the model
    trainer.train_model()

    # Step 4: Export to ONNX
    trainer.export_model()
