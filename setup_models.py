from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pathlib import Path
import shutil

def download_hf(repo, filename, dest):
    p = Path(dest)
    if p.exists():
        print(f"Already exists: {dest}")
        return
    p.parent.mkdir(exist_ok=True)
    print(f"Downloading {repo} ...")
    ckpt = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(ckpt, p)
    print(f"  Saved to {dest}")

def download_hf_url(url, dest):
    p = Path(dest)
    if p.exists():
        print(f"Already exists: {dest}")
        return
    p.parent.mkdir(exist_ok=True)
    print(f"Downloading {url} ...")
    model = YOLO(url)
    model.save(str(p))
    print(f"  Saved to {dest}")

download_hf("Hansung-Cho/yolov8-ppe-detection",           "best.pt", "models/ppe_model1.pt")
download_hf("Tanishjain9/yolov8n-ppe-detection-6classes", "best.pt", "models/ppe_model2.pt")
download_hf("Hexmon/vyra-yolo-ppe-detection",              "best.pt", "models/ppe_model3.pt")

print("\nTest with:")
print("  python main.py --ppe-model models/ppe_model1.pt  # Hansung  (Helmet/Vest/Mask)")
print("  python main.py --ppe-model models/ppe_model2.pt  # Tanishjain (+ Gloves, YOLOv8n)")
print("  python main.py --ppe-model models/ppe_model3.pt  # Hexmon   (+ Gloves, YOLOv8m)")
