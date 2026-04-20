from huggingface_hub import hf_hub_download
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

# Primary PPE model — Hansung-Cho YOLOv8n (Best proxy performance for Helmet + Vest)
download_hf("Hansung-Cho/yolov8-ppe-detection", "best.pt", "models/ppe_model1.pt")

print("\nModels ready. Run inference:")
print("  python main.py --video videos/GUNSAN_cam01_20251222_140441.mp4")
