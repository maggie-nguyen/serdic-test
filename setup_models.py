from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

models = {
    "models/ppe_model1.pt": ("Hansung-Cho/yolov8-ppe-detection",              "best.pt"),
    "models/ppe_model2.pt": ("Tanishjain9/yolov8n-ppe-detection-6classes",    "best.pt"),
}

for dest, (repo, filename) in models.items():
    p = Path(dest)
    if p.exists():
        print(f"Already exists: {dest}")
        continue
    p.parent.mkdir(exist_ok=True)
    print(f"Downloading {repo} ...")
    ckpt = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(ckpt, p)
    print(f"  Saved to {dest}")

print("\nDone. Test with:")
print("  python main.py --ppe-model models/ppe_model1.pt   # Hansung-Cho  (Helmet/Vest/Mask)")
print("  python main.py --ppe-model models/ppe_model2.pt   # Tanishjain9  (Helmet/Vest/Mask/Gloves)")
