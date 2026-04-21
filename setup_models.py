from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

def download_hf(repo: str, filename: str, dest: str):
    p = Path(dest)
    if p.exists():
        print(f"Model already exists: {dest}")
        return
    p.parent.mkdir(exist_ok=True)
    print(f"Downloading {repo} to {dest} ...")
    ckpt = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(ckpt, p)
    print("Download complete.")

def main():
    # Baseline model selected via proxy evaluation
    download_hf("Hansung-Cho/yolov8-ppe-detection", "best.pt", "models/ppe_model1.pt")
    
if __name__ == "__main__":
    main()
