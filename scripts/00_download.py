"""
Stage 0: download the 16 CSV files and the dataset card from Hugging Face,
pinned to one exact revision so the data can never silently change.

Output: data/raw/<bank>_reviews.csv (16 files), data/raw_README.md (dataset card, Apache-2.0)
Usage : python3 scripts/00_download.py   (run from the project folder)
"""
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/Federal-University-Lokoja/Bank-Review"
REVISION = "27d2b6e7d541a0b713dc8eed56b0daf27476d2a7"
FILES = [
    "access_reviews.csv", "ecoBank_reviews.csv", "fbn_reviews.csv", "fcmb_reviews.csv",
    "fidelityBank_reviews.csv", "gtb_reviews.csv", "jaizBank_reviews.csv", "keyStoneBank_reviews.csv",
    "polarisBank_reviews.csv", "stanbicIbtc_reviews.csv", "sterlingBank_reviews.csv", "uba_reviews.csv",
    "unionBank_reviews.csv", "unityBank_reviews.csv", "wemaBank_reviews.csv", "zenithBank_reviews.csv",
]

raw = Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)
for name in FILES:
    target = raw / name
    if target.exists():
        print(f"already have {name}")
        continue
    urllib.request.urlretrieve(f"{REPO}/resolve/{REVISION}/{name}", target)
    print(f"downloaded {name}  ({target.stat().st_size:,} bytes)")
urllib.request.urlretrieve(f"{REPO}/resolve/{REVISION}/README.md", "data/raw_README.md")
print("downloaded dataset card -> data/raw_README.md")
