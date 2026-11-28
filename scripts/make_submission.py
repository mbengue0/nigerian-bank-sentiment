"""
Packaging (not part of the experiment): builds the course submission zip.

  project-cs4714-Mouhamadou-Mbengue.zip
    report.pdf   - final report
    code.py      - the whole project in one file; run:  python code.py
    dataset.csv  - every review used (train / val / test / unseen_zenith), with a "split" column
                   and the hand-read categories of the 150 sampled test errors

Usage: python3 scripts/make_submission.py   (run from the project folder)
"""
import zipfile
from pathlib import Path

import pandas as pd

NAME = "project-cs4714-Mouhamadou-Mbengue"
OUT = Path("submission")
OUT.mkdir(exist_ok=True)

parts = [("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv"), ("unseen_zenith", "unseen_zenithBank.csv")]
frames = []
for split, file in parts:
    df = pd.read_csv(f"data/final/{file}", keep_default_na=False)
    df.insert(0, "split", split)
    frames.append(df)
dataset = pd.concat(frames, ignore_index=True).rename(columns={"content": "text", "strat_bank": "bank_group"})
manual = pd.read_csv("outputs/manual_error_labels.csv", keep_default_na=False)
dataset["manual_error_category"] = dataset["text_hash"].map(dict(zip(manual.text_hash, manual.manual_category))).fillna("")
cols = ["split", "text", "label", "banks", "bank_group", "years", "scores", "text_hash", "manual_error_category"]
dataset[cols].to_csv(OUT / "dataset.csv", index=False)
print(f"dataset.csv: {len(dataset):,} rows")

zip_path = OUT / f"{NAME}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("reports/report.pdf", f"{NAME}/report.pdf")
    z.write("code.py", f"{NAME}/code.py")
    z.write(OUT / "dataset.csv", f"{NAME}/dataset.csv")
print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
