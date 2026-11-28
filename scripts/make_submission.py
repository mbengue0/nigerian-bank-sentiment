"""
Packaging (not part of the experiment): builds the course submission zip.

  project-cs4714-Mouhamadou-Mbengue.zip
    report.pdf     - final report
    dataset.csv    - every review used: train / val / test / unseen_zenith, with a "split" column
    code/          - the project folder: scripts, README, requirements, fixed data splits, results

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
dataset = pd.concat(frames, ignore_index=True)
dataset = dataset.rename(columns={"content": "text", "strat_bank": "bank_group"})
dataset[["split", "text", "label", "banks", "bank_group", "years", "scores", "text_hash"]] \
    .to_csv(OUT / "dataset.csv", index=False)
print(f"dataset.csv: {len(dataset):,} rows")
print(dataset["split"].value_counts().to_string())

# files that go under code/ (everything tracked by git except the reports themselves)
code_files = [p for p in (
    list(Path("scripts").glob("*.py"))
    + list(Path("data/final").glob("*.csv"))
    + [Path("data/raw_README.md"), Path("README.md"), Path("requirements.txt"), Path("ASSISTANCE_LOG.md"),
       Path(".gitignore")]
    + [p for p in Path("outputs").rglob("*") if p.is_file()]
) if p.exists()]

zip_path = OUT / f"{NAME}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("reports/report.pdf", f"{NAME}/report.pdf")
    z.write(OUT / "dataset.csv", f"{NAME}/dataset.csv")
    for p in sorted(code_files):
        z.write(p, f"{NAME}/code/{p.as_posix()}")
print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(code_files) + 2} files)")
