"""
Stage 1: reproduce the dataset audit for the Nigerian bank-review project.

Input : the 16 CSV files from Federal-University-Lokoja/Bank-Review
        (pinned revision 27d2b6e7d541a0b713dc8eed56b0daf27476d2a7), placed in data/raw/
Output: outputs/audit_report.txt       - every count, next to the earlier recorded count
        outputs/removal_log.csv        - how many rows each step removed, and why
        data/interim/candidates.csv.gz - one row per distinct normalized text (the modeling pool)

Usage : python3 scripts/01_audit.py  (run from the nlp_capstone folder)

Label rule: rating 1-2 -> 0 (negative), rating 4-5 -> 1 (positive), rating 3 excluded.
These are weak labels derived from star ratings, not human sentiment judgments.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs")
INTERIM_DIR = Path("data/interim")
OUT_DIR.mkdir(exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Counts recorded in the 17 November inspection (from the project handoff).
EXPECTED_FILE_ROWS = {
    "access_reviews.csv": 50204, "ecoBank_reviews.csv": 17442, "fbn_reviews.csv": 33607,
    "fcmb_reviews.csv": 12006, "fidelityBank_reviews.csv": 3703, "gtb_reviews.csv": 14147,
    "jaizBank_reviews.csv": 1197, "keyStoneBank_reviews.csv": 3978, "polarisBank_reviews.csv": 6100,
    "stanbicIbtc_reviews.csv": 45094, "sterlingBank_reviews.csv": 7974, "uba_reviews.csv": 87609,
    "unionBank_reviews.csv": 9293, "unityBank_reviews.csv": 927, "wemaBank_reviews.csv": 20175,
    "zenithBank_reviews.csv": 20959,
}
EXPECTED = {
    "raw_rows": 334415,
    "empty_text_rows_all_ratings": 96,
    "nonempty_binary_rows": 312104,
    "unique_texts_before_conflicts": 197676,
    "conflicting_groups": 1359,
    "candidates_after_conflicts": 196317,
    "negative_candidates": 93596,
    "positive_candidates": 102721,
    "candidates_3plus_tokens (exploratory)": 184897,
}

report = []
def say(line=""):
    print(line)
    report.append(line)

def compare(name, value):
    exp = EXPECTED.get(name)
    status = "" if exp is None else ("  MATCH" if exp == value else f"  DIFFERS (earlier: {exp:,})")
    say(f"{name:<42}{value:>10,}{status}")

def normalize(text):
    """Strip, casefold, and collapse runs of whitespace to one space."""
    return re.sub(r"\s+", " ", text.strip()).casefold()

# ---------------------------------------------------------------- 1. load files
frames = []
say("== Source files ==")
for path in sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.name.lower()):
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()[:16]
    try:
        raw_bytes.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = "NOT valid utf-8"
    # keep_default_na=False: a review that literally says "NA" or "None" stays text.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")
    if set(df.columns) != {"content", "score", "year"}:  # files store them as year, content, score
        say(f"WARNING {path.name}: unexpected columns {list(df.columns)}")
    df["bank"] = path.name.replace("_reviews.csv", "")
    df["source_file"] = path.name
    df["source_row"] = range(len(df))
    exp = EXPECTED_FILE_ROWS.get(path.name)
    flag = "MATCH" if exp == len(df) else f"DIFFERS (earlier: {exp})"
    say(f"{path.name:<28}{len(df):>8,}  {flag:<22} {encoding:<8} sha256[:16]={sha}")
    frames.append(df)

missing = set(EXPECTED_FILE_ROWS) - {p.name for p in RAW_DIR.glob("*.csv")}
if missing:
    say(f"WARNING missing files: {sorted(missing)}")

data = pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------- 2. validate fields
say("\n== Field checks ==")
data["score_num"] = pd.to_numeric(data["score"], errors="coerce")
bad_scores = data[~data["score_num"].isin([1, 2, 3, 4, 5])]
say(f"rows with score outside 1-5 or non-numeric: {len(bad_scores):,}")
years = pd.to_numeric(data["year"], errors="coerce")
say(f"rows with non-numeric year: {years.isna().sum():,}")
say(f"year range: {int(years.min())}-{int(years.max())}")
say("rows per year: " + ", ".join(f"{int(y)}={n:,}" for y, n in years.value_counts().sort_index().items()))
say("raw rating counts: " + ", ".join(f"{int(s)}*={n:,}" for s, n in data["score_num"].value_counts().sort_index().items()))
literal_na_like = data["content"].str.strip().str.casefold().isin(["na", "n/a", "nan", "none", "null"]).sum()
say(f"reviews whose whole text is NA/None/null-like (kept as text): {literal_na_like:,}")

# ---------------------------------------------------------------- 3. audit sequence
say("\n== Audit (same sequence as 17 November) ==")
log = []
compare("raw_rows", len(data))

data["norm"] = data["content"].map(normalize)
empty = data["norm"] == ""
compare("empty_text_rows_all_ratings", int(empty.sum()))
log.append(("empty text after normalization", int(empty.sum())))

nonempty = data[~empty]
rating3 = nonempty["score_num"] == 3
other = ~nonempty["score_num"].isin([1, 2, 3, 4, 5])
log.append(("rating 3 (excluded from binary task)", int(rating3.sum())))
log.append(("invalid rating", int(other.sum())))
binary = nonempty[nonempty["score_num"].isin([1, 2, 4, 5])].copy()
binary["label"] = (binary["score_num"] >= 4).astype(int)
compare("nonempty_binary_rows", len(binary))

groups = binary.groupby("norm", sort=False).agg(
    n_rows=("label", "size"),
    n_labels=("label", "nunique"),
    label=("label", "first"),
    content=("content", "first"),          # first original text in the group
    banks=("bank", lambda s: "|".join(sorted(set(s)))),
    years=("year", lambda s: "|".join(sorted(set(s)))),
    scores=("score", lambda s: "|".join(sorted(set(s)))),
).reset_index()
compare("unique_texts_before_conflicts", len(groups))
log.append(("exact duplicate rows collapsed (same normalized text)", len(binary) - len(groups)))

conflict = groups["n_labels"] > 1
compare("conflicting_groups", int(conflict.sum()))
rows_in_conflict = int(groups.loc[conflict, "n_rows"].sum())
log.append(("conflicting-label groups removed (groups)", int(conflict.sum())))

cand = groups[~conflict].drop(columns="n_labels").copy()
compare("candidates_after_conflicts", len(cand))
compare("negative_candidates", int((cand["label"] == 0).sum()))
compare("positive_candidates", int((cand["label"] == 1).sum()))

ntok = cand["norm"].str.split().str.len()
compare("candidates_3plus_tokens (exploratory)", int((ntok >= 3).sum()))
say(f"  of which negative / positive: {int(((ntok >= 3) & (cand.label == 0)).sum()):,} / "
    f"{int(((ntok >= 3) & (cand.label == 1)).sum()):,}   (earlier: 91,682 / 93,215; NOT adopted)")

# ---------------------------------------------------------------- 4. descriptive checks
say("\n== Candidate pool description (not a satisfaction rate) ==")
multi_bank = cand["banks"].str.contains("|", regex=False)
say(f"texts that appear under more than one bank: {int(multi_bank.sum()):,}")
say(f"texts that occurred more than once in the raw data: {int((cand['n_rows'] > 1).sum()):,}")
say(f"raw rows inside conflicting groups: {rows_in_conflict:,}")
say("token length (whitespace) quantiles: " +
    ", ".join(f"p{int(q*100)}={int(ntok.quantile(q))}" for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
say("most common texts among candidates:")
for _, r in cand.sort_values("n_rows", ascending=False).head(10).iterrows():
    say(f"  {r['n_rows']:>6,} x  label={r['label']}  {r['content'][:60]!r}")

# ---------------------------------------------------------------- 5. save
cand["text_hash"] = cand["norm"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
cand.to_csv(INTERIM_DIR / "candidates.csv.gz", index=False)
pd.DataFrame(log, columns=["step", "removed"]).to_csv(OUT_DIR / "removal_log.csv", index=False)
(OUT_DIR / "audit_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
say(f"\nSaved {len(cand):,} candidates to {INTERIM_DIR / 'candidates.csv.gz'}")
