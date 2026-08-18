"""Parse the hand-written question banks into a single augmentation CSV."""
import csv, glob, os, re, sys

SRC = os.path.expanduser("~/Training/dl-compression/data/aug")
OUT = os.path.join(SRC, "augment.csv")
LETTERS = list("ABCDE")

rows, seen = [], set()
for path in sorted(glob.glob(f"{SRC}/qbank_*.txt")):
    block = {}
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("Q:"):
            block = {"prompt": line[2:].strip()}
        elif re.match(r"^[A-E]\) ", line):
            block[line[0]] = line[3:].strip()
        elif line.startswith("*:"):
            block["answer"] = line[2:].strip()
            missing = [k for k in ["prompt", "answer"] + LETTERS if not block.get(k)]
            if missing:
                sys.exit(f"{path}: incomplete block {block.get('prompt','?')[:60]} missing {missing}")
            if block["answer"] not in LETTERS:
                sys.exit(f"{path}: bad answer {block['answer']}")
            key = block["prompt"].lower()
            if key in seen:
                sys.exit(f"{path}: duplicate prompt {key[:60]}")
            seen.add(key)
            rows.append(block)
            block = {}

for i, r in enumerate(rows):
    r["id"] = i

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "prompt"] + LETTERS + ["answer"])
    w.writeheader()
    w.writerows(rows)

dist = {l: sum(r["answer"] == l for r in rows) for l in LETTERS}
print(f"{len(rows)} questions -> {OUT}")
print("answer spread", dist)
