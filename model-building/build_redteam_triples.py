from pathlib import Path
import pandas as pd

DATA = Path("dataset")
OUT = Path("redteam_triples.tsv")


def user(x):
    return f"user:{x}"


def computer(x):
    return f"computer:{x}"


red_cols = ["time", "user", "src_computer", "dst_computer"]

red = pd.read_csv(
    DATA / "redteam.txt",
    names=red_cols,
    compression="infer",
)

triples = set()

for row in red.itertuples(index=False):
    u = user(row.user)
    sc = computer(row.src_computer)
    dc = computer(row.dst_computer)

    triples.add((u, "logs_on_to", dc))
    triples.add((sc, "authenticates_to", dc))
    triples.add((u, "uses_source_computer", sc))

with OUT.open("w") as f:
    for h, r, t in sorted(triples):
        f.write(f"{h}\t{r}\t{t}\n")

print(f"Wrote {len(triples):,} redteam candidate triples to {OUT}")