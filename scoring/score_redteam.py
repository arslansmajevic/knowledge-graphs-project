from pathlib import Path
import pandas as pd
import torch

from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory

MODEL_DIR = Path("pykeen-lanl-model")
TRIPLES_PATH = Path("generated-files/triples.tsv")
REDTEAM_PATH = Path("dataset/redteam.txt")

# Load model
model = torch.load(
    MODEL_DIR / "trained_model.pkl",
    map_location="cpu",
    weights_only=False,  # needed for newer PyTorch versions
)

# Load the same triples factory mapping used for labels
tf = TriplesFactory.from_path(
    TRIPLES_PATH,
    create_inverse_triples=True,
)

red_cols = ["time", "user", "src_computer", "dst_computer"]
red = pd.read_csv(REDTEAM_PATH, names=red_cols)

candidate_triples = []

for row in red.itertuples(index=False):
    u = f"user:{row.user}"
    sc = f"computer:{row.src_computer}"
    dc = f"computer:{row.dst_computer}"

    candidate_triples.append((u, "logs_on_to", dc))
    candidate_triples.append((sc, "authenticates_to", dc))
    candidate_triples.append((u, "uses_source_computer", sc))

# Keep only triples whose entities/relations exist in the training mapping
known_triples = []
unknown_triples = []

for h, r, t in candidate_triples:
    if (
        h in tf.entity_to_id
        and t in tf.entity_to_id
        and r in tf.relation_to_id
    ):
        known_triples.append((h, r, t))
    else:
        unknown_triples.append((h, r, t))

print(f"Candidate redteam triples: {len(candidate_triples):,}")
print(f"Known/scorable triples:    {len(known_triples):,}")
print(f"Unknown/unscorable:        {len(unknown_triples):,}")

pack = predict_triples(
    model=model,
    triples=known_triples,
    triples_factory=tf,
)

df = pack.process(factory=tf).df

df.to_csv("generated-files/redteam_scores.csv", index=False)

print(df.head(20))
print("Saved generated-files/redteam_scores.csv")