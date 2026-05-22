from pathlib import Path
import pandas as pd
import torch

from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory

MODEL_DIR = Path("pykeen-lanl-model")
TRIPLES_PATH = Path("generated-files/triples.tsv")

model = torch.load(
    MODEL_DIR / "trained_model.pkl",
    map_location="cpu",
    weights_only=False,
)

tf = TriplesFactory.from_path(
    TRIPLES_PATH,
    create_inverse_triples=True,
)

normal = pd.read_csv(
    TRIPLES_PATH,
    sep="\t",
    names=["head", "relation", "tail"],
)

normal_sample = normal.sample(
    n=min(10_000, len(normal)),
    random_state=42,
)

triples = list(normal_sample.itertuples(index=False, name=None))

pack = predict_triples(
    model=model,
    triples=triples,
    triples_factory=tf,
)

df = pack.process(factory=tf).df
df.to_csv("generated-files/normal_scores.csv", index=False)

print(df.head(20))
print("Saved generated-files/normal_scores.csv")