"""End-to-end knowledge-graph anomaly-detection pipeline.

Run the whole project with a single command:

    python pipeline.py

This performs, in order:

    1. build   - turn the raw LANL logs into knowledge-graph triples
    2. train   - train one or more PyKEEN embedding models on those triples
    3. score   - score the red-team (malicious) and a sample of normal triples
    4. evaluate- compare the two score distributions (ROC-AUC, AP, ...)

Each step writes its artifacts to ``generated-files/`` (and each trained model to
``pykeen-lanl-model/<model>/``) so steps can also be run individually, e.g.::

    python pipeline.py --steps train score evaluate

Use ``--help`` to see all options.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path("dataset")
GEN_DIR = Path("generated-files")
MODEL_DIR = Path("pykeen-lanl-model")

TRIPLES_PATH = GEN_DIR / "triples.tsv"

# Only keep the first day of events to keep the graph small. Set to ``None`` to
# use every event in the dataset.
MAX_TIME = 24 * 60 * 60

# Training hyper-parameters.
#
# ``MODELS`` lists every PyKEEN model to train and evaluate. Any model name from
# PyKEEN's registry works (e.g. "TransE", "DistMult", "ComplEx", "RotatE", ...).
# Each model is trained independently and the ``evaluate`` step prints a
# side-by-side comparison of their anomaly-detection metrics.
MODELS = ["TransE", "DistMult"]

# Optional per-model keyword arguments passed to PyKEEN's ``model_kwargs``. Every
# model gets ``embedding_dim=EMBEDDING_DIM`` by default; add an entry here only
# to override or extend that for a specific model.
MODEL_KWARGS: dict = {}

EMBEDDING_DIM = 64
EPOCHS = 5
BATCH_SIZE = 1024
RANDOM_STATE = 42

# Evaluation speed/accuracy trade-off. Ranking every test triple against all
# ~110k entities on CPU takes many hours, so by default we evaluate on a random
# subsample of the held-out test triples. Set ``EVAL_SAMPLE_SIZE = None`` to
# evaluate on the full test split. ``EVAL_BATCH_SIZE`` avoids PyKEEN's very
# conservative automatic batch size (32) on CPU.
EVAL_SAMPLE_SIZE = 10_000
EVAL_BATCH_SIZE = 256

# Number of normal triples sampled as the "benign" class when evaluating.
NORMAL_SAMPLE_SIZE = 10_000

ALL_STEPS = ("build", "train", "score", "evaluate")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _model_dir(model_name: str) -> Path:
    """Directory where a given model's PyKEEN artifacts are stored."""
    return MODEL_DIR / model_name


def _redteam_scores_path(model_name: str) -> Path:
    return GEN_DIR / f"redteam_scores_{model_name}.csv"


def _normal_scores_path(model_name: str) -> Path:
    return GEN_DIR / f"normal_scores_{model_name}.csv"


def _clean(x):
    """Normalise a raw cell value, mapping missing/``?`` values to ``None``."""
    if pd.isna(x):
        return None
    x = str(x)
    return None if x == "?" else x


def _user(x):
    x = _clean(x)
    return f"user:{x}" if x else None


def _computer(x):
    x = _clean(x)
    return f"computer:{x}" if x else None


def _process(x):
    x = _clean(x)
    return f"process:{x}" if x else None


def _port(x):
    x = _clean(x)
    return f"port:{x}" if x else None


def _add(triples, h, r, t):
    if h and r and t:
        triples.add((h, r, t))


def _read_csv(name, columns):
    df = pd.read_csv(DATA_DIR / name, names=columns, compression="infer")
    if MAX_TIME is not None:
        df = df[df["time"] <= MAX_TIME]
    print(f"[build]   read {name}: {len(df):,} rows")
    return df


def _redteam_candidate_triples():
    """Build the candidate triples implied by the red-team ground truth."""
    red = pd.read_csv(
        DATA_DIR / "redteam.txt",
        names=["time", "user", "src_computer", "dst_computer"],
        compression="infer",
    )

    triples = []
    for row in tqdm(red.itertuples(index=False), total=len(red),
                    desc="[score] redteam", unit="row"):
        u = f"user:{row.user}"
        sc = f"computer:{row.src_computer}"
        dc = f"computer:{row.dst_computer}"
        triples.append((u, "logs_on_to", dc))
        triples.append((sc, "authenticates_to", dc))
        triples.append((u, "uses_source_computer", sc))
    return triples


# --------------------------------------------------------------------------- #
# Step 1: build the knowledge graph
# --------------------------------------------------------------------------- #
def build() -> None:
    """Turn the raw LANL logs into a PyKEEN-compatible triples file."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    triples: set = set()

    # auth.txt: authentication events
    auth = _read_csv(
        "auth.txt",
        [
            "time", "src_user", "dst_user", "src_computer", "dst_computer",
            "auth_type", "logon_type", "orientation", "result",
        ],
    )
    for row in tqdm(auth.itertuples(index=False), total=len(auth),
                    desc="[build] auth", unit="row"):
        su, du = _user(row.src_user), _user(row.dst_user)
        sc, dc = _computer(row.src_computer), _computer(row.dst_computer)
        if _clean(row.orientation) == "LogOn" and _clean(row.result) == "Success":
            _add(triples, su, "logs_on_to", dc)
            _add(triples, sc, "authenticates_to", dc)
            _add(triples, su, "uses_source_computer", sc)
        _add(triples, su, "authenticates_as", du)

    # dns.txt: DNS lookups
    dns = _read_csv("dns.txt", ["time", "src_computer", "resolved_computer"])
    for row in tqdm(dns.itertuples(index=False), total=len(dns),
                    desc="[build] dns", unit="row"):
        _add(triples, _computer(row.src_computer), "dns_resolves",
             _computer(row.resolved_computer))

    # flows.txt: network flows
    flows = _read_csv(
        "flows.txt",
        [
            "time", "duration", "src_computer", "src_port", "dst_computer",
            "dst_port", "protocol", "packet_count", "byte_count",
        ],
    )
    for row in tqdm(flows.itertuples(index=False), total=len(flows),
                    desc="[build] flows", unit="row"):
        sc, dc = _computer(row.src_computer), _computer(row.dst_computer)
        _add(triples, sc, "flows_to", dc)
        _add(triples, sc, "uses_src_port", _port(row.src_port))
        _add(triples, dc, "uses_dst_port", _port(row.dst_port))

    # proc.txt: process start/stop events
    proc = _read_csv("proc.txt", ["time", "user", "computer", "process", "action"])
    for row in tqdm(proc.itertuples(index=False), total=len(proc),
                    desc="[build] proc", unit="row"):
        u, c, p = _user(row.user), _computer(row.computer), _process(row.process)
        action = _clean(row.action)
        if action == "Start":
            _add(triples, u, "starts_process", p)
            _add(triples, c, "runs_process", p)
            _add(triples, u, "active_on_computer", c)
        elif action == "End":
            _add(triples, u, "ends_process", p)
            _add(triples, c, "stops_process", p)

    print(f"[build] sorting and writing {len(triples):,} unique triples ...")
    with TRIPLES_PATH.open("w") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")

    print(f"[build] Wrote {len(triples):,} triples to {TRIPLES_PATH}")


# --------------------------------------------------------------------------- #
# Step 2: train the embedding model
# --------------------------------------------------------------------------- #
def train() -> None:
    """Train every configured PyKEEN model on the generated triples."""
    from pykeen.pipeline import pipeline as pykeen_pipeline
    from pykeen.triples import TriplesFactory

    tf = TriplesFactory.from_path(TRIPLES_PATH, create_inverse_triples=True)
    print(f"[train] {tf}")

    training, testing = tf.split([0.8, 0.2], random_state=RANDOM_STATE)

    # Evaluating every held-out triple against all entities is the slow part of
    # the pipeline. Subsample the test split so evaluation finishes in seconds
    # while still giving a meaningful estimate of model quality.
    eval_factory = testing
    if EVAL_SAMPLE_SIZE is not None and testing.num_triples > EVAL_SAMPLE_SIZE:
        import torch

        generator = torch.Generator().manual_seed(RANDOM_STATE)
        idx = torch.randperm(testing.num_triples, generator=generator)[:EVAL_SAMPLE_SIZE]
        eval_factory = testing.clone_and_exchange_triples(testing.mapped_triples[idx])
        print(
            f"[train] evaluating on a random sample of {eval_factory.num_triples:,} "
            f"of {testing.num_triples:,} test triples"
        )

    for i, model_name in enumerate(MODELS, start=1):
        print(f"\n[train] ({i}/{len(MODELS)}) training {model_name} ...")
        model_kwargs = dict(embedding_dim=EMBEDDING_DIM)
        model_kwargs.update(MODEL_KWARGS.get(model_name, {}))

        result = pykeen_pipeline(
            training=training,
            testing=eval_factory,
            model=model_name,
            epochs=EPOCHS,
            model_kwargs=model_kwargs,
            training_kwargs=dict(batch_size=BATCH_SIZE),
            evaluation_kwargs=dict(batch_size=EVAL_BATCH_SIZE),
        )
        out_dir = _model_dir(model_name)
        result.save_to_directory(str(out_dir))
        print(f"[train] Saved {model_name} model to {out_dir}")


# --------------------------------------------------------------------------- #
# Step 3: score red-team and normal triples
# --------------------------------------------------------------------------- #
def score() -> None:
    """Score the red-team triples and a sample of normal triples per model."""
    import torch
    from pykeen.predict import predict_triples
    from pykeen.triples import TriplesFactory

    GEN_DIR.mkdir(parents=True, exist_ok=True)

    tf = TriplesFactory.from_path(TRIPLES_PATH, create_inverse_triples=True)

    # --- red-team (malicious) triples (model-independent candidates) ---
    candidates = _redteam_candidate_triples()
    known = [
        (h, r, t)
        for h, r, t in candidates
        if h in tf.entity_to_id and t in tf.entity_to_id and r in tf.relation_to_id
    ]
    print(
        f"[score] red-team candidates: {len(candidates):,} | "
        f"scorable: {len(known):,} | unknown: {len(candidates) - len(known):,}"
    )

    # --- normal (benign) triples (same sample reused across models) ---
    normal = pd.read_csv(TRIPLES_PATH, sep="\t", names=["head", "relation", "tail"])
    sample = normal.sample(n=min(NORMAL_SAMPLE_SIZE, len(normal)), random_state=RANDOM_STATE)
    normal_triples = list(sample.itertuples(index=False, name=None))

    for i, model_name in enumerate(MODELS, start=1):
        print(f"\n[score] ({i}/{len(MODELS)}) scoring with {model_name} ...")
        model = torch.load(
            _model_dir(model_name) / "trained_model.pkl",
            map_location="cpu",
            weights_only=False,
        )

        red_df = predict_triples(model=model, triples=known, triples_factory=tf).process(factory=tf).df
        red_path = _redteam_scores_path(model_name)
        red_df.to_csv(red_path, index=False)
        print(f"[score] Saved {red_path}")

        normal_df = predict_triples(model=model, triples=normal_triples, triples_factory=tf).process(factory=tf).df
        normal_path = _normal_scores_path(model_name)
        normal_df.to_csv(normal_path, index=False)
        print(f"[score] Saved {normal_path}")


# --------------------------------------------------------------------------- #
# Step 4: evaluate detection performance
# --------------------------------------------------------------------------- #
def evaluate() -> None:
    """Compare red-team vs. normal score distributions for every model."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    summary = []
    for model_name in MODELS:
        normal = pd.read_csv(_normal_scores_path(model_name))
        red = pd.read_csv(_redteam_scores_path(model_name))
        normal["label"] = 0
        red["label"] = 1

        df = pd.concat([normal, red], ignore_index=True)
        # PyKEEN score: higher = more plausible. Invert it for anomaly detection.
        df["anomaly_score"] = -df["score"]

        auc = roc_auc_score(df["label"], df["anomaly_score"])
        ap = average_precision_score(df["label"], df["anomaly_score"])
        summary.append({"model": model_name, "roc_auc": auc, "average_precision": ap})

        print(f"\n===== [evaluate] {model_name} =====")
        print(f"[evaluate] ROC-AUC: {auc:.4f}")
        print(f"[evaluate] Average precision: {ap:.4f}")
        print("\n[evaluate] Normal anomaly scores:")
        print(df[df["label"] == 0]["anomaly_score"].describe())
        print("\n[evaluate] Red-team anomaly scores:")
        print(df[df["label"] == 1]["anomaly_score"].describe())

        # Also surface the training metrics, if available.
        results_path = _model_dir(model_name) / "results.json"
        if results_path.exists():
            with results_path.open() as f:
                metrics = json.load(f).get("metric_results", {})
            print("\n[evaluate] Training metrics (excerpt):")
            print(json.dumps(metrics, indent=2)[:2000])

    # Side-by-side comparison of all models.
    comparison = pd.DataFrame(summary).sort_values("roc_auc", ascending=False)
    print("\n===== [evaluate] model comparison =====")
    print(comparison.to_string(index=False))


STEP_FUNCS = {
    "build": build,
    "train": train,
    "score": score,
    "evaluate": evaluate,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LANL knowledge-graph anomaly-detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=list(ALL_STEPS),
        help="Subset of steps to run, in order.",
    )
    args = parser.parse_args()

    overall_start = time.perf_counter()
    for i, step in enumerate(args.steps, start=1):
        print(f"\n===== [{i}/{len(args.steps)}] {step} =====")
        step_start = time.perf_counter()
        STEP_FUNCS[step]()
        print(f"===== {step} done in {time.perf_counter() - step_start:.1f}s =====")
    print(f"\nPipeline finished in {time.perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
