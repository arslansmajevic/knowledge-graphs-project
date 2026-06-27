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

import hybrid
import mitre
import reasoning

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

# Artifacts produced by the logical/knowledge components.
MITRE_TRIPLES_PATH = GEN_DIR / "mitre_triples.tsv"      # ATT&CK knowledge base
DERIVED_TRIPLES_PATH = GEN_DIR / "derived_triples.tsv"  # Datalog-derived facts
FLAGGED_PATH = GEN_DIR / "flagged_entities.txt"         # logical anomaly signal
CHAINS_PATH = GEN_DIR / "attack_chains.json"            # minted attack chains

# Only keep the first day of events to keep the graph small. Set to ``None`` to
# use every event in the dataset.
MAX_TIME = 24 * 60 * 60

# --- Hybrid (logical + sub-symbolic) settings -------------------------------- #
# Whether to ingest the MITRE ATT&CK knowledge base and fold it into the graph
# the embeddings are trained on (a second data model alongside the logs).
INCLUDE_MITRE_TRIPLES = True
# Whether to feed the Datalog-derived triples (lateral_movement, attack chains,
# ...) back into the embedding graph as logical "provenance" metadata.
INCLUDE_DERIVED_TRIPLES = True
# Maximum lateral-movement chain length the reasoner explores (bounds the
# recursive transitive closure on large graphs).
MAX_CHAIN_DEPTH = 6
# Weight of the logical flag when combined with the standardised KGE anomaly
# score in the hybrid detector (see ``hybrid.py``).
HYBRID_LAMBDA = 1.0

# For the ``evolve`` step (LO8): the later time window (in seconds) used to
# simulate the Knowledge Graph growing as new logs arrive.
EVOLVE_TIME = 2 * 24 * 60 * 60

# Training hyper-parameters.
#
# ``MODELS`` lists every PyKEEN model to train and evaluate. Any model name from
# PyKEEN's registry works (e.g. "TransE", "DistMult", "ComplEx", "RotatE", ...).
# Each model is trained independently and the ``evaluate`` step prints a
# side-by-side comparison of their anomaly-detection metrics.
MODELS = ["TransE", "DistMult", "RotatE", "ComplEx"]

# Optional per-model keyword arguments passed to PyKEEN's ``model_kwargs``. Every
# model gets ``embedding_dim=EMBEDDING_DIM`` by default; add an entry here only
# to override or extend that for a specific model.
MODEL_KWARGS: dict = {}

EMBEDDING_DIM = 64
EPOCHS = 5
RANDOM_STATE = 42

# Training batch size. Larger batches keep the GPU busier and train faster, but
# GPUs handle big batches far better than CPUs, so the device-appropriate value
# is chosen automatically at run time (see ``train``). On a CPU a smaller batch
# avoids thrashing; on a GPU a large batch maximises utilisation.
CPU_BATCH_SIZE = 1024
GPU_BATCH_SIZE = 8192

# DataLoader workers used to feed the model during training. Extra workers
# overlap batch preparation with GPU compute so the GPU does not sit idle
# waiting for data; on CPU keep this at 0 to avoid process-spawn overhead.
GPU_NUM_WORKERS = 2

# Evaluation speed/accuracy trade-off. Ranking every test triple against all
# ~110k entities on CPU takes many hours, so by default we evaluate on a random
# subsample of the held-out test triples. Set ``EVAL_SAMPLE_SIZE = None`` to
# evaluate on the full test split. The eval batch size also adapts to the device
# (a GPU can score far more triples at once than a CPU).
EVAL_SAMPLE_SIZE = 10_000
CPU_EVAL_BATCH_SIZE = 256
GPU_EVAL_BATCH_SIZE = 4096

# Number of normal triples sampled as the "benign" class when evaluating.
NORMAL_SAMPLE_SIZE = 100_000

ALL_STEPS = ("build", "reason", "train", "score", "evaluate")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _model_dir(model_name: str) -> Path:
    """Directory where a given model's PyKEEN artifacts are stored."""
    return MODEL_DIR / model_name


def _select_device():
    """Return the best available torch device ("cuda" if a GPU is present)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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


def _read_csv(name, columns, max_time=MAX_TIME):
    df = pd.read_csv(DATA_DIR / name, names=columns, compression="infer")
    if max_time is not None:
        df = df[df["time"] <= max_time]
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
def build(max_time=MAX_TIME, triples_path: Path = TRIPLES_PATH) -> None:
    """Turn the raw LANL logs into a PyKEEN-compatible triples file.

    Also ingests the MITRE ATT&CK knowledge base as a second data model
    (``INCLUDE_MITRE_TRIPLES``), linking ATT&CK techniques to the log signals
    they are detected through so both live in one knowledge graph (LO4/LO7).
    """
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    triples: set = set()

    # auth.txt: authentication events
    auth = _read_csv(
        "auth.txt",
        [
            "time", "src_user", "dst_user", "src_computer", "dst_computer",
            "auth_type", "logon_type", "orientation", "result",
        ],
        max_time=max_time,
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
    dns = _read_csv("dns.txt", ["time", "src_computer", "resolved_computer"],
                    max_time=max_time)
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
        max_time=max_time,
    )
    for row in tqdm(flows.itertuples(index=False), total=len(flows),
                    desc="[build] flows", unit="row"):
        sc, dc = _computer(row.src_computer), _computer(row.dst_computer)
        _add(triples, sc, "flows_to", dc)
        _add(triples, sc, "uses_src_port", _port(row.src_port))
        _add(triples, dc, "uses_dst_port", _port(row.dst_port))

    # proc.txt: process start/stop events
    proc = _read_csv("proc.txt", ["time", "user", "computer", "process", "action"],
                     max_time=max_time)
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
    with triples_path.open("w") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")

    print(f"[build] Wrote {len(triples):,} triples to {triples_path}")

    # MITRE ATT&CK knowledge base -> a second data model in the same graph.
    if INCLUDE_MITRE_TRIPLES:
        n = mitre.write_attack_triples(MITRE_TRIPLES_PATH)
        print(f"[build] Wrote {n:,} MITRE ATT&CK triples to {MITRE_TRIPLES_PATH}")


# --------------------------------------------------------------------------- #
# Step 2: logical reasoning (Datalog over the knowledge graph)
# --------------------------------------------------------------------------- #
def reason() -> None:
    """Run the symbolic reasoner (``reasoning.py``) on the built graph.

    Encodes MITRE ATT&CK / cyber-kill-chain patterns as Datalog rules and uses
    *full recursion* (lateral-movement transitive closure) plus *object
    creation* (minting ``attack_chain`` entities) to discover unknown parts of
    the graph (LO2/LO6). The outputs are:

    * ``derived_triples.tsv`` – fed back into the embedding graph as logical
      provenance (the hybrid link, LO2/LO12);
    * ``flagged_entities.txt`` – the logical anomaly signal used by ``evaluate``;
    * ``attack_chains.json`` – materialised chains shown in the dashboard.
    """
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    facts = reasoning.load_facts(TRIPLES_PATH)
    print(f"[reason] loaded EDB relations: {sorted(facts)}")
    derived = reasoning.run(facts, max_depth=MAX_CHAIN_DEPTH)

    n_triples = reasoning.write_derived(DERIVED_TRIPLES_PATH, derived)
    n_flagged = reasoning.write_flagged(FLAGGED_PATH, derived)
    n_chains = reasoning.write_chains(CHAINS_PATH, derived)

    summary = reasoning.summary(derived)
    print(f"[reason] derived: {summary}")
    print(f"[reason] Wrote {n_triples:,} derived triples to {DERIVED_TRIPLES_PATH}")
    print(f"[reason] Wrote {n_flagged:,} flagged entities to {FLAGGED_PATH}")
    print(f"[reason] Wrote {n_chains:,} attack chains to {CHAINS_PATH}")


def _training_triple_files() -> list[Path]:
    """Triple files combined for embedding training.

    Always the log graph; optionally the MITRE ATT&CK triples and the
    Datalog-derived triples (logical provenance), depending on the
    ``INCLUDE_*`` switches and whether the files exist.
    """
    files = [TRIPLES_PATH]
    if INCLUDE_MITRE_TRIPLES and MITRE_TRIPLES_PATH.exists():
        files.append(MITRE_TRIPLES_PATH)
    if INCLUDE_DERIVED_TRIPLES and DERIVED_TRIPLES_PATH.exists():
        files.append(DERIVED_TRIPLES_PATH)
    return files


def _load_training_factory():
    """Build one ``TriplesFactory`` from all configured triple sources."""
    import numpy as np
    from pykeen.triples import TriplesFactory

    files = _training_triple_files()
    arrays = []
    for path in files:
        arr = pd.read_csv(path, sep="\t", names=["h", "r", "t"], dtype=str).to_numpy()
        arrays.append(arr)
        print(f"[train] + {len(arr):,} triples from {path.name}")
    triples = np.unique(np.vstack(arrays), axis=0)
    print(f"[train] combined {len(triples):,} unique triples from {len(files)} source(s)")
    return TriplesFactory.from_labeled_triples(
        triples, create_inverse_triples=True
    )


# --------------------------------------------------------------------------- #
# Step 3: train the embedding model
# --------------------------------------------------------------------------- #
def train() -> None:
    """Train every configured PyKEEN model on the generated triples."""
    from pykeen.pipeline import pipeline as pykeen_pipeline

    tf = _load_training_factory()
    print(f"[train] {tf}")

    # Pick the device once and tune the workload to it: a GPU can chew through
    # much larger batches than a CPU, so use bigger batches and overlap data
    # loading with compute to keep the GPU busy instead of idling on I/O.
    device = _select_device()
    on_gpu = device == "cuda"
    batch_size = GPU_BATCH_SIZE if on_gpu else CPU_BATCH_SIZE
    eval_batch_size = GPU_EVAL_BATCH_SIZE if on_gpu else CPU_EVAL_BATCH_SIZE
    num_workers = GPU_NUM_WORKERS if on_gpu else 0
    if on_gpu:
        import torch

        print(f"[train] using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[train] no GPU detected, training on CPU")
    print(
        f"[train] device={device} batch_size={batch_size:,} "
        f"eval_batch_size={eval_batch_size:,} num_workers={num_workers}"
    )

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

    # Overlap host-to-device transfers with compute when a GPU is available.
    training_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    if on_gpu:
        training_kwargs["pin_memory"] = True

    for i, model_name in enumerate(MODELS, start=1):
        print(f"\n[train] ({i}/{len(MODELS)}) training {model_name} ...")
        model_kwargs = dict(embedding_dim=EMBEDDING_DIM)
        model_kwargs.update(MODEL_KWARGS.get(model_name, {}))

        result = pykeen_pipeline(
            training=training,
            testing=eval_factory,
            model=model_name,
            epochs=EPOCHS,
            device=device,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            evaluation_kwargs=dict(batch_size=eval_batch_size),
        )
        out_dir = _model_dir(model_name)
        result.save_to_directory(str(out_dir))
        print(f"[train] Saved {model_name} model to {out_dir}")


# --------------------------------------------------------------------------- #
# Step 4: score red-team and normal triples
# --------------------------------------------------------------------------- #
def score() -> None:
    """Score the red-team triples and a sample of normal triples per model."""
    import torch
    from pykeen.predict import predict_triples

    GEN_DIR.mkdir(parents=True, exist_ok=True)

    # Use the SAME combined factory the models were trained on so entity and
    # relation ids line up with each saved model's embeddings.
    tf = _load_training_factory()

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
# Step 5: evaluate detection performance (KGE-only vs. hybrid)
# --------------------------------------------------------------------------- #
def evaluate() -> None:
    """Compare red-team vs. normal scores for every model, KGE-only and hybrid.

    The hybrid detector adds the symbolic reasoner's logical signal (entities
    implicated in a lateral-movement / attack chain) on top of the embedding
    anomaly score, directly evidencing how symbolic and sub-symbolic methods
    combine in a KG (LO2/LO12). The comparison table reports both so the report
    can show whether the hybrid beats the pure-KGE baseline.
    """
    flagged = reasoning.load_flagged(FLAGGED_PATH)
    if flagged:
        print(f"[evaluate] loaded {len(flagged):,} flagged entities (logical signal)")
    else:
        print("[evaluate] no flagged entities found — run the 'reason' step to "
              "enable the hybrid detector (falling back to KGE-only)")

    summary = []
    for model_name in MODELS:
        normal = pd.read_csv(_normal_scores_path(model_name))
        red = pd.read_csv(_redteam_scores_path(model_name))
        df = hybrid.labelled_frame(normal, red)

        res = hybrid.evaluate_frame(df, flagged, lam=HYBRID_LAMBDA)
        summary.append(
            {
                "model": model_name,
                "kge_roc_auc": res["kge"]["roc_auc"],
                "hybrid_roc_auc": res["hybrid"]["roc_auc"],
                "roc_auc_delta": res["roc_auc_delta"],
                "kge_avg_precision": res["kge"]["average_precision"],
                "hybrid_avg_precision": res["hybrid"]["average_precision"],
            }
        )

        anomaly = hybrid.kge_anomaly(df)
        print(f"\n===== [evaluate] {model_name} =====")
        print(f"[evaluate] KGE-only   ROC-AUC: {res['kge']['roc_auc']:.4f} | "
              f"AP: {res['kge']['average_precision']:.4f}")
        print(f"[evaluate] Hybrid     ROC-AUC: {res['hybrid']['roc_auc']:.4f} | "
              f"AP: {res['hybrid']['average_precision']:.4f} "
              f"(Δ ROC-AUC {res['roc_auc_delta']:+.4f})")
        print(f"[evaluate] logical flag hits: {res['flagged_positive_rate']:.2%} of "
              f"red-team vs {res['flagged_negative_rate']:.2%} of normal triples")
        print("\n[evaluate] Normal KGE anomaly scores:")
        print(pd.Series(anomaly[df["label"] == 0]).describe())
        print("\n[evaluate] Red-team KGE anomaly scores:")
        print(pd.Series(anomaly[df["label"] == 1]).describe())

        # Also surface the training metrics, if available.
        results_path = _model_dir(model_name) / "results.json"
        if results_path.exists():
            with results_path.open() as f:
                metrics = json.load(f).get("metric_results", {})
            print("\n[evaluate] Training metrics (excerpt):")
            print(json.dumps(metrics, indent=2)[:2000])

    # Side-by-side comparison of all models (ranked by the hybrid detector).
    comparison = pd.DataFrame(summary).sort_values("hybrid_roc_auc", ascending=False)
    print("\n===== [evaluate] model comparison (KGE-only vs hybrid) =====")
    print(comparison.to_string(index=False))


# --------------------------------------------------------------------------- #
# Optional step: demonstrate Knowledge Graph evolution (LO8)
# --------------------------------------------------------------------------- #
def evolve() -> None:
    """Show the KG growing as new logs arrive, then re-detect on the larger graph.

    Rebuilds the graph over a *later* time window (``EVOLVE_TIME`` instead of
    ``MAX_TIME``), reports how many triples were added, re-runs the reasoner and
    re-scores. This is the Knowledge-Graph-completion side of evolution (LO8):
    new events extend the graph and link prediction / reasoning are re-applied.
    """
    base_count = _count_lines(TRIPLES_PATH) if TRIPLES_PATH.exists() else 0
    print(f"[evolve] current graph: {base_count:,} triples (window <= {MAX_TIME}s)")

    print(f"[evolve] rebuilding over the larger window <= {EVOLVE_TIME}s ...")
    build(max_time=EVOLVE_TIME, triples_path=TRIPLES_PATH)
    new_count = _count_lines(TRIPLES_PATH)
    print(f"[evolve] graph grew by {new_count - base_count:,} triples "
          f"({base_count:,} -> {new_count:,})")

    print("[evolve] re-running the reasoner on the evolved graph ...")
    reason()
    print("[evolve] retraining / re-scoring on the evolved graph ...")
    train()
    score()
    evaluate()


def _count_lines(path: Path) -> int:
    with Path(path).open() as f:
        return sum(1 for _ in f)


STEP_FUNCS = {
    "build": build,
    "reason": reason,
    "train": train,
    "score": score,
    "evaluate": evaluate,
    "evolve": evolve,
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
        choices=list(ALL_STEPS) + ["evolve"],
        default=list(ALL_STEPS),
        help="Subset of steps to run, in order. The default runs build → reason "
             "→ train → score → evaluate. 'evolve' (LO8) is opt-in and rebuilds "
             "the graph over a larger time window before re-detecting.",
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
