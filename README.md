# Knowledge-Graph Anomaly Detection on the LANL Dataset

This project turns the [LANL "Comprehensive, Multi-Source Cyber-Security Events"](https://csr.lanl.gov/data/cyber1/)
logs into a knowledge graph, trains a graph-embedding model on it, and uses the
model to score how *plausible* known red-team (malicious) authentication events
are compared to normal activity. Implausible events are flagged as anomalies.

## The dataset

The raw logs are **not** committed to this repository because they are far too
large. Download them from LANL and place the (optionally gzipped) files in a
`dataset/` directory at the repository root:

```
dataset/
├── auth.txt      # authentication events
├── proc.txt      # process start/stop events
├── flows.txt     # network flow events
├── dns.txt       # DNS lookup events
└── redteam.txt   # ground-truth malicious events
```

Each file is comma-delimited and missing values are written as `?`:

| File          | Format |
|---------------|--------|
| `auth.txt`    | `time,src_user@domain,dst_user@domain,src_computer,dst_computer,auth_type,logon_type,orientation,success/failure` |
| `proc.txt`    | `time,user@domain,computer,process,start/end` |
| `flows.txt`   | `time,duration,src_computer,src_port,dst_computer,dst_port,protocol,packet_count,byte_count` |
| `dns.txt`     | `time,src_computer,resolved_computer` |
| `redteam.txt` | `time,user@domain,src_computer,dst_computer` (known compromises) |

## Setup

```bash
pip install -r requirements.txt
```

## Run everything with one command

```bash
python pipeline.py
```

That single command runs the whole project end to end and prints the detection
metrics (ROC-AUC, average precision and the score distributions). All artifacts
are written to `generated-files/` and the trained models to
`pykeen-lanl-model/<model>/`.

While it runs you'll see live progress: each step is announced with a
`[i/n] step` header and a timing summary, the build step prints `tqdm` progress
bars for each input file (auth/dns/flows/proc), and PyKEEN shows its own
training progress bars.

### What the pipeline does

`pipeline.py` runs four steps in order:

1. **build** – parses `auth/proc/flows/dns` into a deduplicated set of
   knowledge-graph triples (`generated-files/triples.tsv`). By default only the
   first day of events is used to keep the graph small (see `MAX_TIME` in
   `pipeline.py`).
2. **train** – trains the configured PyKEEN embedding models (`TransE` and
   `DistMult` by default) on the triples and saves each one to
   `pykeen-lanl-model/<model>/`.
3. **score** – scores the red-team triples and a random sample of normal
   triples with every trained model, writing
   `generated-files/redteam_scores_<model>.csv` and
   `generated-files/normal_scores_<model>.csv`.
4. **evaluate** – treats each model score as an anomaly score (lower
   plausibility = more anomalous), reports ROC-AUC, average precision and
   per-class score statistics per model, and prints a side-by-side comparison
   table ranking the models.

### Running individual steps

Steps can be run on their own (in order), which is handy when iterating without
retraining:

```bash
python pipeline.py --steps build
python pipeline.py --steps train
python pipeline.py --steps score evaluate
```

See all options with `python pipeline.py --help`. Model and graph settings
(embedding dimension, epochs, sample size, time window, ...) live as constants
at the top of `pipeline.py`.

### Choosing which models to train

The `train`/`score`/`evaluate` steps run over every model listed in the
`MODELS` constant at the top of `pipeline.py`:

```python
MODELS = ["TransE", "DistMult"]
```

Add or remove any model name from
[PyKEEN's model registry](https://pykeen.readthedocs.io/en/stable/reference/models.html)
(e.g. `"ComplEx"`, `"RotatE"`, `"TransH"`) and each will be trained, scored and
evaluated independently. The `evaluate` step prints a side-by-side comparison
table ranking the models by ROC-AUC. Every model uses `embedding_dim`
(`EMBEDDING_DIM`); to pass extra/override hyper-parameters to a specific model,
add an entry to the `MODEL_KWARGS` dict, e.g.
`MODEL_KWARGS = {"RotatE": {"embedding_dim": 128}}`.

### Keeping the run time reasonable

Two things dominate the run time, and both are tunable via constants at the top
of `pipeline.py`:

- **Graph size** – `MAX_TIME` keeps only the first day of events (set to `None`
  to use everything). A bigger window means more triples and slower training.
- **Evaluation** – PyKEEN's link-prediction evaluation ranks every held-out
  triple against all ~110k entities, which on CPU can take many hours for the
  full test split. `EVAL_SAMPLE_SIZE` (default `10_000`) evaluates on a random
  subsample instead, finishing in seconds while still giving a meaningful
  quality estimate; set it to `None` to evaluate on the full split.
  `EVAL_BATCH_SIZE` avoids PyKEEN's very conservative automatic CPU batch size.

The trained model is unaffected by `EVAL_SAMPLE_SIZE` — it only changes how many
triples are used to *measure* quality, so the downstream `score`/`evaluate`
steps still use the full model.

