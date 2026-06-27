# Knowledge-Graph APT Detection on the LANL Dataset

This project turns the [LANL "Comprehensive, Multi-Source Cyber-Security Events"](https://csr.lanl.gov/data/cyber1/)
logs into a knowledge graph and detects Advanced-Persistent-Threat (APT)
lateral movement with a **hybrid** approach that combines two kinds of
reasoning over the same graph:

* a **symbolic** half — a small Datalog reasoner that encodes MITRE ATT&CK /
  cyber-kill-chain rules (lateral movement, attack chains), and
* a **sub-symbolic** half — knowledge-graph embeddings (PyKEEN) that score how
  *plausible* known red-team (malicious) events are versus normal activity.

Implausible events are flagged as anomalies, and the logical reasoner's output
both *feeds into* the embedding graph and *combines with* the embedding score to
form the final hybrid detector.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full architecture, the engine
choice (and how it relates to Vadalog), the two-data-model comparison, and the
mapping of every component to the course learning outcomes (LO1–LO12).

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

## Run it faster on a free GPU (Google Colab)

Training and evaluation are much faster on a GPU. If your local machine only has
a CPU, run the pipeline on a free Colab GPU instead — PyKEEN auto-detects CUDA,
so no code changes are needed.

1. Open [`colab.ipynb`](colab.ipynb) in Google Colab
   ([colab.research.google.com](https://colab.research.google.com/) → *File →
   Open notebook → GitHub*, or upload the file).
2. Enable the GPU: **Runtime → Change runtime type → Hardware accelerator → GPU**.
3. Run the cells in order. The notebook clones the repo, installs the
   requirements, helps you make the LANL dataset available (via Google Drive, since
   the raw logs are not committed), and then runs `python pipeline.py` on the GPU.

The notebook also shows how to run individual steps and how to copy the results
back to Drive so they survive Colab's ephemeral sessions.


### What the pipeline does

`pipeline.py` runs five steps in order:

1. **build** – parses `auth/proc/flows/dns` into a deduplicated set of
   knowledge-graph triples (`generated-files/triples.tsv`) and ingests the
   **MITRE ATT&CK** knowledge base (`generated-files/mitre_triples.tsv`),
   linking each technique to the log signal it is detected through. By default
   only the first day of events is used to keep the graph small (see `MAX_TIME`
   in `pipeline.py`).
2. **reason** – runs the symbolic reasoner (`reasoning.py`): a small Datalog
   engine that applies MITRE-mapped rules with *full recursion* (lateral-movement
   transitive closure) and *object creation* (minting `attack_chain` entities).
   It writes the derived triples (`derived_triples.tsv`), the flagged-entity
   logical signal (`flagged_entities.txt`), and the materialised chains
   (`attack_chains.json`).
3. **train** – trains the configured PyKEEN embedding models (`TransE`,
   `DistMult`, `RotatE`, `ComplEx` by default) on the combined graph (log +
   MITRE + derived triples) and saves each one to `pykeen-lanl-model/<model>/`.
4. **score** – scores the red-team triples and a random sample of normal
   triples with every trained model, writing
   `generated-files/redteam_scores_<model>.csv` and
   `generated-files/normal_scores_<model>.csv`.
5. **evaluate** – treats each model score as an anomaly score (lower
   plausibility = more anomalous) and reports **both** the pure-KGE baseline and
   the **hybrid** detector (KGE anomaly score + the reasoner's logical flag),
   with a side-by-side ROC-AUC / average-precision comparison per model.

An optional **evolve** step (`python pipeline.py --steps evolve`) demonstrates
Knowledge-Graph evolution (LO8): it rebuilds the graph over a larger time window
(`EVOLVE_TIME`), reports how many triples were added, and re-runs the reasoner,
training, scoring and evaluation on the evolved graph.

### The SOC dashboard

A Streamlit console (`app.py`) is the services layer over the graph. After
running the pipeline, launch it with:

```bash
streamlit run app.py
```

It reads only the artifacts in `generated-files/` (no GPU or raw logs needed)
and provides three tabs: an **overview** of the graph, **detections**
(KGE-only vs hybrid metrics, score distributions, top-ranked anomalies), and
**attack chains** — including the query *"show attack chains involving computer
X."*

### Tests

The logical reasoner, MITRE ingestion and hybrid scoring are covered by tests
that run without the dataset or PyKEEN:

```bash
pytest
```

### Running individual steps

Steps can be run on their own (in order), which is handy when iterating without
retraining:

```bash
python pipeline.py --steps build
python pipeline.py --steps reason
python pipeline.py --steps train
python pipeline.py --steps score evaluate
```

`reason` only needs `build` to have run; `train` picks up the derived and MITRE
triples automatically if they exist (toggle with `INCLUDE_DERIVED_TRIPLES` /
`INCLUDE_MITRE_TRIPLES`). See all options with `python pipeline.py --help`.
Model and graph settings (embedding dimension, epochs, sample size, time window,
hybrid weight, ...) live as constants at the top of `pipeline.py`.

### Choosing which models to train

The `train`/`score`/`evaluate` steps run over every model listed in the
`MODELS` constant at the top of `pipeline.py`:

```python
MODELS = ["TransE", "DistMult", "RotatE", "ComplEx"]
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
  The eval batch size avoids PyKEEN's very conservative automatic CPU batch size.

The trained model is unaffected by `EVAL_SAMPLE_SIZE` — it only changes how many
triples are used to *measure* quality, so the downstream `score`/`evaluate`
steps still use the full model.

### GPU acceleration

The `train` step auto-detects CUDA and, when a GPU is present, uses it
automatically — no flags required. It also prints the selected device and tunes
the workload to it:

- **Device** – the model and PyKEEN pipeline run on `cuda` when a GPU is
  available, otherwise on `cpu`.
- **Batch size** – GPUs handle far larger batches than CPUs, so training and
  evaluation use bigger batches on a GPU (`GPU_BATCH_SIZE` / `GPU_EVAL_BATCH_SIZE`)
  and smaller ones on a CPU (`CPU_BATCH_SIZE` / `CPU_EVAL_BATCH_SIZE`). Larger
  batches keep the GPU busy and train faster; raise `GPU_BATCH_SIZE` further if
  your GPU has spare memory, or lower it if you hit out-of-memory errors.
- **Data loading** – on a GPU, `GPU_NUM_WORKERS` DataLoader workers plus pinned
  memory overlap batch preparation with compute so the GPU does not idle waiting
  for data.

All of these are constants at the top of `pipeline.py`. To run the whole thing
on a free GPU, see [Run it faster on a free GPU (Google Colab)](#run-it-faster-on-a-free-gpu-google-colab).

