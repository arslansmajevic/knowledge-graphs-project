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
metrics (ROC-AUC, average precision and the score distributions). All artefacts
are written to `generated-files/` and the trained model to `pykeen-lanl-model/`.

### What the pipeline does

`pipeline.py` runs four steps in order:

1. **build** – parses `auth/proc/flows/dns` into a deduplicated set of
   knowledge-graph triples (`generated-files/triples.tsv`). By default only the
   first day of events is used to keep the graph small (see `MAX_TIME` in
   `pipeline.py`).
2. **train** – trains a PyKEEN embedding model (`TransE` by default) on the
   triples and saves it to `pykeen-lanl-model/`.
3. **score** – scores the red-team triples and a random sample of normal
   triples, writing `generated-files/redteam_scores.csv` and
   `generated-files/normal_scores.csv`.
4. **evaluate** – treats the model score as an anomaly score (lower plausibility
   = more anomalous) and reports ROC-AUC, average precision and per-class score
   statistics.

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
