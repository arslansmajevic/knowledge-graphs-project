# Architecture & Learning-Outcome Map

This document describes the hybrid (symbolic + sub-symbolic) Knowledge-Graph
architecture of the project and maps every component to the course learning
outcomes (LOs). It is the written backing for **LO5** (designing a KG
architecture), **LO9** (real-world application), and the data-model comparison
of **LO4**.

## 1. The big picture

The project is a **Security-Operations-Centre (SOC)** application: it ingests
enterprise security logs, represents them as a Knowledge Graph, and detects
Advanced-Persistent-Threat (APT) lateral movement by combining *logical
reasoning* with *knowledge-graph embeddings*. This is exactly the
melting-pot idea behind **LO12** — traditional logic-based reasoning and
ML-based (embedding) reasoning operating over one shared graph.

```
                    ┌──────────────────────────────────────────────────────┐
                    │                   Knowledge Graph                      │
                    │            (RDF-style triples, one vocabulary)         │
                    └──────────────────────────────────────────────────────┘
                                          ▲
   raw logs                               │ triples
 ┌───────────┐   build    ┌───────────────┴───────────────┐
 │ LANL logs │──────────► │ log triples (auth/proc/flows/  │  property-graph-
 │ auth/proc │            │ dns) — generated-files/        │  like event stream
 │ flows/dns │            │ triples.tsv                    │
 └───────────┘            └───────────────┬───────────────┘
                                          │
 ┌───────────┐   build    ┌───────────────┴───────────────┐
 │ MITRE     │──────────► │ ATT&CK triples — techniques    │  semantic-web-
 │ ATT&CK    │  (mitre.py)│ linked to log signals          │  like STIX/RDF KB
 └───────────┘            └───────────────┬───────────────┘
                                          │
                                          ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  reason (reasoning.py) — Datalog reasoner                         │
        │   • full recursion: lateral_movement = transitive closure        │  LO2, LO6
        │   • object creation: mint attack_chain entities (existential)    │
        │   → derived_triples.tsv, flagged_entities.txt, attack_chains.json │
        └───────────────┬─────────────────────────────────────┬───────────┘
                        │ derived triples (provenance)         │ flagged entities
                        ▼                                       │ (logical signal)
        ┌───────────────────────────────────┐                  │
        │  train (PyKEEN) — KG embeddings    │  LO1             │
        │  TransE / DistMult / RotatE /      │                  │
        │  ComplEx over the combined graph   │                  │
        └───────────────┬───────────────────┘                  │
                        │ plausibility scores                   │
                        ▼                                       ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  score + evaluate (hybrid.py) — combine KGE anomaly score with    │  LO2, LO8,
        │  the logical flag → KGE-only vs hybrid ROC-AUC / AP               │  LO12
        └───────────────┬─────────────────────────────────────────────────┘
                        ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  app.py — Streamlit SOC console (overview / detections / chains) │  LO9, LO11
        └─────────────────────────────────────────────────────────────────┘
```

## 2. Which capability lives where (LO5)

Designing the architecture is largely about deciding *what the Knowledge Graph
handles and what external code handles*:

| Capability | Where it lives | Why |
|---|---|---|
| Symbolic storage of events | `triples.tsv` (the KG) | shared substrate both halves read |
| Recursive reasoning, object creation | `reasoning.py` (Datalog) | declarative, decidable, full recursion |
| Sub-symbolic generalisation | PyKEEN models (`train`) | learns plausibility from structure |
| Combining the two | `hybrid.py` | thin, auditable fusion of the signals |
| Services (query, visualise, alert) | `app.py` (Streamlit) | application code, *not* the KG |

The boundary follows the LO5 principle: **the KG stores and reasons; the
application code queries and presents.** Large-scale storage and the recursive
reasoning could be pushed into a dedicated KG system (see §4); here they are
kept lightweight so the project runs end-to-end on a laptop or a free Colab GPU.

## 3. Two data models in one graph (LO4)

The project deliberately integrates two *different* data models so they can be
compared with real sources rather than in the abstract:

| | LANL logs | MITRE ATT&CK |
|---|---|---|
| Origin community | databases / data science | semantic web |
| Native shape | property-graph-like event stream | STIX 2.1 / RDF-style typed objects |
| Identity | local (`computer:C123`) | global, curated IDs (`technique:T1021`) |
| Schema | implicit, derived from columns | explicit, externally governed |
| Volume | huge, fast-changing | small, slow-changing |

`mitre.py` performs the **schema mapping** (LO7) that brings them together:
every ATT&CK technique is linked (`mitre_detected_via`) to the *log relation* it
is observed through (e.g. `T1021 Remote Services → authenticates_to`). Once both
are triples over one vocabulary, the embedding model can learn over the union
and the reasoner can use technique annotations.

## 4. Engine choice & the Vadalog connection (LO5)

The one-pager proposed expressing the rules in **Datalog / Vadalog** and running
them on the **Vadalog system** (Oxford / TU Wien / Banca d'Italia). Vadalog is
the natural production target because it is built precisely for KGs that need
*full recursion* **and** *existential quantification* (Warded Datalog±). It is,
however, **gated** — not freely pip-installable — so it cannot be the runnable
default in this repository.

Engines evaluated as the runnable fallback:

| Engine | Recursion | Object creation | Availability | Verdict |
|---|---|---|---|---|
| Vadalog | ✅ (Warded Datalog±) | ✅ existential | gated / not public | production target |
| pyDatalog | ✅ | ✗ (no Skolem) | pip, but unmaintained | works on 3.12, fragile |
| clingo / ASP | ✅ | partial (via terms) | pip (`clingo` 5.x) | heavier paradigm |
| SQLite recursive CTE | ✅ | ✗ (manual Skolem) | stdlib | DB-community option |
| **bundled engine** (`reasoning.py`) | ✅ | ✅ Skolem terms | none needed | **default here** |

The bundled ~200-line engine was chosen because it needs no extra dependency, is
fully transparent (the rules in the report are exactly the rules that run), and
demonstrates the two features the lecture highlights for KGs — **full recursion**
(lateral-movement transitive closure) and **object creation / existential
quantification** (minting `attack_chain` entities). The *same rules* could be
lifted to Vadalog unchanged.

## 5. Knowledge-Graph evolution (LO8)

`pipeline.py --steps evolve` demonstrates the **completion** side of KG
evolution: it rebuilds the graph over a *later* time window (`EVOLVE_TIME`),
reports how many triples were added, then re-runs the reasoner and re-scores.
Two completion mechanisms are shown side by side — **link prediction** via KG
embeddings (LO1/LO8) and **logical inference** of new facts (lateral movement,
attack chains) via the recursive rules.

## 6. Real-world framing (LO9)

The whole pipeline mirrors a SOC workflow: telemetry ingestion → correlation
into an attack narrative → analyst triage. The Streamlit console (`app.py`)
is the analyst-facing service: it ranks anomalies, shows the KGE-vs-hybrid
detection quality, and answers the operational question *"show me the attack
chains involving host X."* This is the LO11 services layer (structured query +
visualisation + alerting) grounded in a concrete LO9 application.

## 7. Deliberate scope exclusions (LO3, LO10)

Consistent with the one-pager, two LOs are **intentionally out of scope** so the
project stays focused:

* **LO3 (Graph Neural Networks).** The sub-symbolic half uses KG *embeddings*
  (PyKEEN), not GNN message passing. GNNs would be a natural extension (the same
  graph could feed a GNN-based link predictor) but are not implemented.
* **LO10 (Financial KGs).** The domain is cyber-security (LANL + MITRE), not
  banking / insurance / central-banking data.

Stating these exclusions explicitly is itself part of a good architecture
write-up: scope is a design decision, not an omission.

## 8. LO coverage summary

| LO | Covered by |
|---|---|
| LO1 Knowledge Graph Embeddings | PyKEEN `train` (TransE/DistMult/RotatE/ComplEx) |
| LO2 Logical knowledge in KGs | `reasoning.py` recursive + existential rules |
| LO3 Graph Neural Networks | *excluded (see §7)* |
| LO4 Compare KG data models | logs vs MITRE (§3) |
| LO5 Design KG architecture | this document |
| LO6 Scalable reasoning | recursive Datalog reasoning (`reason`) |
| LO7 Create a KG | `build` + MITRE schema mapping (`mitre.py`) |
| LO8 Evolve a KG | `evolve` step + link prediction |
| LO9 Real-world applications | SOC framing + dashboard |
| LO10 Financial KG applications | *excluded (see §7)* |
| LO11 Services through a KG | Streamlit console (`app.py`) |
| LO12 KG ↔ ML ↔ AI connections | hybrid logic+embedding detector (`hybrid.py`) |
