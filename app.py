"""Streamlit dashboard — services layer over the Knowledge Graph (LO11, LO9).

This is the "services" end of the KG lifecycle: a small Security-Operations-
Centre (SOC) style console that surfaces what the hybrid detector found and lets
an analyst *query the graph* for attack chains.  It reads only the artifacts the
pipeline already writes to ``generated-files/`` so it needs no GPU and no raw
logs.

Run it with::

    streamlit run app.py

Panels:

* **Overview** – how big the graph is and what the reasoner derived.
* **Detections** – per-model KGE-only vs. hybrid ROC-AUC / average precision and
  the score distributions, plus the top-ranked anomalies.
* **Attack chains** – browse the minted ``attack_chain`` objects and run the
  "show attack chains involving computer X" query against the graph.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import hybrid
import mitre
import reasoning

GEN_DIR = Path("generated-files")
TRIPLES_PATH = GEN_DIR / "triples.tsv"
MITRE_TRIPLES_PATH = GEN_DIR / "mitre_triples.tsv"
DERIVED_TRIPLES_PATH = GEN_DIR / "derived_triples.tsv"
FLAGGED_PATH = GEN_DIR / "flagged_entities.txt"
CHAINS_PATH = GEN_DIR / "attack_chains.json"

MODELS = ["TransE", "DistMult", "RotatE", "ComplEx"]
HYBRID_LAMBDA = 1.0


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


@st.cache_data(show_spinner=False)
def _load_flagged() -> set[str]:
    return reasoning.load_flagged(FLAGGED_PATH)


@st.cache_data(show_spinner=False)
def _load_chains() -> list[dict]:
    return reasoning.load_chains(CHAINS_PATH)


@st.cache_data(show_spinner=False)
def _load_scores(model: str) -> pd.DataFrame | None:
    normal = GEN_DIR / f"normal_scores_{model}.csv"
    red = GEN_DIR / f"redteam_scores_{model}.csv"
    if not (normal.exists() and red.exists()):
        return None
    return hybrid.labelled_frame(pd.read_csv(normal), pd.read_csv(red))


def _available_models() -> list[str]:
    return [m for m in MODELS if _load_scores(m) is not None]


def overview_panel() -> None:
    st.header("Knowledge graph overview")
    if not TRIPLES_PATH.exists():
        st.warning(
            "No graph found. Run `python pipeline.py` first to build the graph "
            "and produce the artifacts this dashboard reads from "
            "`generated-files/`."
        )
        return
    cols = st.columns(4)
    cols[0].metric("Log triples", f"{_count_lines(TRIPLES_PATH):,}")
    cols[1].metric("MITRE ATT&CK triples", f"{_count_lines(MITRE_TRIPLES_PATH):,}")
    cols[2].metric("Derived triples", f"{_count_lines(DERIVED_TRIPLES_PATH):,}")
    cols[3].metric("Attack chains", f"{len(_load_chains()):,}")

    flagged = _load_flagged()
    st.caption(
        f"The symbolic reasoner flagged **{len(flagged):,}** entities as part of "
        "a lateral-movement / attack chain. These drive the hybrid detector."
    )


def detections_panel() -> None:
    st.header("Detections — KGE-only vs. hybrid")
    models = _available_models()
    if not models:
        st.info("No score files yet. Run `python pipeline.py --steps score evaluate`.")
        return

    flagged = _load_flagged()
    rows = []
    for model in models:
        df = _load_scores(model)
        res = hybrid.evaluate_frame(df, flagged, lam=HYBRID_LAMBDA)
        rows.append(
            {
                "model": model,
                "KGE ROC-AUC": round(res["kge"]["roc_auc"], 4),
                "Hybrid ROC-AUC": round(res["hybrid"]["roc_auc"], 4),
                "Δ ROC-AUC": round(res["roc_auc_delta"], 4),
                "KGE AP": round(res["kge"]["average_precision"], 4),
                "Hybrid AP": round(res["hybrid"]["average_precision"], 4),
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("model"))

    model = st.selectbox("Inspect a model", models)
    df = _load_scores(model)
    df = df.assign(
        kge_anomaly=hybrid.kge_anomaly(df),
        hybrid_anomaly=hybrid.hybrid_anomaly(df, flagged, lam=HYBRID_LAMBDA),
        klass=df["label"].map({0: "normal", 1: "red-team"}),
    )

    st.subheader("Anomaly-score distribution")
    st.vega_lite_chart(
        df,
        {
            "mark": "bar",
            "encoding": {
                "x": {"field": "hybrid_anomaly", "bin": {"maxbins": 40}, "title": "hybrid anomaly score"},
                "y": {"aggregate": "count", "title": "triples"},
                "color": {"field": "klass", "title": "class"},
            },
        },
    )

    st.subheader("Top-ranked anomalies (hybrid)")
    top = df.sort_values("hybrid_anomaly", ascending=False).head(25)
    st.dataframe(
        top[["head", "relation", "tail", "klass", "kge_anomaly", "hybrid_anomaly"]],
    )


def chains_panel() -> None:
    st.header("Attack chains")
    chains = _load_chains()
    if not chains:
        st.info("No attack chains yet. Run `python pipeline.py --steps reason`.")
        return

    names = mitre.technique_names()
    st.caption(
        "Each chain is an `attack_chain` object the reasoner *created* (object "
        "creation / existential quantification) for a lateral-movement path."
    )

    query = st.text_input(
        "Show attack chains involving entity (e.g. `computer:C123` or `user:U45@DOM1`)",
        "",
    ).strip()
    shown = [c for c in chains if not query or query in c["members"]]
    st.write(f"{len(shown):,} / {len(chains):,} chains")
    for chain in shown[:200]:
        tech_id = (chain.get("technique") or "").replace("technique:", "")
        tech = f"{tech_id} {names.get(tech_id, '')}".strip()
        with st.expander(f"{chain['id']}  —  {tech or 'unmapped'}"):
            st.write("**Members:**", ", ".join(chain["members"]))


def main() -> None:
    st.set_page_config(page_title="KG APT Detection", layout="wide")
    st.title("Knowledge-Graph APT Detection — SOC console")
    st.caption(
        "Hybrid (logical + KG-embedding) anomaly detection over the LANL "
        "cyber-security knowledge graph."
    )
    tab_overview, tab_detect, tab_chains = st.tabs(
        ["Overview", "Detections", "Attack chains"]
    )
    with tab_overview:
        overview_panel()
    with tab_detect:
        detections_panel()
    with tab_chains:
        chains_panel()


if __name__ == "__main__":
    main()
