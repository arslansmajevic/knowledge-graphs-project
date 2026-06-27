"""MITRE ATT&CK knowledge-base integration (LO4, LO7).

This module ingests a slice of the `MITRE ATT&CK
<https://attack.mitre.org/>`_ Enterprise knowledge base and turns it into
knowledge-graph triples that share the same vocabulary as the log-derived graph
produced by :mod:`pipeline`.  Two things make this interesting for the project:

* **Data integration / schema mapping (LO7).**  ATT&CK is published as
  semantic-web-style STIX 2.1 JSON (objects with global IDs and typed
  relationships), whereas the LANL logs are a property-graph-like stream of
  events.  Bringing them into one graph is a concrete schema-mapping exercise:
  every ATT&CK technique is mapped to the log *relation* (signal) it can be
  detected through (e.g. ``T1021 Remote Services`` -> ``authenticates_to``).

* **Comparing data models (LO4).**  Because both sources end up as triples we
  can talk about the trade-offs of the underlying models — see
  ``ARCHITECTURE.md`` for the written comparison that this code backs up with
  two *real* sources.

The full ATT&CK corpus is large, so we bundle a curated subset in
``data/mitre_attack_subset.json`` that works fully offline.  :func:`load_attack`
can optionally fetch the live STIX bundle and falls back to the bundled subset
when the network is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent / "data"
ATTACK_SUBSET_PATH = DATA_DIR / "mitre_attack_subset.json"

# Public STIX 2.1 bundle published by MITRE.  Used opportunistically by
# ``load_attack(prefer_live=True)``; the bundled subset is always the fallback.
ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# Relations (knowledge-graph predicates) introduced by ATT&CK ingestion.  They
# are namespaced with ``mitre_`` so they never collide with the log relations.
REL_TECHNIQUE_OF_TACTIC = "mitre_technique_of_tactic"
REL_SUBTECHNIQUE_OF = "mitre_subtechnique_of"
REL_DETECTED_VIA = "mitre_detected_via"


def _node(prefix: str, value: str) -> str:
    """Namespace an entity id the same way :mod:`pipeline` does (``kind:value``)."""
    return f"{prefix}:{value}"


def technique_node(technique_id: str) -> str:
    return _node("technique", technique_id)


def tactic_node(tactic_id: str) -> str:
    return _node("tactic", tactic_id)


def signal_node(relation: str) -> str:
    """Entity standing for a log signal (a relation produced by ``build``)."""
    return _node("signal", relation)


def load_attack(path: Path | str | None = None, prefer_live: bool = False) -> dict:
    """Return the ATT&CK subset as a ``dict``.

    Parameters
    ----------
    path:
        Override the bundled subset location (mainly for tests).
    prefer_live:
        When ``True`` attempt to download the full STIX bundle first and only
        fall back to the bundled subset on any failure (network blocked, parse
        error, ...).  Defaults to ``False`` so runs are deterministic/offline.
    """
    if prefer_live:
        live = _try_load_live()
        if live is not None:
            return live
    p = Path(path) if path is not None else ATTACK_SUBSET_PATH
    with p.open() as f:
        return json.load(f)


def _try_load_live() -> dict | None:
    """Best-effort fetch + normalisation of the live STIX bundle.

    Returns ``None`` (so the caller falls back to the bundled subset) if the
    download fails for any reason, which is the common case in sandboxed or
    offline environments.
    """
    try:  # pragma: no cover - network dependent, exercised only when reachable
        import urllib.request

        with urllib.request.urlopen(ATTACK_STIX_URL, timeout=15) as resp:
            bundle = json.load(resp)
        return _normalise_stix(bundle)
    except Exception:  # pragma: no cover - offline is the expected sandbox case
        return None


def _normalise_stix(bundle: dict) -> dict:
    """Reduce a full STIX bundle to our compact ``{tactics, techniques}`` schema.

    Only the fields we need are kept.  ``log_signals`` cannot be inferred from
    STIX, so live techniques are matched back to the bundled subset to recover
    the signal mapping; techniques without a known mapping are dropped (they
    cannot be linked to the log graph anyway).
    """
    subset = load_attack(prefer_live=False)
    signal_by_id = {t["id"]: t["log_signals"] for t in subset["techniques"]}

    tactics: dict[str, dict] = {}
    techniques: list[dict] = []
    for obj in bundle.get("objects", []):
        if obj.get("type") == "x-mitre-tactic":
            ext = _external_id(obj)
            if ext:
                tactics[ext] = {"id": ext, "name": obj.get("name", ext)}
        elif obj.get("type") == "attack-pattern":
            ext = _external_id(obj)
            if ext and ext in signal_by_id:
                techniques.append(
                    {
                        "id": ext,
                        "name": obj.get("name", ext),
                        "tactic": None,
                        "parent": ext.split(".")[0] if "." in ext else None,
                        "description": obj.get("description", ""),
                        "log_signals": signal_by_id[ext],
                    }
                )
    if not techniques:
        raise ValueError("no usable techniques found in STIX bundle")
    return {
        "source": "MITRE ATT&CK Enterprise (live STIX)",
        "version": bundle.get("spec_version", "stix-2.1"),
        "tactics": list(tactics.values()),
        "techniques": techniques,
    }


def _external_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"]
    return None


def attack_triples(attack: dict | None = None) -> list[tuple[str, str, str]]:
    """Turn the ATT&CK knowledge base into ``(head, relation, tail)`` triples.

    The resulting triples connect the ATT&CK knowledge to the *same* log
    signals used by the embedding graph, so the two data models live in one
    knowledge graph.
    """
    attack = attack or load_attack()
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(h: str, r: str, t: str) -> None:
        tpl = (h, r, t)
        if tpl not in seen:
            seen.add(tpl)
            triples.append(tpl)

    for tech in attack["techniques"]:
        tnode = technique_node(tech["id"])
        if tech.get("tactic"):
            add(tnode, REL_TECHNIQUE_OF_TACTIC, tactic_node(tech["tactic"]))
        if tech.get("parent"):
            add(tnode, REL_SUBTECHNIQUE_OF, technique_node(tech["parent"]))
        for signal in tech.get("log_signals", []):
            # Link technique <-> the log relation it is detected through.
            add(tnode, REL_DETECTED_VIA, signal_node(signal))
    return triples


def signal_to_techniques(attack: dict | None = None) -> dict[str, list[str]]:
    """Map each log relation to the ATT&CK technique ids that detect it."""
    attack = attack or load_attack()
    mapping: dict[str, list[str]] = {}
    for tech in attack["techniques"]:
        for signal in tech.get("log_signals", []):
            mapping.setdefault(signal, []).append(tech["id"])
    return mapping


def technique_names(attack: dict | None = None) -> dict[str, str]:
    attack = attack or load_attack()
    return {t["id"]: t["name"] for t in attack["techniques"]}


def write_attack_triples(path: Path | str, attack: dict | None = None) -> int:
    """Write ATT&CK triples to ``path`` (TSV) and return the number written."""
    triples = attack_triples(attack)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    return len(triples)


def append_triples(path: Path | str, triples: Iterable[tuple[str, str, str]]) -> int:
    """Append ``triples`` to an existing TSV file, returning the count appended."""
    triples = list(triples)
    with Path(path).open("a") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    return len(triples)
