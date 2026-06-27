"""Tests for MITRE ATT&CK ingestion (LO4, LO7)."""

from __future__ import annotations

import mitre


def test_subset_loads_with_expected_shape():
    attack = mitre.load_attack()
    assert attack["techniques"], "subset must define techniques"
    ids = {t["id"] for t in attack["techniques"]}
    assert {"T1078", "T1021"} <= ids
    for tech in attack["techniques"]:
        assert tech["log_signals"], f"{tech['id']} must map to >=1 log signal"


def test_attack_triples_link_to_log_signals():
    triples = mitre.attack_triples()
    rels = {r for _, r, _ in triples}
    assert mitre.REL_DETECTED_VIA in rels
    # T1021 Remote Services must be linked to the authenticates_to log signal,
    # which is exactly how the ATT&CK KB connects to the log-derived graph.
    assert (
        mitre.technique_node("T1021"),
        mitre.REL_DETECTED_VIA,
        mitre.signal_node("authenticates_to"),
    ) in triples


def test_subtechnique_relations_present():
    triples = set(mitre.attack_triples())
    assert (
        mitre.technique_node("T1021.001"),
        mitre.REL_SUBTECHNIQUE_OF,
        mitre.technique_node("T1021"),
    ) in triples


def test_signal_to_techniques_inverts_mapping():
    mapping = mitre.signal_to_techniques()
    assert "T1021" in mapping["authenticates_to"]
    assert "T1078" in mapping["logs_on_to"]


def test_write_attack_triples(tmp_path):
    out = tmp_path / "mitre.tsv"
    n = mitre.write_attack_triples(out)
    assert n > 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == n
    assert all(len(line.split("\t")) == 3 for line in lines)
