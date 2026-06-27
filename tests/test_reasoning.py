"""Tests for the bundled Datalog engine and the MITRE rule set (LO2, LO6)."""

from __future__ import annotations

import reasoning as R
from reasoning import Rule, Var, Skolem, evaluate


def test_engine_full_recursion_transitive_closure():
    """A classic recursive Datalog program: transitive closure of an edge."""
    A, B, C = Var("A"), Var("B"), Var("C")
    rules = [
        Rule([("path", (A, B))], [("edge", (A, B))]),
        Rule([("path", (A, C))], [("edge", (A, B)), ("path", (B, C))]),
    ]
    facts = {"edge": {("a", "b"), ("b", "c"), ("c", "d")}}
    out = evaluate(rules, facts)
    assert out["path"] == {
        ("a", "b"), ("b", "c"), ("c", "d"),
        ("a", "c"), ("b", "d"), ("a", "d"),
    }


def test_engine_recursion_terminates_on_cycles():
    A, B, C = Var("A"), Var("B"), Var("C")
    rules = [
        Rule([("path", (A, B))], [("edge", (A, B))]),
        Rule([("path", (A, C))], [("edge", (A, B)), ("path", (B, C))]),
    ]
    facts = {"edge": {("a", "b"), ("b", "a")}}  # a cycle
    out = evaluate(rules, facts)
    # Set semantics => fixpoint is finite even though the graph has a cycle.
    assert ("a", "a") in out["path"]
    assert ("a", "b") in out["path"]


def test_engine_object_creation_is_deterministic():
    """Skolem head terms mint a fresh, deterministic constant (existential)."""
    A, B = Var("A"), Var("B")
    new_obj = Skolem("obj", (A, B))
    rules = [Rule([("made", (new_obj, A))], [("pair", (A, B))])]
    facts = {"pair": {("x", "y")}}
    out = evaluate(rules, facts)
    assert ("obj(x|y)", "x") in out["made"]
    # Re-running yields the same object id (no infinite invention of objects).
    out2 = evaluate(rules, out)
    assert out2["made"] == out["made"]


def test_max_depth_bounds_recursion():
    A, B, C = Var("A"), Var("B"), Var("C")
    rules = [
        Rule([("path", (A, B))], [("edge", (A, B))]),
        Rule([("path", (A, C))], [("edge", (A, B)), ("path", (B, C))]),
    ]
    facts = {"edge": {("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")}}
    shallow = evaluate(rules, facts, max_iterations=2)
    full = evaluate(rules, facts, max_iterations=100)
    assert ("a", "e") in full["path"]
    assert ("a", "e") not in shallow["path"]  # depth-bounded


def _lateral_facts():
    # u signs on at c1; c1 -> c2 -> c3 via remote auth.
    return {
        R.EDB_AUTHENTICATES_TO: {("computer:c1", "computer:c2"),
                                 ("computer:c2", "computer:c3")},
        R.EDB_USES_SOURCE_COMPUTER: {("user:u", "computer:c1")},
        R.EDB_LOGS_ON_TO: {("user:u", "computer:c2")},
    }


def test_mitre_rules_derive_lateral_movement_chain():
    derived = R.run(_lateral_facts())
    lm = derived[R.REL_LATERAL_MOVEMENT]
    # Direct and transitive (multi-hop) lateral movement is derived.
    assert ("computer:c1", "computer:c2") in lm
    assert ("computer:c1", "computer:c3") in lm  # 2 hops => recursion


def test_mitre_rules_mint_attack_chain_object():
    derived = R.run(_lateral_facts())
    chains = R.attack_chains(derived)
    assert chains, "expected at least one minted attack_chain object"
    members = set().union(*(set(c["members"]) for c in chains))
    assert "user:u" in members
    assert any(c["technique"] == "technique:T1021" for c in chains)


def test_flagged_entities_and_query():
    derived = R.run(_lateral_facts())
    flagged = R.flagged_entities(derived)
    assert "computer:c3" in flagged
    hits = R.chains_for_entity(derived, "computer:c3")
    assert hits and all("computer:c3" in c["members"] for c in hits)


def test_credential_use_rule_maps_to_t1078():
    derived = R.run(_lateral_facts())
    assert ("user:u", "computer:c2") in derived[R.REL_CREDENTIAL_USE]
    assert R.DERIVED_TECHNIQUE[R.REL_CREDENTIAL_USE] == "T1078"
