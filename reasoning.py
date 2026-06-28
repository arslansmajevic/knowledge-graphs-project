"""Logical reasoning component (LO2, LO6).

The KGE pipeline (``pipeline.py``) is the *sub-symbolic* half of the project.
This module is the *symbolic* half: a small, self-contained **Datalog** engine
plus a set of rules that encode MITRE ATT&CK / cyber-kill-chain patterns over
the same knowledge-graph triples.

Why a bundled engine?
---------------------
The one-pager proposed expressing these rules in **Datalog / Vadalog**.  Vadalog
(Oxford/TU Wien) is the production target but is *gated* — it is not freely
pip-installable — so it cannot be the runnable default here.  We evaluated the
pip-installable alternatives (``pyDatalog`` works on Python 3.12 but is
unmaintained; ``clingo``/ASP and SQLite *recursive CTEs* are both available) and
chose to ship this ~200-line engine instead because it (a) needs no extra
dependency, (b) is fully transparent so the rules in the report are exactly the
rules that run, and (c) directly demonstrates the two features the lecture
highlights for Knowledge Graphs:

* **Full recursion** — ``lateral_movement`` is the transitive closure of remote
  authentication, exactly the recursion needed for graph reachability (LO6).
* **Object creation / existential quantification** — rules may *mint* brand-new
  entities (Skolem terms).  We use this to materialise an ``attack_chain`` object
  for every multi-hop lateral-movement path, i.e. we discover a previously
  unnamed part of the Knowledge Graph (LO2).

The same rules could be run unchanged on Vadalog; this engine is the portable
fallback (documented in ``ARCHITECTURE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Union
import time

# --------------------------------------------------------------------------- #
# A tiny Datalog engine
# --------------------------------------------------------------------------- #
# Terms are either a ``Var`` (logic variable), a plain ``str``/constant, or a
# ``Skolem`` functional term (only allowed in rule heads) that fabricates a new
# constant from its bound arguments -> this is how we create new objects.


@dataclass(frozen=True)
class Var:
    """A logic variable, e.g. ``Var("X")``."""

    name: str


@dataclass(frozen=True)
class Skolem:
    """A function term that mints a fresh constant from its arguments.

    ``Skolem("ac", (Var("A"), Var("B")))`` evaluated under a binding produces a
    deterministic id such as ``"ac(c1|c5)"``.  Because it is deterministic the
    same path always maps to the same object (set semantics, terminating), yet
    it is an object that did not exist in the input graph — existential
    quantification in logic terms.
    """

    functor: str
    args: tuple

    def ground(self, binding: dict) -> str:
        vals = [_resolve(a, binding) for a in self.args]
        return f"{self.functor}(" + "|".join(vals) + ")"


Term = Union[Var, Skolem, str]  # a constant, a logic variable, or a function term
Atom = tuple  # (predicate: str, terms: tuple[Term, ...])


@dataclass(frozen=True)
class Rule:
    """A Datalog rule ``heads :- body``.

    ``heads`` is a list of atoms (a conjunction in the head is sugar for several
    rules sharing one body — convenient when creating an object and several
    facts about it at once).  ``body`` is a list of positive atoms.
    """

    heads: list
    body: list


def _resolve(term: Term, binding: dict) -> str:
    if isinstance(term, Var):
        return binding[term.name]
    if isinstance(term, Skolem):
        return term.ground(binding)
    return term  # constant


Facts = dict  # predicate -> set[tuple[str, ...]]


def _build_index(facts: Facts) -> dict:
    """Index facts by ``(predicate, position) -> {value: [tuples]}`` for joins."""
    index: dict = {}
    for pred, rows in facts.items():
        per_pred: dict[int, dict] = {}
        for row in rows:
            for pos, val in enumerate(row):
                per_pred.setdefault(pos, {}).setdefault(val, []).append(row)
        index[pred] = per_pred
    return index


def _match_atom(atom: Atom, binding: dict, facts: Facts, index: dict) -> Iterator[dict]:
    pred, terms = atom
    rows = facts.get(pred)
    if not rows:
        return
    # Probe an index on the first already-bound position to avoid a full scan.
    probe_pos = None
    probe_val = None
    for pos, term in enumerate(terms):
        if isinstance(term, Var) and term.name in binding:
            probe_pos, probe_val = pos, binding[term.name]
            break
        if isinstance(term, str):
            probe_pos, probe_val = pos, term
            break
    if probe_pos is not None:
        candidates = index.get(pred, {}).get(probe_pos, {}).get(probe_val, ())
    else:
        candidates = rows
    for row in candidates:
        new_binding = _unify(terms, row, binding)
        if new_binding is not None:
            yield new_binding


def _unify(
    terms: tuple, row: tuple, binding: dict[str, str]
) -> dict[str, str] | None:
    if len(terms) != len(row):
        return None
    out = binding
    for term, val in zip(terms, row):
        if isinstance(term, Var):
            bound = out.get(term.name)
            if bound is None:
                if out is binding:
                    out = dict(binding)
                out[term.name] = val
            elif bound != val:
                return None
        else:  # constant must match
            if term != val:
                return None
    return out


def _eval_body(body: list, facts: Facts, index: dict) -> Iterator[dict]:
    def rec(i: int, binding: dict) -> Iterator[dict]:
        if i == len(body):
            yield binding
            return
        for nb in _match_atom(body[i], binding, facts, index):
            yield from rec(i + 1, nb)

    yield from rec(0, {})


def evaluate(rules: Iterable[Rule], facts: Facts, max_iterations: int = 100) -> Facts:
    rules = list(rules)
    derived: Facts = {p: set(rows) for p, rows in facts.items()}

    for iteration in range(1, max_iterations + 1):
        started = time.perf_counter()
        index = _build_index(derived)

        added = False
        additions = 0

        for rule in rules:
            for binding in _eval_body(rule.body, derived, index):
                for pred, terms in rule.heads:
                    new_row = tuple(_resolve(t, binding) for t in terms)
                    bucket = derived.setdefault(pred, set())
                    if new_row not in bucket:
                        bucket.add(new_row)
                        additions += 1
                        added = True

        counts = {
            pred: len(rows)
            for pred, rows in derived.items()
            if pred in IDB_RELATIONS
        }
        elapsed = time.perf_counter() - started
        print(
            f"[reason] iteration {iteration}/{max_iterations}: "
            f"+{additions:,} facts in {elapsed:.1f}s | {counts}"
        )

        if not added:
            break

    return derived


# --------------------------------------------------------------------------- #
# MITRE ATT&CK / cyber-kill-chain rules over the LANL knowledge graph
# --------------------------------------------------------------------------- #
# Derived relations (the IDB) and the ATT&CK technique each one evidences.
REL_REMOTE_AUTH = "remote_auth"            # T1021 Remote Services
REL_LATERAL_MOVEMENT = "lateral_movement"  # T1021 (recursive / transitive)
REL_CREDENTIAL_USE = "credential_use"      # T1078 Valid Accounts
REL_SUSPICIOUS_CHAIN = "suspicious_chain"  # user-driven lateral movement
REL_PART_OF = "part_of_attack_chain"       # links minted object -> members
REL_CHAIN_TECHNIQUE = "attack_chain_technique"
REL_CHAIN_LENGTH = "attack_chain_min_length"

# Which derived relation maps to which ATT&CK technique (for provenance/report).
DERIVED_TECHNIQUE = {
    REL_REMOTE_AUTH: "T1021",
    REL_LATERAL_MOVEMENT: "T1021",
    REL_CREDENTIAL_USE: "T1078",
}

# Relations consumed from the log graph (the EDB).
EDB_AUTHENTICATES_TO = "authenticates_to"
EDB_LOGS_ON_TO = "logs_on_to"
EDB_USES_SOURCE_COMPUTER = "uses_source_computer"


def mitre_rules() -> list[Rule]:
    """The rule set encoding ATT&CK lateral-movement patterns.

    Reads like Datalog; e.g. the recursive rule is::

        lateral_movement(A, C) :- remote_auth(A, B), lateral_movement(B, C).
    """
    A, B, C = Var("A"), Var("B"), Var("C")
    U = Var("U")
    chain = Skolem("ac", (U, A, C))
    return [
        # T1021 Remote Services: a remote authentication edge between hosts.
        Rule([(REL_REMOTE_AUTH, (A, B))], [(EDB_AUTHENTICATES_TO, (A, B))]),
        # T1078 Valid Accounts: a user logging on to a host with valid creds.
        Rule([(REL_CREDENTIAL_USE, (U, A))], [(EDB_LOGS_ON_TO, (U, A))]),
        # Lateral movement = transitive closure of remote auth (FULL RECURSION).
        Rule([(REL_LATERAL_MOVEMENT, (A, B))], [(REL_REMOTE_AUTH, (A, B))]),
        Rule(
            [(REL_LATERAL_MOVEMENT, (A, C))],
            [(REL_REMOTE_AUTH, (A, B)), (REL_LATERAL_MOVEMENT, (B, C))],
        ),
        # A user that reaches a host via multi-hop lateral movement from the
        # host they signed in on. This is the suspicious kill-chain pattern...
        Rule(
            [(REL_SUSPICIOUS_CHAIN, (U, A, C))],
            [
                (EDB_USES_SOURCE_COMPUTER, (U, A)),
                (REL_LATERAL_MOVEMENT, (A, C)),
            ],
        ),
        # ...and for each such pattern we MINT a new attack_chain object
        # (existential quantification / object creation) and record facts about
        # it: its members and the ATT&CK technique it evidences.
        Rule(
            [
                (REL_PART_OF, (chain, U)),
                (REL_PART_OF, (chain, A)),
                (REL_PART_OF, (chain, C)),
                (REL_CHAIN_TECHNIQUE, (chain, "technique:T1021")),
            ],
            [
                (EDB_USES_SOURCE_COMPUTER, (U, A)),
                (REL_LATERAL_MOVEMENT, (A, C)),
            ],
        ),
    ]


IDB_RELATIONS = (
    REL_REMOTE_AUTH,
    REL_LATERAL_MOVEMENT,
    REL_CREDENTIAL_USE,
    REL_SUSPICIOUS_CHAIN,
    REL_PART_OF,
    REL_CHAIN_TECHNIQUE,
)

# Only the *binary* derived relations can be exported as ``(h, r, t)`` triples
# and fed back into the embedding graph. ``suspicious_chain`` is ternary, so it
# is represented for downstream use through the minted attack-chain object and
# its ``part_of_attack_chain`` edges instead.
BINARY_IDB_RELATIONS = (
    REL_REMOTE_AUTH,
    REL_LATERAL_MOVEMENT,
    REL_CREDENTIAL_USE,
    REL_PART_OF,
    REL_CHAIN_TECHNIQUE,
)


# --------------------------------------------------------------------------- #
# Loading facts and running the reasoner
# --------------------------------------------------------------------------- #
def load_facts(triples_path: Path | str) -> Facts:
    """Load a ``triples.tsv`` file into an EDB ``Facts`` dict (by relation)."""
    facts: Facts = {}
    with Path(triples_path).open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            facts.setdefault(r, set()).add((h, t))
    return facts


def facts_from_triples(triples: Iterable[tuple[str, str, str]]) -> Facts:
    facts: Facts = {}
    for h, r, t in triples:
        facts.setdefault(r, set()).add((h, t))
    return facts


def run(facts: Facts, max_depth: int = 8) -> Facts:
    """Run rules only over facts used by the symbolic program."""
    relevant = {
        EDB_AUTHENTICATES_TO: set(facts.get(EDB_AUTHENTICATES_TO, ())),
        EDB_LOGS_ON_TO: set(facts.get(EDB_LOGS_ON_TO, ())),
        EDB_USES_SOURCE_COMPUTER: set(facts.get(EDB_USES_SOURCE_COMPUTER, ())),
    }
    return evaluate(mitre_rules(), relevant, max_iterations=max_depth)


def derived_triples(derived: Facts) -> list[tuple[str, str, str]]:
    """Return only the *newly derived* IDB relations as ``(h, r, t)`` triples.

    These are the triples fed back into the embedding graph as logical
    provenance (the hybrid link, LO2/LO12).
    """
    out: list[tuple[str, str, str]] = []
    for rel in BINARY_IDB_RELATIONS:
        for h, t in sorted(derived.get(rel, ())):
            out.append((h, rel, t))
    return out


def flagged_entities(derived: Facts) -> set[str]:
    """Entities that participate in a suspicious lateral-movement chain.

    Used as the logical signal in hybrid scoring: a scored triple whose head or
    tail is flagged gets a logic boost on top of its embedding anomaly score.
    """
    flagged: set[str] = set()
    for u, a, c in derived.get(REL_SUSPICIOUS_CHAIN, ()):
        flagged.update((u, a, c))
    # Also anyone reachable through lateral movement that originates the chain.
    for a, c in derived.get(REL_LATERAL_MOVEMENT, ()):
        if a in flagged or c in flagged:
            flagged.update((a, c))
    return flagged


def attack_chains(derived: Facts) -> list[dict]:
    """Materialised attack-chain objects with their members and technique."""
    members: dict[str, set[str]] = {}
    for chain_id, member in derived.get(REL_PART_OF, ()):
        members.setdefault(chain_id, set()).add(member)
    technique: dict[str, str] = {}
    for chain_id, tech in derived.get(REL_CHAIN_TECHNIQUE, ()):
        technique[chain_id] = tech
    chains = []
    for chain_id, mem in sorted(members.items()):
        chains.append(
            {
                "id": chain_id,
                "members": sorted(mem),
                "technique": technique.get(chain_id),
            }
        )
    return chains


def chains_for_entity(derived: Facts, entity: str) -> list[dict]:
    """All attack chains that involve ``entity`` (used by the dashboard query)."""
    return [c for c in attack_chains(derived) if entity in c["members"]]


def write_derived(path: Path | str, derived: Facts) -> int:
    """Write derived triples to ``path`` (TSV); return the number written."""
    triples = derived_triples(derived)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    return len(triples)


def write_flagged(path: Path | str, derived: Facts) -> int:
    """Persist the flagged-entity set (one per line) for hybrid scoring."""
    flagged = sorted(flagged_entities(derived))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(flagged) + ("\n" if flagged else ""))
    return len(flagged)


def load_flagged(path: Path | str) -> set[str]:
    """Load a flagged-entity set written by :func:`write_flagged`."""
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def write_chains(path: Path | str, derived: Facts) -> int:
    """Persist materialised attack chains as JSON (consumed by the dashboard)."""
    import json

    chains = attack_chains(derived)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chains, indent=2))
    return len(chains)


def load_chains(path: Path | str) -> list[dict]:
    import json

    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def summary(derived: Facts) -> dict:
    """Small dict summarising what the reasoner derived (for logging/reports)."""
    return {
        "remote_auth": len(derived.get(REL_REMOTE_AUTH, ())),
        "lateral_movement": len(derived.get(REL_LATERAL_MOVEMENT, ())),
        "credential_use": len(derived.get(REL_CREDENTIAL_USE, ())),
        "suspicious_chain": len(derived.get(REL_SUSPICIOUS_CHAIN, ())),
        "attack_chains": len(attack_chains(derived)),
        "flagged_entities": len(flagged_entities(derived)),
    }
