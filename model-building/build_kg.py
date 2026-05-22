from pathlib import Path
import pandas as pd

DATA = Path("dataset")
OUT = Path("generated-files/triples.tsv")

MAX_ROWS_PER_FILE = 200_000_000
MAX_TIME = 24 * 60 * 60  # first day only; remove later if wanted


def clean(x):
    if pd.isna(x):
        return None
    x = str(x)
    if x == "?":
        return None
    return x


def user(x):
    x = clean(x)
    return f"user:{x}" if x else None


def computer(x):
    x = clean(x)
    return f"computer:{x}" if x else None


def process(x):
    x = clean(x)
    return f"process:{x}" if x else None


def port(x):
    x = clean(x)
    return f"port:{x}" if x else None


def add(triples, h, r, t):
    if h and r and t:
        triples.add((h, r, t))


def read_csv(name, columns):
    path = DATA / name
    return pd.read_csv(
        path,
        names=columns,
        nrows=MAX_ROWS_PER_FILE,
        compression="infer",
    )


triples = set()

# -----------------------
# auth.txt / auth.txt.gz
# -----------------------
auth_cols = [
    "time",
    "src_user",
    "dst_user",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "orientation",
    "result",
]

auth = read_csv("auth.txt", auth_cols)
auth = auth[auth["time"] <= MAX_TIME]

for row in auth.itertuples(index=False):
    su = user(row.src_user)
    du = user(row.dst_user)
    sc = computer(row.src_computer)
    dc = computer(row.dst_computer)

    orientation = clean(row.orientation)
    result = clean(row.result)

    # Keep first version simple
    if orientation == "LogOn" and result == "Success":
        add(triples, su, "logs_on_to", dc)
        add(triples, sc, "authenticates_to", dc)
        add(triples, su, "uses_source_computer", sc)

    # Optional: connect source and destination user accounts
    add(triples, su, "authenticates_as", du)


# -----------------------
# dns.txt / dns.txt.gz
# -----------------------
dns_cols = ["time", "src_computer", "resolved_computer"]

dns = read_csv("dns.txt", dns_cols)
dns = dns[dns["time"] <= MAX_TIME]

for row in dns.itertuples(index=False):
    add(
        triples,
        computer(row.src_computer),
        "dns_resolves",
        computer(row.resolved_computer),
    )


# -----------------------
# flows.txt / flows.txt.gz
# -----------------------
flow_cols = [
    "time",
    "duration",
    "src_computer",
    "src_port",
    "dst_computer",
    "dst_port",
    "protocol",
    "packet_count",
    "byte_count",
]

flows = read_csv("flows.txt", flow_cols)
flows = flows[flows["time"] <= MAX_TIME]

for row in flows.itertuples(index=False):
    sc = computer(row.src_computer)
    dc = computer(row.dst_computer)

    add(triples, sc, "flows_to", dc)

    # Optional port nodes
    add(triples, sc, "uses_src_port", port(row.src_port))
    add(triples, dc, "uses_dst_port", port(row.dst_port))


# -----------------------
# proc.txt / proc.txt.gz
# -----------------------
proc_cols = ["time", "user", "computer", "process", "action"]

proc_df = read_csv("proc.txt", proc_cols)
proc_df = proc_df[proc_df["time"] <= MAX_TIME]

for row in proc_df.itertuples(index=False):
    u = user(row.user)
    c = computer(row.computer)
    p = process(row.process)

    action = clean(row.action)

    if action == "Start":
        add(triples, u, "starts_process", p)
        add(triples, c, "runs_process", p)
        add(triples, u, "active_on_computer", c)

    elif action == "End":
        add(triples, u, "ends_process", p)
        add(triples, c, "stops_process", p)


# Write PyKEEN-compatible TSV
with OUT.open("w") as f:
    for h, r, t in sorted(triples):
        f.write(f"{h}\t{r}\t{t}\n")

print(f"Wrote {len(triples):,} triples to {OUT}")