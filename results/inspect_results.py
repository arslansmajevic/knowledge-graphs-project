import json
from pathlib import Path

path = Path("pykeen-lanl-model/results.json")

with path.open() as f:
    results = json.load(f)

# Print top-level keys so you can see the structure
print(results.keys())

# Try common metric locations
metric_results = results.get("metric_results", {})
print(json.dumps(metric_results, indent=2)[:5000])