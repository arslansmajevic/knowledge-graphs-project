import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

normal = pd.read_csv("generated-files/normal_scores.csv")
red = pd.read_csv("generated-files/redteam_scores.csv")

normal["label"] = 0
red["label"] = 1

df = pd.concat([normal, red], ignore_index=True)

# PyKEEN score: higher = more plausible
# For anomaly detection, invert it
df["anomaly_score"] = -df["score"]

auc = roc_auc_score(df["label"], df["anomaly_score"])
ap = average_precision_score(df["label"], df["anomaly_score"])

print(f"ROC-AUC: {auc:.4f}")
print(f"Average precision: {ap:.4f}")

print("\nNormal anomaly scores:")
print(df[df["label"] == 0]["anomaly_score"].describe())

print("\nRedteam anomaly scores:")
print(df[df["label"] == 1]["anomaly_score"].describe())