import pandas as pd

normal = pd.read_csv("generated-files/normal_scores.csv")
red = pd.read_csv("generated-files/redteam_scores.csv")

print("Normal scores:")
print(normal["score"].describe())

print("\nRedteam scores:")
print(red["score"].describe())