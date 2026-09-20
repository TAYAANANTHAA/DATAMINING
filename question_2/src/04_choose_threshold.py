from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FOLDER = Path("outputs")

# Read scores produced by the previous program
pairs = pd.read_csv(OUTPUT_FOLDER / "03_labelled_pair_scores.csv")

# We selected word TF-IDF because it had the higher AUROC
scores = pairs["word_score"]
actual_same = (pairs["label"] == "same")

# Product rule:
# false merge = different pair incorrectly called same = cost 10
# missed duplicate = same pair incorrectly called different = cost 1
FALSE_MERGE_COST = 10
MISSED_DUPLICATE_COST = 1

results = []

# Test every observed score as a possible decision threshold
for threshold in sorted(scores.unique()):

    predicted_same = scores >= threshold

    true_positive = ((predicted_same) & (actual_same)).sum()
    false_positive = ((predicted_same) & (~actual_same)).sum()
    false_negative = ((~predicted_same) & (actual_same)).sum()
    true_negative = ((~predicted_same) & (~actual_same)).sum()

    total_cost = (
        FALSE_MERGE_COST * false_positive
        + MISSED_DUPLICATE_COST * false_negative
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0 else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0 else 0
    )

    results.append({
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive_false_merges": false_positive,
        "false_negative_missed_duplicates": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "total_cost": total_cost
    })

results = pd.DataFrame(results)

# If several thresholds have same lowest cost,
# choose the highest one because false merges are dangerous.
best = (
    results[results["total_cost"] == results["total_cost"].min()]
    .sort_values("threshold", ascending=False)
    .iloc[0]
)

print("\n--- SCORE RANGES ---")
print(
    pairs.groupby("label")["word_score"]
    .agg(["min", "max", "mean"])
    .round(4)
)

print("\n--- COST SETTINGS ---")
print("False merge cost:", FALSE_MERGE_COST)
print("Missed duplicate cost:", MISSED_DUPLICATE_COST)

print("\n--- CHOSEN DECISION THRESHOLD ---")
print("Threshold:", round(best["threshold"], 4))
print("True positives:", int(best["true_positive"]))
print("False merges:", int(best["false_positive_false_merges"]))
print("Missed duplicates:", int(best["false_negative_missed_duplicates"]))
print("Precision:", round(best["precision"], 4))
print("Recall:", round(best["recall"], 4))
print("Total weighted cost:", int(best["total_cost"]))

# Save table for report
results.to_csv(OUTPUT_FOLDER / "04_threshold_costs.csv", index=False)

# Plot the cost of each threshold
plt.figure(figsize=(10, 6))
plt.plot(results["threshold"], results["total_cost"], color="purple")
plt.scatter(
    best["threshold"],
    best["total_cost"],
    color="red",
    label=f"chosen threshold = {best['threshold']:.4f}"
)
plt.title("Cost-aware threshold selection")
plt.xlabel("Word TF-IDF similarity threshold")
plt.ylabel("Weighted cost: 10 × false merges + missed duplicates")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "04_threshold_cost_curve.png", dpi=150)

print("\nSaved: outputs/04_threshold_costs.csv")
print("Saved: outputs/04_threshold_cost_curve.png")