from pathlib import Path
from collections import defaultdict
import hashlib
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_FOLDER = Path("outputs")

CHOSEN_BANDS = 80
CHOSEN_ROWS_PER_BAND = 5

# 80 bands × 5 rows = 400 MinHash values
assert CHOSEN_BANDS * CHOSEN_ROWS_PER_BAND == 400


def bucket_hash(signature_part):
    """Convert one band of MinHash values into a stable bucket number."""

    digest = hashlib.blake2b(
        signature_part.tobytes(),
        digest_size=8
    ).digest()

    return int.from_bytes(digest, byteorder="big")


# Read saved MinHash signatures and notice IDs
signatures = np.load(OUTPUT_FOLDER / "05_minhash_signatures.npy")
notice_ids = pd.read_csv(OUTPUT_FOLDER / "05_minhash_notice_ids.csv")
pairs = pd.read_csv(OUTPUT_FOLDER / "05_minhash_pair_errors.csv")

notice_row = pd.Series(
    notice_ids.index,
    index=notice_ids["notice_id"]
)

print("\n--- LSH CONFIGURATION ---")
print("Notices:", len(notice_ids))
print("MinHash values per notice:", signatures.shape[1])
print("Bands:", CHOSEN_BANDS)
print("Rows per band:", CHOSEN_ROWS_PER_BAND)

start_time = time.perf_counter()

# Build LSH buckets
buckets = defaultdict(list)

for row_number in range(len(notice_ids)):

    for band_number in range(CHOSEN_BANDS):

        start = band_number * CHOSEN_ROWS_PER_BAND
        end = start + CHOSEN_ROWS_PER_BAND

        key = (
            band_number,
            bucket_hash(signatures[row_number, start:end])
        )

        buckets[key].append(row_number)

print("Total LSH buckets:", len(buckets))

# Candidate list for every notice
candidate_counts = np.zeros(len(notice_ids), dtype=int)
bucket_rows_examined = np.zeros(len(notice_ids), dtype=int)

for row_number in range(len(notice_ids)):

    candidates = set()
    rows_examined = 0

    for band_number in range(CHOSEN_BANDS):

        start = band_number * CHOSEN_ROWS_PER_BAND
        end = start + CHOSEN_ROWS_PER_BAND

        key = (
            band_number,
            bucket_hash(signatures[row_number, start:end])
        )

        members = buckets[key]

        # These are bucket rows PostgreSQL would retrieve.
        rows_examined += len(members) - 1

        candidates.update(members)

    candidates.discard(row_number)

    candidate_counts[row_number] = len(candidates)
    bucket_rows_examined[row_number] = rows_examined

    if (row_number + 1) % 1000 == 0:
        print(f"Candidate lists completed: {row_number + 1} of {len(notice_ids)}")

runtime_seconds = time.perf_counter() - start_time

# Save candidate workload, needed later for Section B(e)
candidate_results = notice_ids.copy()
candidate_results["candidate_count"] = candidate_counts
candidate_results["bucket_rows_examined"] = bucket_rows_examined

candidate_results.to_csv(
    OUTPUT_FOLDER / "06_lsh_candidate_workload.csv",
    index=False
)

# Check whether every labelled pair survives to candidate stage
pair_row_a = pairs["notice_id_a"].map(notice_row).astype(int).to_numpy()
pair_row_b = pairs["notice_id_b"].map(notice_row).astype(int).to_numpy()

signature_a = signatures[pair_row_a].reshape(
    len(pairs),
    CHOSEN_BANDS,
    CHOSEN_ROWS_PER_BAND
)

signature_b = signatures[pair_row_b].reshape(
    len(pairs),
    CHOSEN_BANDS,
    CHOSEN_ROWS_PER_BAND
)

# A pair survives if it has identical MinHash values in at least one band.
pairs["survives_lsh"] = np.any(
    np.all(signature_a == signature_b, axis=2),
    axis=1
)

print("\n--- LABELLED-PAIR CANDIDATE RESULTS ---")
print(
    pairs.groupby("label")["survives_lsh"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"sum": "survived", "mean": "survival_rate"})
    .round(4)
)

print("\n--- CANDIDATE WORKLOAD ---")
print("Average candidates per notice:",
      round(candidate_results["candidate_count"].mean(), 2))
print("Median candidates per notice:",
      round(candidate_results["candidate_count"].median(), 2))
print("95th percentile candidates:",
      round(candidate_results["candidate_count"].quantile(0.95), 2))
print("99th percentile candidates:",
      round(candidate_results["candidate_count"].quantile(0.99), 2))
print("Maximum candidates:",
      int(candidate_results["candidate_count"].max()))
print("Runtime in seconds:", round(runtime_seconds, 2))

# Compare possible 400-hash LSH configurations on labelled pairs
settings = [
    (40, 10),
    (80, 5),
    (100, 4)
]

configuration_results = []

for bands, rows_per_band in settings:

    pair_a = signatures[pair_row_a].reshape(
        len(pairs),
        bands,
        rows_per_band
    )

    pair_b = signatures[pair_row_b].reshape(
        len(pairs),
        bands,
        rows_per_band
    )

    survives = np.any(
        np.all(pair_a == pair_b, axis=2),
        axis=1
    )

    same_recall = survives[pairs["label"].to_numpy() == "same"].mean()
    different_survival = survives[
        pairs["label"].to_numpy() == "different"
    ].mean()

    configuration_results.append({
        "bands": bands,
        "rows_per_band": rows_per_band,
        "same_pair_recall": same_recall,
        "different_pair_survival_rate": different_survival
    })

configuration_results = pd.DataFrame(configuration_results)

print("\n--- CONFIGURATION COMPARISON ---")
print(configuration_results.round(4))

configuration_results.to_csv(
    OUTPUT_FOLDER / "06_lsh_configuration_comparison.csv",
    index=False
)

pairs.to_csv(
    OUTPUT_FOLDER / "06_lsh_labelled_pair_results.csv",
    index=False
)

# Plot survival probability against exact Jaccard similarity
similarities = np.linspace(0, 1, 200)

theoretical_survival = 1 - (
    1 - similarities ** CHOSEN_ROWS_PER_BAND
) ** CHOSEN_BANDS

# Bin labelled pairs by their known exact Jaccard similarity
pairs["similarity_bin"] = pd.cut(
    pairs["exact_jaccard"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True
)

empirical = (
    pairs.groupby("similarity_bin", observed=True)
    .agg(
        mean_exact_similarity=("exact_jaccard", "mean"),
        survival_rate=("survives_lsh", "mean"),
        pair_count=("survives_lsh", "size")
    )
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.plot(
    similarities,
    theoretical_survival,
    label="Theoretical LSH survival probability",
    color="purple"
)

plt.scatter(
    empirical["mean_exact_similarity"],
    empirical["survival_rate"],
    label="Observed labelled-pair survival rate",
    color="darkorange",
    s=55
)

plt.axvline(
    0.4771,
    linestyle="--",
    color="red",
    label="Final similarity decision threshold"
)

plt.title("Probability that a pair reaches the LSH candidate stage")
plt.xlabel("Exact Jaccard similarity")
plt.ylabel("Probability of candidate survival")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_FOLDER / "06_lsh_survival_curve.png",
    dpi=150
)

print("\nSaved: outputs/06_lsh_candidate_workload.csv")
print("Saved: outputs/06_lsh_configuration_comparison.csv")
print("Saved: outputs/06_lsh_labelled_pair_results.csv")
print("Saved: outputs/06_lsh_survival_curve.png")