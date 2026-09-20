from pathlib import Path
from collections import defaultdict
import hashlib
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasketch import MinHash

OUTPUT_FOLDER = Path("outputs")

NUMBER_OF_HASHES = 400
SHINGLE_SIZE = 3
BANDS = 80
ROWS_PER_BAND = 5

AGGREGATOR_PORTALS = {
    "P001", "P002", "P003",
    "P004", "P005", "P006"
}

assert BANDS * ROWS_PER_BAND == NUMBER_OF_HASHES


def clean_text(value):
    """Apply the same text normalisation used in Section A(a)."""

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_aggregator_preamble(portal_id, body):
    """
    For known aggregator portals only, retain text from 'Name of work:'
    onward. This removes their repeated legal/instruction boilerplate.
    """

    body = "" if pd.isna(body) else str(body)

    if portal_id not in AGGREGATOR_PORTALS:
        return body, False

    match = re.search(r"\bname\s+of\s+work\s*:", body, flags=re.IGNORECASE)

    if match is None:
        return body, False

    return body[match.start():], True


def make_shingles(text, size=3):
    """Create a set of overlapping three-word shingles."""

    words = re.findall(r"[a-z0-9]+", str(text).lower())

    if len(words) < size:
        return set(words)

    return {
        " ".join(words[position:position + size])
        for position in range(len(words) - size + 1)
    }


def bucket_hash(signature_part):
    """Create a stable LSH bucket key."""

    return hashlib.blake2b(
        signature_part.tobytes(),
        digest_size=8
    ).hexdigest()


# Read original prepared notice data
notices = pd.read_csv(OUTPUT_FOLDER / "02_notices_cleaned.csv")
pairs = pd.read_csv(
    r"C:\Users\Huawei\Downloads\data_2 (1)\data_2\labelled_pairs.csv"
)

# Baseline results for before/after comparison
baseline_workload = pd.read_csv(
    OUTPUT_FOLDER / "06_lsh_candidate_workload.csv"
)

baseline_pair_results = pd.read_csv(
    OUTPUT_FOLDER / "06_lsh_labelled_pair_results.csv"
)

print("\n--- APPLYING PORTAL-AWARE BOILERPLATE MITIGATION ---")

mitigated_bodies = []
preamble_removed = []

for _, row in notices.iterrows():

    new_body, removed = remove_aggregator_preamble(
        row["portal_id"],
        row["body"]
    )

    mitigated_bodies.append(new_body)
    preamble_removed.append(removed)

notices["mitigated_body"] = mitigated_bodies
notices["preamble_removed"] = preamble_removed

notices["mitigated_text"] = (
    notices["title"].fillna("")
    + " "
    + notices["mitigated_body"].fillna("")
).apply(clean_text)

notices["mitigated_length"] = notices["mitigated_text"].str.len()

print("Notices with preamble removed:",
      int(notices["preamble_removed"].sum()))

print("Average baseline text length:",
      round(notices["clean_text"].str.len().mean(), 2))

print("Average mitigated text length:",
      round(notices["mitigated_length"].mean(), 2))

# Build new 400-value MinHash signatures
print("\n--- BUILDING MITIGATED MINHASH SIGNATURES ---")

mitigated_signatures = np.zeros(
    (len(notices), NUMBER_OF_HASHES),
    dtype=np.uint64
)

start_minhash = time.perf_counter()

for row_number, row in notices.iterrows():

    shingles = make_shingles(
        row["mitigated_text"],
        SHINGLE_SIZE
    )

    signature = MinHash(
        num_perm=NUMBER_OF_HASHES,
        seed=42
    )

    for shingle in shingles:
        signature.update(shingle.encode("utf-8"))

    mitigated_signatures[row_number] = signature.hashvalues

    if (row_number + 1) % 500 == 0:
        print(f"MinHash completed: {row_number + 1} of {len(notices)}")

minhash_runtime = time.perf_counter() - start_minhash

np.save(
    OUTPUT_FOLDER / "08_mitigated_minhash_signatures.npy",
    mitigated_signatures
)

# Build new LSH buckets
print("\n--- BUILDING MITIGATED LSH CANDIDATE LISTS ---")

start_lsh = time.perf_counter()
buckets = defaultdict(list)

for row_number in range(len(notices)):

    for band_no in range(BANDS):

        start = band_no * ROWS_PER_BAND
        end = start + ROWS_PER_BAND

        key = (
            band_no,
            bucket_hash(
                mitigated_signatures[row_number, start:end]
            )
        )

        buckets[key].append(row_number)

candidate_counts = np.zeros(len(notices), dtype=int)
bucket_rows_examined = np.zeros(len(notices), dtype=int)

for row_number in range(len(notices)):

    candidates = set()
    examined = 0

    for band_no in range(BANDS):

        start = band_no * ROWS_PER_BAND
        end = start + ROWS_PER_BAND

        key = (
            band_no,
            bucket_hash(
                mitigated_signatures[row_number, start:end]
            )
        )

        members = buckets[key]

        examined += len(members) - 1
        candidates.update(members)

    candidates.discard(row_number)

    candidate_counts[row_number] = len(candidates)
    bucket_rows_examined[row_number] = examined

lsh_runtime = time.perf_counter() - start_lsh

mitigated_workload = notices[
    ["notice_id", "portal_id", "preamble_removed"]
].copy()

mitigated_workload["candidate_count"] = candidate_counts
mitigated_workload["bucket_rows_examined"] = bucket_rows_examined

mitigated_workload.to_csv(
    OUTPUT_FOLDER / "08_mitigated_candidate_workload.csv",
    index=False
)

# Measure labelled-pair candidate recall after mitigation
notice_row = pd.Series(
    notices.index,
    index=notices["notice_id"]
)

row_a = pairs["notice_id_a"].map(notice_row).astype(int).to_numpy()
row_b = pairs["notice_id_b"].map(notice_row).astype(int).to_numpy()

signature_a = mitigated_signatures[row_a].reshape(
    len(pairs), BANDS, ROWS_PER_BAND
)

signature_b = mitigated_signatures[row_b].reshape(
    len(pairs), BANDS, ROWS_PER_BAND
)

pairs["survives_mitigated_lsh"] = np.any(
    np.all(signature_a == signature_b, axis=2),
    axis=1
)

pairs.to_csv(
    OUTPUT_FOLDER / "08_mitigated_labelled_pair_results.csv",
    index=False
)

# Before/after summary
baseline_same_recall = baseline_pair_results.loc[
    baseline_pair_results["label"] == "same",
    "survives_lsh"
].mean()

mitigated_same_recall = pairs.loc[
    pairs["label"] == "same",
    "survives_mitigated_lsh"
].mean()

baseline_different_rate = baseline_pair_results.loc[
    baseline_pair_results["label"] == "different",
    "survives_lsh"
].mean()

mitigated_different_rate = pairs.loc[
    pairs["label"] == "different",
    "survives_mitigated_lsh"
].mean()

summary = pd.DataFrame([
    {
        "version": "before_mitigation",
        "average_candidates": baseline_workload["candidate_count"].mean(),
        "p95_candidates": baseline_workload["candidate_count"].quantile(0.95),
        "p99_candidates": baseline_workload["candidate_count"].quantile(0.99),
        "max_candidates": baseline_workload["candidate_count"].max(),
        "same_pair_candidate_recall": baseline_same_recall,
        "different_pair_survival_rate": baseline_different_rate,
        "minhash_runtime_seconds": np.nan,
        "lsh_runtime_seconds": 4.22
    },
    {
        "version": "after_mitigation",
        "average_candidates": mitigated_workload["candidate_count"].mean(),
        "p95_candidates": mitigated_workload["candidate_count"].quantile(0.95),
        "p99_candidates": mitigated_workload["candidate_count"].quantile(0.99),
        "max_candidates": mitigated_workload["candidate_count"].max(),
        "same_pair_candidate_recall": mitigated_same_recall,
        "different_pair_survival_rate": mitigated_different_rate,
        "minhash_runtime_seconds": minhash_runtime,
        "lsh_runtime_seconds": lsh_runtime
    }
])

summary.to_csv(
    OUTPUT_FOLDER / "08_before_after_summary.csv",
    index=False
)

print("\n--- BEFORE / AFTER RESULTS ---")
print(summary.round(4).to_string(index=False))

# Portal-level analysis
before_by_portal = (
    baseline_workload.groupby("portal_id")["candidate_count"]
    .mean()
    .rename("before_average_candidates")
)

after_by_portal = (
    mitigated_workload.groupby("portal_id")["candidate_count"]
    .mean()
    .rename("after_average_candidates")
)

portal_comparison = pd.concat(
    [before_by_portal, after_by_portal],
    axis=1
).fillna(0)

portal_comparison["reduction"] = (
    portal_comparison["before_average_candidates"]
    - portal_comparison["after_average_candidates"]
)

portal_comparison = portal_comparison.sort_values(
    "before_average_candidates",
    ascending=False
)

portal_comparison.to_csv(
    OUTPUT_FOLDER / "08_portal_workload_comparison.csv"
)

print("\n--- TOP 15 PORTALS BEFORE MITIGATION ---")
print(portal_comparison.head(15).round(2))

# Plot top 15 heavy portals before and after
top_portals = portal_comparison.head(15)

plt.figure(figsize=(12, 7))

positions = np.arange(len(top_portals))
width = 0.38

plt.bar(
    positions - width / 2,
    top_portals["before_average_candidates"],
    width,
    label="Before mitigation",
    color="tomato"
)

plt.bar(
    positions + width / 2,
    top_portals["after_average_candidates"],
    width,
    label="After mitigation",
    color="steelblue"
)

plt.xticks(positions, top_portals.index, rotation=45)
plt.ylabel("Average candidates per notice")
plt.title("Portal workload before and after boilerplate mitigation")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "08_portal_workload_before_after.png",
    dpi=150
)

print("\nSaved: outputs/08_before_after_summary.csv")
print("Saved: outputs/08_portal_workload_comparison.csv")
print("Saved: outputs/08_portal_workload_before_after.png")