from pathlib import Path
from collections import defaultdict
import hashlib
import re
import time

import numpy as np
import pandas as pd
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


def clean_text(value):
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_aggregator_preamble(portal_id, body):
    body = "" if pd.isna(body) else str(body)

    if portal_id not in AGGREGATOR_PORTALS:
        return body

    match = re.search(r"\bname\s+of\s+work\s*:", body, re.IGNORECASE)

    if match is None:
        return body

    return body[match.start():]


def remove_repeated_template_sections(body):
    """
    Keep tender-specific header information.
    Remove the generic scope/BOQ template that causes LSH collisions.
    """

    match = re.search(
        r"\b(scope\s+of\s+work|abstract\s+bill\s+of\s+quantities)\b",
        body,
        flags=re.IGNORECASE
    )

    if match is None:
        return body, False

    return body[:match.start()], True


def make_shingles(text, size=3):
    words = re.findall(r"[a-z0-9]+", str(text).lower())

    if len(words) < size:
        return set(words)

    return {
        " ".join(words[position:position + size])
        for position in range(len(words) - size + 1)
    }


def bucket_hash(signature_part):
    return hashlib.blake2b(
        signature_part.tobytes(),
        digest_size=8
    ).hexdigest()


# Read original prepared notices and labels
notices = pd.read_csv(OUTPUT_FOLDER / "02_notices_cleaned.csv")
pairs = pd.read_csv(
    r"C:\Users\Huawei\Downloads\data_2 (1)\data_2\labelled_pairs.csv"
)

baseline_workload = pd.read_csv(
    OUTPUT_FOLDER / "06_lsh_candidate_workload.csv"
)

baseline_pairs = pd.read_csv(
    OUTPUT_FOLDER / "06_lsh_labelled_pair_results.csv"
)

print("\n--- FINAL TEMPLATE-SECTION MITIGATION ---")

final_texts = []
template_sections_removed = []

for _, row in notices.iterrows():

    body = remove_aggregator_preamble(
        row["portal_id"],
        row["body"]
    )

    body, removed = remove_repeated_template_sections(body)

    final_text = clean_text(
        str(row["title"]) + " " + body
    )

    final_texts.append(final_text)
    template_sections_removed.append(removed)

notices["final_mitigated_text"] = final_texts
notices["template_sections_removed"] = template_sections_removed

print("Notices with scope/BOQ section removed:",
      int(notices["template_sections_removed"].sum()))

print("Average final text length:",
      round(notices["final_mitigated_text"].str.len().mean(), 2))

# Build final MinHash signatures
print("\n--- BUILDING FINAL MINHASH SIGNATURES ---")

signatures = np.zeros(
    (len(notices), NUMBER_OF_HASHES),
    dtype=np.uint64
)

start_minhash = time.perf_counter()

for row_number, row in notices.iterrows():

    shingles = make_shingles(
        row["final_mitigated_text"],
        SHINGLE_SIZE
    )

    signature = MinHash(
        num_perm=NUMBER_OF_HASHES,
        seed=42
    )

    for shingle in shingles:
        signature.update(shingle.encode("utf-8"))

    signatures[row_number] = signature.hashvalues

    if (row_number + 1) % 500 == 0:
        print(f"MinHash completed: {row_number + 1} of {len(notices)}")

minhash_runtime = time.perf_counter() - start_minhash

np.save(
    OUTPUT_FOLDER / "09_final_minhash_signatures.npy",
    signatures
)

# Build final LSH buckets and candidate lists
print("\n--- BUILDING FINAL LSH CANDIDATE LISTS ---")

start_lsh = time.perf_counter()
buckets = defaultdict(list)

for row_number in range(len(notices)):

    for band_no in range(BANDS):

        start = band_no * ROWS_PER_BAND
        end = start + ROWS_PER_BAND

        key = (
            band_no,
            bucket_hash(signatures[row_number, start:end])
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
            bucket_hash(signatures[row_number, start:end])
        )

        members = buckets[key]
        examined += len(members) - 1
        candidates.update(members)

    candidates.discard(row_number)

    candidate_counts[row_number] = len(candidates)
    bucket_rows_examined[row_number] = examined

lsh_runtime = time.perf_counter() - start_lsh

final_workload = notices[
    ["notice_id", "portal_id", "template_sections_removed"]
].copy()

final_workload["candidate_count"] = candidate_counts
final_workload["bucket_rows_examined"] = bucket_rows_examined

final_workload.to_csv(
    OUTPUT_FOLDER / "09_final_candidate_workload.csv",
    index=False
)

# Measure labelled-pair candidate recall
notice_row = pd.Series(
    notices.index,
    index=notices["notice_id"]
)

row_a = pairs["notice_id_a"].map(notice_row).astype(int).to_numpy()
row_b = pairs["notice_id_b"].map(notice_row).astype(int).to_numpy()

signature_a = signatures[row_a].reshape(
    len(pairs), BANDS, ROWS_PER_BAND
)

signature_b = signatures[row_b].reshape(
    len(pairs), BANDS, ROWS_PER_BAND
)

pairs["survives_final_lsh"] = np.any(
    np.all(signature_a == signature_b, axis=2),
    axis=1
)

pairs.to_csv(
    OUTPUT_FOLDER / "09_final_labelled_pair_results.csv",
    index=False
)

# Compare baseline and final mitigation
baseline_same_recall = baseline_pairs.loc[
    baseline_pairs["label"] == "same",
    "survives_lsh"
].mean()

baseline_different_survival = baseline_pairs.loc[
    baseline_pairs["label"] == "different",
    "survives_lsh"
].mean()

final_same_recall = pairs.loc[
    pairs["label"] == "same",
    "survives_final_lsh"
].mean()

final_different_survival = pairs.loc[
    pairs["label"] == "different",
    "survives_final_lsh"
].mean()

summary = pd.DataFrame([
    {
        "version": "baseline",
        "average_candidates": baseline_workload["candidate_count"].mean(),
        "p95_candidates": baseline_workload["candidate_count"].quantile(0.95),
        "p99_candidates": baseline_workload["candidate_count"].quantile(0.99),
        "maximum_candidates": baseline_workload["candidate_count"].max(),
        "same_pair_candidate_recall": baseline_same_recall,
        "different_pair_survival_rate": baseline_different_survival
    },
    {
        "version": "final_template_mitigation",
        "average_candidates": final_workload["candidate_count"].mean(),
        "p95_candidates": final_workload["candidate_count"].quantile(0.95),
        "p99_candidates": final_workload["candidate_count"].quantile(0.99),
        "maximum_candidates": final_workload["candidate_count"].max(),
        "same_pair_candidate_recall": final_same_recall,
        "different_pair_survival_rate": final_different_survival
    }
])

summary.to_csv(
    OUTPUT_FOLDER / "09_final_before_after_summary.csv",
    index=False
)

print("\n--- BASELINE / FINAL COMPARISON ---")
print(summary.round(4).to_string(index=False))

# Show affected portals
portal_summary = (
    final_workload.groupby("portal_id")["candidate_count"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- TOP 15 PORTALS AFTER FINAL MITIGATION ---")
print(portal_summary.head(15).round(2))

print("\nMinHash runtime (seconds):", round(minhash_runtime, 2))
print("LSH runtime (seconds):", round(lsh_runtime, 2))

print("\nSaved: outputs/09_final_minhash_signatures.npy")
print("Saved: outputs/09_final_candidate_workload.csv")
print("Saved: outputs/09_final_labelled_pair_results.csv")
print("Saved: outputs/09_final_before_after_summary.csv")