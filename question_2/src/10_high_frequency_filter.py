from pathlib import Path
from collections import defaultdict
import hashlib
import re
import time

import numpy as np
import pandas as pd
from datasketch import MinHash
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_FOLDER = Path("outputs")

NUMBER_OF_HASHES = 400
BANDS = 80
ROWS_PER_BAND = 5

AGGREGATOR_PORTALS = {
    "P001", "P002", "P003",
    "P004", "P005", "P006"
}

assert BANDS * ROWS_PER_BAND == NUMBER_OF_HASHES


def remove_aggregator_preamble(portal_id, body):
    """Remove known portal preamble only from P001–P006."""

    body = "" if pd.isna(body) else str(body)

    if portal_id not in AGGREGATOR_PORTALS:
        return body

    match = re.search(
        r"\bname\s+of\s+work\s*:",
        body,
        flags=re.IGNORECASE
    )

    if match is None:
        return body

    return body[match.start():]


def clean_text(value):
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def bucket_hash(signature_part):
    return hashlib.blake2b(
        signature_part.tobytes(),
        digest_size=8
    ).hexdigest()


# Read data and baseline results
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

print("\n--- HIGH-FREQUENCY BOILERPLATE FILTER ---")

# First remove known aggregator preambles.
base_texts = []

for _, row in notices.iterrows():

    body = remove_aggregator_preamble(
        row["portal_id"],
        row["body"]
    )

    base_texts.append(
        clean_text(str(row["title"]) + " " + body)
    )

notices["filtered_source_text"] = base_texts

# Learn frequent terms from this corpus.
# max_df=0.05 removes terms occurring in more than 5% of notices.
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.05,
    sublinear_tf=True
)

vectorizer.fit(notices["filtered_source_text"])

allowed_features = set(vectorizer.vocabulary_.keys())
analyser = vectorizer.build_analyzer()

print("Retained rare/discriminative features:",
      len(allowed_features))

# Build filtered feature sets.
# Common boilerplate terms/bigrams are excluded.
filtered_feature_sets = []

for row_number, text in enumerate(notices["filtered_source_text"]):

    features = {
        feature
        for feature in analyser(text)
        if feature in allowed_features
    }

    filtered_feature_sets.append(features)

    if (row_number + 1) % 1000 == 0:
        print(f"Feature filtering completed: {row_number + 1} of {len(notices)}")

print("Average retained features per notice:",
      round(np.mean([len(x) for x in filtered_feature_sets]), 2))

# Build MinHash signatures from filtered features
print("\n--- BUILDING FILTERED MINHASH SIGNATURES ---")

signatures = np.zeros(
    (len(notices), NUMBER_OF_HASHES),
    dtype=np.uint64
)

start_minhash = time.perf_counter()

for row_number, features in enumerate(filtered_feature_sets):

    signature = MinHash(
        num_perm=NUMBER_OF_HASHES,
        seed=42
    )

    for feature in features:
        signature.update(feature.encode("utf-8"))

    signatures[row_number] = signature.hashvalues

    if (row_number + 1) % 500 == 0:
        print(f"MinHash completed: {row_number + 1} of {len(notices)}")

minhash_runtime = time.perf_counter() - start_minhash

np.save(
    OUTPUT_FOLDER / "10_filtered_minhash_signatures.npy",
    signatures
)

# Build LSH buckets
print("\n--- BUILDING FILTERED LSH CANDIDATE LISTS ---")

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

filtered_workload = notices[["notice_id", "portal_id"]].copy()
filtered_workload["candidate_count"] = candidate_counts
filtered_workload["bucket_rows_examined"] = bucket_rows_examined

filtered_workload.to_csv(
    OUTPUT_FOLDER / "10_filtered_candidate_workload.csv",
    index=False
)

# Measure labelled-pair survival
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

pairs["survives_filtered_lsh"] = np.any(
    np.all(signature_a == signature_b, axis=2),
    axis=1
)

pairs.to_csv(
    OUTPUT_FOLDER / "10_filtered_labelled_pair_results.csv",
    index=False
)

# Before/after metrics
baseline_same_recall = baseline_pairs.loc[
    baseline_pairs["label"] == "same",
    "survives_lsh"
].mean()

baseline_different_survival = baseline_pairs.loc[
    baseline_pairs["label"] == "different",
    "survives_lsh"
].mean()

filtered_same_recall = pairs.loc[
    pairs["label"] == "same",
    "survives_filtered_lsh"
].mean()

filtered_different_survival = pairs.loc[
    pairs["label"] == "different",
    "survives_filtered_lsh"
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
        "version": "high_frequency_filter",
        "average_candidates": filtered_workload["candidate_count"].mean(),
        "p95_candidates": filtered_workload["candidate_count"].quantile(0.95),
        "p99_candidates": filtered_workload["candidate_count"].quantile(0.99),
        "maximum_candidates": filtered_workload["candidate_count"].max(),
        "same_pair_candidate_recall": filtered_same_recall,
        "different_pair_survival_rate": filtered_different_survival
    }
])

summary.to_csv(
    OUTPUT_FOLDER / "10_final_before_after_summary.csv",
    index=False
)

print("\n--- BASELINE / FILTERED COMPARISON ---")
print(summary.round(4).to_string(index=False))

# Find expensive portals after filtering
portal_summary = (
    filtered_workload.groupby("portal_id")["candidate_count"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- TOP 15 PORTALS AFTER FILTERING ---")
print(portal_summary.head(15).round(2))

print("\nMinHash runtime (seconds):", round(minhash_runtime, 2))
print("LSH runtime (seconds):", round(lsh_runtime, 2))

print("\nSaved: outputs/10_filtered_minhash_signatures.npy")
print("Saved: outputs/10_filtered_candidate_workload.csv")
print("Saved: outputs/10_filtered_labelled_pair_results.csv")
print("Saved: outputs/10_final_before_after_summary.csv")