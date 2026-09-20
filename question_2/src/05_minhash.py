from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasketch import MinHash

OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)

NUMBER_OF_HASHES = 400
SHINGLE_SIZE = 3


def make_shingles(text, size=3):
    """Turn text into overlapping groups of three words."""

    words = re.findall(r"[a-z0-9]+", str(text).lower())

    if len(words) < size:
        return set(words)

    return {
        " ".join(words[position:position + size])
        for position in range(len(words) - size + 1)
    }


def exact_jaccard(set_a, set_b):
    """Exact similarity using the original shingle sets."""

    union_size = len(set_a | set_b)

    if union_size == 0:
        return 0.0

    return len(set_a & set_b) / union_size


# Read cleaned notices and labels
notices = pd.read_csv(OUTPUT_FOLDER / "02_notices_cleaned.csv")
pairs = pd.read_csv(
    r"C:\Users\Huawei\Downloads\data_2 (1)\data_2\labelled_pairs.csv"
)

# We only need full shingle sets in memory for labelled notices.
labelled_ids = set(pairs["notice_id_a"]) | set(pairs["notice_id_b"])
labelled_shingles = {}

# One 400-number signature for every notice
signatures = np.zeros(
    (len(notices), NUMBER_OF_HASHES),
    dtype=np.uint64
)

print("\n--- BUILDING MINHASH SIGNATURES ---")
print("Notices:", len(notices))
print("Hashes per notice:", NUMBER_OF_HASHES)
print("Word shingle size:", SHINGLE_SIZE)

for row_number, row in notices.iterrows():

    shingles = make_shingles(row["clean_text"], SHINGLE_SIZE)

    signature = MinHash(
        num_perm=NUMBER_OF_HASHES,
        seed=42
    )

    for shingle in shingles:
        signature.update(shingle.encode("utf-8"))

    signatures[row_number] = signature.hashvalues

    # Keep exact text sets only for the manually labelled pairs
    if row["notice_id"] in labelled_ids:
        labelled_shingles[row["notice_id"]] = shingles

    if (row_number + 1) % 500 == 0:
        print(f"Completed {row_number + 1} of {len(notices)} notices")

# Save compact signatures and their notice-ID order
np.save(OUTPUT_FOLDER / "05_minhash_signatures.npy", signatures)

notices[["notice_id", "portal_id"]].to_csv(
    OUTPUT_FOLDER / "05_minhash_notice_ids.csv",
    index=False
)

# Map notice ID to MinHash row number
notice_row = pd.Series(
    notices.index,
    index=notices["notice_id"]
)

# Compare exact Jaccard with MinHash estimate on labelled pairs
exact_scores = []
estimated_scores = []

for _, pair in pairs.iterrows():

    id_a = pair["notice_id_a"]
    id_b = pair["notice_id_b"]

    exact = exact_jaccard(
        labelled_shingles[id_a],
        labelled_shingles[id_b]
    )

    row_a = notice_row[id_a]
    row_b = notice_row[id_b]

    estimated = np.mean(
        signatures[row_a] == signatures[row_b]
    )

    exact_scores.append(exact)
    estimated_scores.append(estimated)

pairs["exact_jaccard"] = exact_scores
pairs["minhash_estimate"] = estimated_scores
pairs["absolute_error"] = abs(
    pairs["exact_jaccard"] - pairs["minhash_estimate"]
)

print("\n--- MINHASH ACCURACY ---")
print("Mean absolute error:",
      round(pairs["absolute_error"].mean(), 4))
print("95th percentile error:",
      round(pairs["absolute_error"].quantile(0.95), 4))
print("Maximum error:",
      round(pairs["absolute_error"].max(), 4))

# Theoretical worst-case 95% error for 400 hashes
planned_error = 1.96 * 0.5 / np.sqrt(NUMBER_OF_HASHES)

print("\nPlanned worst-case 95% error:",
      round(planned_error, 4))

pairs.to_csv(
    OUTPUT_FOLDER / "05_minhash_pair_errors.csv",
    index=False
)

# Plot exact similarity against estimated similarity
plt.figure(figsize=(7, 7))

for label, colour in [("different", "tomato"), ("same", "steelblue")]:
    subset = pairs[pairs["label"] == label]

    plt.scatter(
        subset["exact_jaccard"],
        subset["minhash_estimate"],
        alpha=0.6,
        label=label,
        color=colour
    )

plt.plot([0, 1], [0, 1], "--", color="black")
plt.title("Exact Jaccard versus 400-hash MinHash estimate")
plt.xlabel("Exact Jaccard similarity")
plt.ylabel("Estimated MinHash similarity")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_FOLDER / "05_minhash_accuracy.png",
    dpi=150
)

print("\nSaved: outputs/05_minhash_signatures.npy")
print("Saved: outputs/05_minhash_pair_errors.csv")
print("Saved: outputs/05_minhash_accuracy.png")