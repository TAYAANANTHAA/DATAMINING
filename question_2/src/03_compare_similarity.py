from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Read cleaned notices and human-checked labels
notices = pd.read_csv(OUTPUT_FOLDER / "02_notices_cleaned.csv")
pairs = pd.read_csv(
    r"C:\Users\Huawei\Downloads\data_2 (1)\data_2\labelled_pairs.csv"
)

# Convert notice_id to its row number for fast lookup
notice_row = pd.Series(notices.index, index=notices["notice_id"])

# Find row numbers for every labelled pair
pairs["row_a"] = pairs["notice_id_a"].map(notice_row)
pairs["row_b"] = pairs["notice_id_b"].map(notice_row)

# Safety check: every labelled notice must exist
if pairs["row_a"].isna().any() or pairs["row_b"].isna().any():
    raise ValueError("Some labelled notice IDs were not found.")

pairs["row_a"] = pairs["row_a"].astype(int)
pairs["row_b"] = pairs["row_b"].astype(int)

texts = notices["clean_text"].fillna("")

print("\nBuilding WORD TF-IDF representation...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True,
    dtype=np.float32
)

word_matrix = word_vectorizer.fit_transform(texts)

# TF-IDF vectors are already normalised.
# Dot product = cosine similarity.
pairs["word_score"] = np.asarray(
    word_matrix[pairs["row_a"]]
    .multiply(word_matrix[pairs["row_b"]])
    .sum(axis=1)
).ravel()

print("Building CHARACTER 5-GRAM TF-IDF representation...")
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(5, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True,
    dtype=np.float32
)

char_matrix = char_vectorizer.fit_transform(texts)

pairs["char5_score"] = np.asarray(
    char_matrix[pairs["row_a"]]
    .multiply(char_matrix[pairs["row_b"]])
    .sum(axis=1)
).ravel()

# Convert text label to 1/0 for AUROC
pairs["is_same"] = (pairs["label"] == "same").astype(int)

word_auc = roc_auc_score(pairs["is_same"], pairs["word_score"])
char_auc = roc_auc_score(pairs["is_same"], pairs["char5_score"])

print("\n--- SIMILARITY RESULTS ---")
print("Word TF-IDF AUROC:       ", round(word_auc, 4))
print("Character 5-gram AUROC:  ", round(char_auc, 4))

print("\n--- AVERAGE SCORE BY LABEL ---")
print(
    pairs.groupby("label")[["word_score", "char5_score"]]
    .mean()
    .round(4)
)

# Save every pair score for the report
pairs.to_csv(OUTPUT_FOLDER / "03_labelled_pair_scores.csv", index=False)

# Make a graph: same and different pair score distributions
plt.figure(figsize=(10, 6))

for label, colour in [("different", "tomato"), ("same", "steelblue")]:
    values = pairs.loc[pairs["label"] == label, "char5_score"]
    plt.hist(
        values,
        bins=30,
        alpha=0.60,
        label=label,
        color=colour
    )

plt.title("Character 5-gram similarity: same vs different pairs")
plt.xlabel("Cosine similarity score")
plt.ylabel("Number of labelled pairs")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_FOLDER / "03_char5_score_distribution.png",
    dpi=150
)

print("\nSaved: outputs/03_labelled_pair_scores.csv")
print("Saved: outputs/03_char5_score_distribution.png")