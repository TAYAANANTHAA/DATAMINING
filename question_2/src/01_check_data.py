from pathlib import Path
import pandas as pd

# Dataset folder location
DATA_FOLDER = Path(r"C:\Users\Huawei\Downloads\data_2 (1)\data_2")

# Read all notice CSV files inside notices folder
notice_files = list((DATA_FOLDER / "notices").glob("*.csv"))
notices = pd.concat(
    [pd.read_csv(file) for file in notice_files],
    ignore_index=True
)

# Read human-checked labels
pairs = pd.read_csv(DATA_FOLDER / "labelled_pairs.csv")

# Basic dataset checks
print("\n--- DATASET SUMMARY ---")
print("Total notices:", len(notices))
print("Unique notice IDs:", notices["notice_id"].nunique())
print("Total portals:", notices["portal_id"].nunique())

print("\n--- LABEL BALANCE ---")
print(pairs["label"].value_counts())
print("\nLabel percentages:")
print((pairs["label"].value_counts(normalize=True) * 100).round(2))

print("\n--- TOP 10 PORTALS BY NOTICE COUNT ---")
print(notices["portal_id"].value_counts().head(10))

print("\n--- NOTICE COLUMNS ---")
print(list(notices.columns))

# Save summary results for the report
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

summary = pd.DataFrame({
    "metric": [
        "total_notices",
        "unique_notice_ids",
        "total_portals",
        "same_pairs",
        "different_pairs"
    ],
    "value": [
        len(notices),
        notices["notice_id"].nunique(),
        notices["portal_id"].nunique(),
        (pairs["label"] == "same").sum(),
        (pairs["label"] == "different").sum()
    ]
})

summary.to_csv(output_folder / "01_dataset_summary.csv", index=False)

print("\nSaved: outputs/01_dataset_summary.csv")