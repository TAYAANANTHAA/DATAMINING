from pathlib import Path
import re
import unicodedata
import pandas as pd

DATA_FOLDER = Path(r"C:\Users\Huawei\Downloads\data_2 (1)\data_2")
OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)


def clean_text(value):
    """Convert one title/body into a consistent comparison form."""

    if pd.isna(value):
        return ""

    text = str(value)

    # Make characters consistent, for example curly quotes / strange spaces
    text = unicodedata.normalize("NFKD", text)

    # Upper/lower-case differences should not matter
    text = text.lower()

    # Remove known aggregator names mentioned in portal_profiles.md
    text = re.sub(
        r"national procurement aggregation service",
        " ",
        text
    )
    text = re.sub(
        r"state procurement cell",
        " ",
        text
    )

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Turn repeated spaces/newlines into one space
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Read every notice CSV
notice_files = list((DATA_FOLDER / "notices").glob("*.csv"))
notices = pd.concat(
    [pd.read_csv(file) for file in notice_files],
    ignore_index=True
)

# Keep the original title/body; create an additional cleaned text field
notices["combined_text"] = (
    notices["title"].fillna("") + " " + notices["body"].fillna("")
)

notices["clean_text"] = notices["combined_text"].apply(clean_text)

# Useful quality-check columns
notices["original_length"] = notices["combined_text"].str.len()
notices["clean_length"] = notices["clean_text"].str.len()

# Save it for the next programs
output_file = OUTPUT_FOLDER / "02_notices_cleaned.csv"
notices.to_csv(output_file, index=False)

print("\n--- TEXT PREPARATION COMPLETE ---")
print("Notices cleaned:", len(notices))
print("Average original text length:",
      round(notices["original_length"].mean(), 2))
print("Average cleaned text length:",
      round(notices["clean_length"].mean(), 2))

print("\n--- EXAMPLE ---")
example = notices.iloc[0]

print("\nOriginal title:")
print(example["title"])

print("\nCleaned text (first 300 characters):")
print(example["clean_text"][:300])

print("\nSaved:", output_file)