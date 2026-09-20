from pathlib import Path
import hashlib
import sqlite3
import time
import uuid

import numpy as np
import pandas as pd

OUTPUT_FOLDER = Path("outputs")

DATABASE_FILE = OUTPUT_FOLDER / "dedup.db"
NUMBER_OF_BANDS = 80
ROWS_PER_BAND = 5


def bucket_hash(signature_part):
    """Create a stable text bucket key for one LSH band."""

    return hashlib.blake2b(
        signature_part.tobytes(),
        digest_size=8
    ).hexdigest()


# Read prepared data
notices = pd.read_csv(OUTPUT_FOLDER / "02_notices_cleaned.csv")
signatures = np.load(OUTPUT_FOLDER / "05_minhash_signatures.npy")

assert NUMBER_OF_BANDS * ROWS_PER_BAND == signatures.shape[1]

connection = sqlite3.connect(DATABASE_FILE)
cursor = connection.cursor()

# Foreign keys are required for relational integrity
cursor.execute("PRAGMA foreign_keys = ON")

# Main notice table
cursor.execute("""
CREATE TABLE IF NOT EXISTS notices (
    notice_id TEXT PRIMARY KEY,
    portal_id TEXT NOT NULL,
    published_at TEXT,
    title TEXT,
    body TEXT,
    clean_text TEXT,
    estimated_value REAL,
    closing_date TEXT
)
""")

# Persistent LSH retrieval structure
cursor.execute("""
CREATE TABLE IF NOT EXISTS lsh_buckets (
    band_no INTEGER NOT NULL,
    bucket_hash TEXT NOT NULL,
    notice_id TEXT NOT NULL,
    PRIMARY KEY (band_no, bucket_hash, notice_id),
    FOREIGN KEY (notice_id) REFERENCES notices(notice_id)
)
""")

# B-tree index used by the candidate lookup path
cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_lsh_lookup
ON lsh_buckets (band_no, bucket_hash)
""")

# Stable opportunity/card identity
cursor.execute("""
CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS notice_opportunity (
    notice_id TEXT PRIMARY KEY,
    opportunity_id TEXT NOT NULL,
    confidence REAL,
    model_version TEXT NOT NULL,
    assigned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notice_id) REFERENCES notices(notice_id),
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id)
)
""")

# If two old opportunities are later determined to be the same,
# the old card can redirect without breaking an existing bookmark.
cursor.execute("""
CREATE TABLE IF NOT EXISTS opportunity_alias (
    old_opportunity_id TEXT PRIMARY KEY,
    canonical_opportunity_id TEXT NOT NULL,
    FOREIGN KEY (old_opportunity_id) REFERENCES opportunities(opportunity_id),
    FOREIGN KEY (canonical_opportunity_id) REFERENCES opportunities(opportunity_id)
)
""")

print("\n--- INSERTING NOTICES ---")

notice_rows = []

for _, row in notices.iterrows():

    estimated_value = row["estimated_value"]

    if pd.isna(estimated_value):
        estimated_value = None

    notice_rows.append((
        row["notice_id"],
        row["portal_id"],
        row["published_at"],
        row["title"],
        row["body"],
        row["clean_text"],
        estimated_value,
        row["closing_date"]
    ))

cursor.executemany("""
INSERT OR IGNORE INTO notices (
    notice_id, portal_id, published_at, title, body,
    clean_text, estimated_value, closing_date
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", notice_rows)

connection.commit()

print("Notices stored:", len(notice_rows))

print("\n--- INSERTING LSH BUCKETS ---")

bucket_rows = []
start_time = time.perf_counter()

for row_number, row in notices.iterrows():

    for band_no in range(NUMBER_OF_BANDS):

        start = band_no * ROWS_PER_BAND
        end = start + ROWS_PER_BAND

        bucket_rows.append((
            band_no,
            bucket_hash(signatures[row_number, start:end]),
            row["notice_id"]
        ))

    # Insert periodically instead of keeping all 960,000 rows in memory
    if len(bucket_rows) >= 20000:

        cursor.executemany("""
        INSERT OR IGNORE INTO lsh_buckets (
            band_no, bucket_hash, notice_id
        )
        VALUES (?, ?, ?)
        """, bucket_rows)

        connection.commit()
        bucket_rows = []

        print(f"Processed {row_number + 1} of {len(notices)} notices")

# Insert remaining bucket rows
if bucket_rows:

    cursor.executemany("""
    INSERT OR IGNORE INTO lsh_buckets (
        band_no, bucket_hash, notice_id
    )
    VALUES (?, ?, ?)
    """ , bucket_rows)

    connection.commit()

bucket_insert_time = time.perf_counter() - start_time

print("\n--- CREATING STABLE INITIAL OPPORTUNITY IDs ---")

for notice_id in notices["notice_id"]:

    existing = cursor.execute("""
        SELECT opportunity_id
        FROM notice_opportunity
        WHERE notice_id = ?
    """, (notice_id,)).fetchone()

    # Existing mappings are never overwritten on rerun.
    if existing is None:

        opportunity_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO opportunities (opportunity_id)
            VALUES (?)
        """, (opportunity_id,))

        cursor.execute("""
            INSERT INTO notice_opportunity (
                notice_id, opportunity_id, confidence, model_version
            )
            VALUES (?, ?, ?, ?)
        """, (
            notice_id,
            opportunity_id,
            1.0,
            "initial-ingestion"
        ))

connection.commit()

# Demonstrate the indexed lookup path using one notice
sample_notice_id = notices.iloc[0]["notice_id"]
sample_row = 0

query_start = time.perf_counter()
candidate_ids = set()
rows_returned = 0

for band_no in range(NUMBER_OF_BANDS):

    start = band_no * ROWS_PER_BAND
    end = start + ROWS_PER_BAND

    key = bucket_hash(signatures[sample_row, start:end])

    found = cursor.execute("""
        SELECT notice_id
        FROM lsh_buckets
        WHERE band_no = ? AND bucket_hash = ?
    """, (band_no, key)).fetchall()

    rows_returned += len(found)
    candidate_ids.update(row[0] for row in found)

candidate_ids.discard(sample_notice_id)

lookup_time_ms = (time.perf_counter() - query_start) * 1000

# Show SQLite's physical access plan
plan = cursor.execute("""
EXPLAIN QUERY PLAN
SELECT notice_id
FROM lsh_buckets
WHERE band_no = ? AND bucket_hash = ?
""", (0, bucket_hash(signatures[sample_row, 0:ROWS_PER_BAND]))).fetchall()

# Compare with a full scan of notices
scan_start = time.perf_counter()

all_notice_ids = cursor.execute("""
SELECT notice_id
FROM notices
""").fetchall()

full_scan_time_ms = (time.perf_counter() - scan_start) * 1000

# Database counts
notice_count = cursor.execute("""
SELECT COUNT(*) FROM notices
""").fetchone()[0]

bucket_count = cursor.execute("""
SELECT COUNT(*) FROM lsh_buckets
""").fetchone()[0]

opportunity_count = cursor.execute("""
SELECT COUNT(*) FROM opportunities
""").fetchone()[0]

print("\n--- DATABASE SUMMARY ---")
print("Database file:", DATABASE_FILE)
print("Stored notices:", notice_count)
print("Stored LSH bucket rows:", bucket_count)
print("Initial stable opportunity IDs:", opportunity_count)
print("Bucket insertion time (seconds):", round(bucket_insert_time, 2))

print("\n--- INDEXED LOOKUP DEMONSTRATION ---")
print("Sample notice:", sample_notice_id)
print("Bucket rows returned:", rows_returned)
print("Unique candidate notices:", len(candidate_ids))
print("Indexed lookup time (ms):", round(lookup_time_ms, 3))
print("Full notices-table scan time (ms):", round(full_scan_time_ms, 3))

print("\n--- SQLITE QUERY PLAN ---")
for plan_row in plan:
    print(plan_row)

# Save measurements and access-plan proof for the report
report = f"""DATABASE ACCESS-PATH REPORT

Database file: {DATABASE_FILE}
Notices stored: {notice_count}
LSH bucket rows stored: {bucket_count}
Stable opportunity IDs: {opportunity_count}
LSH-bucket insertion time (seconds): {bucket_insert_time:.2f}

Chosen lookup path:
80 B-tree point lookups on (band_no, bucket_hash),
followed by union and de-duplication of returned notice IDs.

Sample notice: {sample_notice_id}
Bucket rows returned: {rows_returned}
Unique candidate notices: {len(candidate_ids)}
Indexed lookup time (ms): {lookup_time_ms:.3f}
Full notices-table scan time (ms): {full_scan_time_ms:.3f}

SQLite query plan:
{plan}
"""

(OUTPUT_FOLDER / "07_database_access_report.txt").write_text(
    report,
    encoding="utf-8"
)

connection.close()

print("\nSaved: outputs/dedup.db")
print("Saved: outputs/07_database_access_report.txt")