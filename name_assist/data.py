"""Download, parse, and cache SSA baby names data."""

import io
import os
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "names.db"
ZIP_PATH = DATA_DIR / "names.zip"
SSA_URL = "https://www.ssa.gov/oact/babynames/names.zip"


def _download_zip() -> None:
    """Download the SSA names zip file if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        return
    print("Downloading SSA baby names data...")
    resp = requests.get(SSA_URL, timeout=120)
    resp.raise_for_status()
    ZIP_PATH.write_bytes(resp.content)
    print("Download complete.")


def _parse_zip_to_db() -> None:
    """Extract all yobYYYY.txt files from the zip and load into SQLite."""
    if DB_PATH.exists():
        return

    _download_zip()
    print("Parsing SSA data into database...")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE names (name TEXT, sex TEXT, year INT, count INT)"
    )

    year_pattern = re.compile(r"yob(\d{4})\.txt")
    rows = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for entry in zf.namelist():
            match = year_pattern.search(entry)
            if not match:
                continue
            year = int(match.group(1))
            with zf.open(entry) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                for line in text:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue
                    name, sex, count = parts[0], parts[1], int(parts[2])
                    rows.append((name, sex, year, count))

            # Batch insert per year to keep memory reasonable
            if len(rows) > 100_000:
                conn.executemany(
                    "INSERT INTO names VALUES (?, ?, ?, ?)", rows
                )
                rows.clear()

    if rows:
        conn.executemany("INSERT INTO names VALUES (?, ?, ?, ?)", rows)

    conn.execute("CREATE INDEX idx_name_sex ON names (name, sex)")
    conn.execute("CREATE INDEX idx_year ON names (year)")
    conn.commit()
    conn.close()
    print("Database ready.")


def load_df() -> pd.DataFrame:
    """Load the full names DataFrame from SQLite. Call once; cache externally."""
    _parse_zip_to_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM names", conn)
    conn.close()
    return df


def get_year_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total births per year per sex."""
    return df.groupby(["year", "sex"])["count"].sum().reset_index()


def get_name_series(name: str, sex: str, df: pd.DataFrame) -> pd.DataFrame:
    """Get year-by-year data for a specific name and sex."""
    mask = (df["name"].str.lower() == name.lower()) & (df["sex"] == sex)
    return df[mask].sort_values("year")
