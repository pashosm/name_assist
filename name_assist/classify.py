"""SSA-derived classifications: trend, commonality, peak era, syllables."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------

def count_syllables(name: str) -> int:
    """Heuristic syllable count based on vowel groups."""
    name = name.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in name:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent trailing 'e' adjustment
    if name.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def syllable_label(name: str) -> str:
    """Classify name length by syllable count."""
    s = count_syllables(name)
    if s <= 2:
        return "Short"
    elif s == 3:
        return "Medium"
    else:
        return "Long"


# ---------------------------------------------------------------------------
# Rate computation helpers
# ---------------------------------------------------------------------------

def _compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'rate' column = count / total births for that sex+year."""
    year_totals = df.groupby(["year", "sex"])["count"].sum().rename("year_total")
    merged = df.merge(year_totals, on=["year", "sex"])
    merged["rate"] = merged["count"] / merged["year_total"]
    return merged


def _decade(year: int) -> str:
    """Convert year to decade label, e.g. 1990 -> '1990s'."""
    return f"{(year // 10) * 10}s"


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

def classify_trend(name_rates: pd.DataFrame) -> str:
    """
    Classify a name's popularity trend given its yearly rate data.

    name_rates: DataFrame with columns [year, rate] for a single name+sex.
    Returns one of: Timeless, Rising, Trendy, Declining, Old-fashioned,
                    Comeback, Rare, Steady
    """
    if name_rates.empty:
        return "Rare"

    total_count = name_rates["count"].sum() if "count" in name_rates.columns else 0

    # Aggregate by decade
    nr = name_rates.copy()
    nr["decade"] = nr["year"].apply(_decade)
    decade_rates = nr.groupby("decade")["rate"].mean()
    n_decades = len(decade_rates)

    if total_count < 1000 or n_decades < 3:
        return "Rare"

    historical_mean = decade_rates.mean()
    peak_decade = decade_rates.idxmax()
    peak_rate = decade_rates.max()

    # Recent = last available decade(s)
    sorted_decades = sorted(decade_rates.index)
    recent_decades = sorted_decades[-2:]  # last 2 decades
    recent_rate = decade_rates[recent_decades].mean()

    # Recent slope: compare last 2 decades
    if len(sorted_decades) >= 2:
        last = decade_rates[sorted_decades[-1]]
        prev = decade_rates[sorted_decades[-2]]
        recent_slope = (last - prev) / max(prev, 1e-10)
    else:
        recent_slope = 0.0

    peak_year_int = int(peak_decade[:-1])  # e.g. "1990s" -> 1990
    latest_decade_int = int(sorted_decades[-1][:-1])

    # Classification rules (order matters — first match wins)

    # Old-fashioned: peaked before 1960, very low now
    if peak_year_int < 1960 and recent_rate < 0.2 * peak_rate:
        # Check for comeback: recent uptick
        if recent_slope > 0.3 and recent_rate > 0.05 * peak_rate:
            return "Comeback"
        return "Old-fashioned"

    # Timeless: stays within a band across most decades
    within_band = ((decade_rates > 0.3 * historical_mean) &
                   (decade_rates < 3.0 * historical_mean)).sum()
    if within_band >= 0.7 * n_decades and n_decades >= 8:
        return "Timeless"

    # Rising: recent rate well above historical, still climbing
    if (recent_rate > 1.5 * historical_mean and
            recent_slope > 0.1 and
            peak_year_int >= latest_decade_int - 10):
        return "Rising"

    # Trendy: sharp peak in recent decades, now declining
    if (peak_rate > 3 * historical_mean and
            peak_year_int >= 1970 and
            recent_rate < 0.6 * peak_rate):
        return "Trendy"

    # Declining: was moderately popular, dropping steadily
    if (recent_rate < 0.3 * peak_rate and
            peak_year_int < latest_decade_int - 20):
        return "Declining"

    return "Steady"


# ---------------------------------------------------------------------------
# Commonality classification
# ---------------------------------------------------------------------------

COMMONALITY_LABELS = [
    (0.01, "Very Common"),
    (0.05, "Common"),
    (0.20, "Moderate"),
    (0.50, "Uncommon"),
    (1.00, "Rare"),
]


def classify_commonality(rank: int, total_names: int) -> str:
    """Assign commonality label based on rank percentile."""
    percentile = rank / total_names
    for threshold, label in COMMONALITY_LABELS:
        if percentile <= threshold:
            return label
    return "Rare"


# ---------------------------------------------------------------------------
# Build summary for all names
# ---------------------------------------------------------------------------

def _classify_group(group: pd.DataFrame) -> pd.Series:
    """Classify trend and peak decade for a single (name, sex) group."""
    group = group.copy()
    group["decade"] = group["year"].apply(_decade)
    trend = classify_trend(group[["year", "rate", "count"]])
    if not group.empty:
        peak = group.groupby("decade")["rate"].mean().idxmax()
    else:
        peak = "Unknown"
    return pd.Series({"trend": trend, "peak_decade": peak})


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute SSA-derived dimensions for every unique (name, sex) pair.

    Returns DataFrame with columns:
        name, sex, total_count, rank, commonality, trend, peak_decade, syllables, length_label
    """
    rated = _compute_rates(df)

    # Total counts and ranks per sex
    totals = df.groupby(["name", "sex"])["count"].sum().reset_index()
    totals.rename(columns={"count": "total_count"}, inplace=True)
    totals["rank"] = totals.groupby("sex")["total_count"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Total unique names per sex for commonality
    sex_counts = totals.groupby("sex")["name"].count()

    # Commonality labels
    totals["commonality"] = totals.apply(
        lambda r: classify_commonality(r["rank"], sex_counts[r["sex"]]),
        axis=1,
    )

    # Syllable / length
    unique_names = totals["name"].unique()
    syl_map = {n: count_syllables(n) for n in unique_names}
    len_map = {n: syllable_label(n) for n in unique_names}
    totals["syllables"] = totals["name"].map(syl_map)
    totals["length_label"] = totals["name"].map(len_map)

    # Trend and peak decade — use groupby for efficiency
    trend_data = rated.groupby(["name", "sex"]).apply(
        _classify_group, include_groups=False
    ).reset_index()

    totals = totals.merge(trend_data, on=["name", "sex"], how="left")
    totals["trend"] = totals["trend"].fillna("Rare")
    totals["peak_decade"] = totals["peak_decade"].fillna("Unknown")

    return totals.sort_values("total_count", ascending=False).reset_index(drop=True)
