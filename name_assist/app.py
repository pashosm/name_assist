"""Streamlit UI for name_assist — baby name discovery app."""

import os

import pandas as pd
import streamlit as st

from name_assist.classify import build_summary, count_syllables, syllable_label
from name_assist.data import get_name_series, get_year_totals, load_df
from name_assist.enrich import load_enrichments

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="name_assist", page_icon="\U0001f476", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Downloading and loading SSA data...")
def cached_load_df():
    return load_df()


@st.cache_data(show_spinner="Computing name classifications...")
def cached_build_summary(_df):
    return build_summary(_df)


@st.cache_data(show_spinner="Loading enrichments...")
def cached_load_enrichments():
    return load_enrichments()


@st.cache_data
def cached_year_totals(_df):
    return get_year_totals(_df)


# ---------------------------------------------------------------------------
# Constants for filter options
# ---------------------------------------------------------------------------

TREND_OPTIONS = [
    "Timeless", "Rising", "Trendy", "Declining",
    "Old-fashioned", "Comeback", "Rare", "Steady",
]

COMMONALITY_OPTIONS = ["Very Common", "Common", "Moderate", "Uncommon", "Rare"]

LENGTH_OPTIONS = ["Short", "Medium", "Long"]

CLASS_OPTIONS = ["Upper", "Upper-middle", "Middle", "Working", "Lower"]

HERITAGE_OPTIONS = [
    "English", "Hebrew", "Greek", "German", "Irish", "Spanish",
    "Arabic", "Japanese", "African", "French", "Latin", "Scandinavian",
    "Slavic", "Italian", "Scottish", "Welsh", "Persian", "Sanskrit",
    "Korean", "Thai", "Chinese", "Vietnamese", "Native American",
]

ORIGIN_OPTIONS = [
    "Biblical", "Mythological", "Literary", "Presidential", "Nature",
    "Virtue", "Invented", "Traditional", "Celebrity", "Place", "Occupational",
]

SENTIMENT_OPTIONS = [
    "Whimsical", "Mythical", "Stoic", "Strong", "Gentle", "Feminine",
    "Masculine", "Rural", "Urban", "Elegant", "Playful", "Serious",
    "Warm", "Cool", "Classic", "Modern", "Earthy", "Ethereal", "Bold", "Soft",
]

SOUND_OPTIONS = ["Soft", "Sharp", "Melodic", "Punchy", "Flowing", "Crisp"]

DECADE_OPTIONS = [f"{d}s" for d in range(1880, 2030, 10)]


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("name_assist")
    st.caption("Discover the perfect baby name across 10 dimensions")

    # Load data
    df = cached_load_df()
    summary = cached_build_summary(df)
    enrichments = cached_load_enrichments()
    has_enrichments = len(enrichments) > 0

    # Merge enrichment data into summary if available
    if has_enrichments:
        enrich_df = pd.DataFrame(enrichments.values())
        enrich_df.rename(columns={"name": "enrich_name"}, inplace=True)
        # Normalize for merge
        summary["name_lower"] = summary["name"].str.lower()
        enrich_df["name_lower"] = enrich_df["enrich_name"].str.lower()
        summary = summary.merge(
            enrich_df[["name_lower", "class", "heritage", "origin_type",
                        "sentiment_tags", "sound"]],
            on="name_lower",
            how="left",
        )
    else:
        summary["class"] = None
        summary["heritage"] = None
        summary["origin_type"] = None
        summary["sentiment_tags"] = None
        summary["sound"] = None

    # ----- Filters -----
    st.markdown("### Filters")

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        sex_filter = st.selectbox("Sex", ["Both", "Female", "Male"])
    with col2:
        trend_filter = st.multiselect("Popularity Trend", TREND_OPTIONS)
    with col3:
        common_filter = st.multiselect("Commonness", COMMONALITY_OPTIONS)

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        decade_filter = st.multiselect("Peak Era", DECADE_OPTIONS)
    with col5:
        length_filter = st.multiselect("Name Length", LENGTH_OPTIONS)
    with col6:
        class_filter = st.multiselect(
            "Class", CLASS_OPTIONS, disabled=not has_enrichments,
            help="Run enrichment to enable" if not has_enrichments else None,
        )

    # Row 3
    col7, col8, col9 = st.columns(3)
    with col7:
        heritage_filter = st.multiselect(
            "Heritage", HERITAGE_OPTIONS, disabled=not has_enrichments,
            help="Run enrichment to enable" if not has_enrichments else None,
        )
    with col8:
        origin_filter = st.multiselect(
            "Origin Type", ORIGIN_OPTIONS, disabled=not has_enrichments,
            help="Run enrichment to enable" if not has_enrichments else None,
        )
    with col9:
        sentiment_filter = st.multiselect(
            "Sentiment", SENTIMENT_OPTIONS, disabled=not has_enrichments,
            help="Run enrichment to enable" if not has_enrichments else None,
        )

    # Row 4
    col10, col11 = st.columns(2)
    with col10:
        sound_filter = st.multiselect(
            "Sound", SOUND_OPTIONS, disabled=not has_enrichments,
            help="Run enrichment to enable" if not has_enrichments else None,
        )
    with col11:
        name_search = st.text_input("Name contains", placeholder="e.g. 'ann'")

    # ----- Apply filters -----
    filtered = summary.copy()

    if sex_filter == "Female":
        filtered = filtered[filtered["sex"] == "F"]
    elif sex_filter == "Male":
        filtered = filtered[filtered["sex"] == "M"]

    if trend_filter:
        filtered = filtered[filtered["trend"].isin(trend_filter)]

    if common_filter:
        filtered = filtered[filtered["commonality"].isin(common_filter)]

    if decade_filter:
        filtered = filtered[filtered["peak_decade"].isin(decade_filter)]

    if length_filter:
        filtered = filtered[filtered["length_label"].isin(length_filter)]

    if class_filter:
        filtered = filtered[filtered["class"].isin(class_filter)]

    if heritage_filter:
        filtered = filtered[filtered["heritage"].isin(heritage_filter)]

    if origin_filter:
        filtered = filtered[filtered["origin_type"].isin(origin_filter)]

    if sentiment_filter:
        # sentiment_tags is a list — check if any selected tag is present
        filtered = filtered[filtered["sentiment_tags"].apply(
            lambda tags: (
                isinstance(tags, list)
                and any(t in tags for t in sentiment_filter)
            )
        )]

    if sound_filter:
        filtered = filtered[filtered["sound"].isin(sound_filter)]

    if name_search:
        filtered = filtered[
            filtered["name"].str.lower().str.contains(name_search.lower())
        ]

    # Already sorted by total_count descending from build_summary

    # ----- Results -----
    st.markdown("---")
    st.markdown(f"### Results — {len(filtered):,} names match")

    if filtered.empty:
        st.info("No names match your current filters. Try broadening your selection.")
        return

    # Prepare display columns
    display_cols = ["name", "sex", "trend", "peak_decade", "commonality",
                    "total_count", "length_label"]
    col_config = {
        "name": st.column_config.TextColumn("Name"),
        "sex": st.column_config.TextColumn("Sex"),
        "trend": st.column_config.TextColumn("Trend"),
        "peak_decade": st.column_config.TextColumn("Peak Era"),
        "commonality": st.column_config.TextColumn("Commonality"),
        "total_count": st.column_config.NumberColumn("Total Count", format="%d"),
        "length_label": st.column_config.TextColumn("Length"),
    }

    if has_enrichments:
        display_cols.extend(["class", "heritage", "origin_type", "sound"])
        col_config.update({
            "class": st.column_config.TextColumn("Class"),
            "heritage": st.column_config.TextColumn("Heritage"),
            "origin_type": st.column_config.TextColumn("Origin"),
            "sound": st.column_config.TextColumn("Sound"),
        })

    # Show top 500 in the table for performance
    st.dataframe(
        filtered[display_cols].head(500),
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    if len(filtered) > 500:
        st.caption(f"Showing top 500 of {len(filtered):,} results. Narrow filters to see more.")

    # ----- Detail view -----
    st.markdown("---")
    st.markdown("### Name Detail")

    detail_options = filtered["name"].head(200).tolist()
    selected_name = st.selectbox(
        "Select a name to explore",
        options=[""] + detail_options,
        format_func=lambda x: "Choose a name..." if x == "" else x,
    )

    if selected_name:
        _render_detail(selected_name, filtered, df, enrichments)


def _render_detail(name: str, summary: pd.DataFrame, df: pd.DataFrame,
                   enrichments: dict):
    """Render detailed analysis for a selected name."""
    # Find matching rows in summary (could be M and/or F)
    matches = summary[summary["name"] == name]

    for _, row in matches.iterrows():
        sex_label = "Female" if row["sex"] == "F" else "Male"

        with st.expander(f"{name} ({sex_label})", expanded=True):
            # Info columns
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trend", row["trend"])
            c2.metric("Commonness", row["commonality"])
            c3.metric("Peak Era", row["peak_decade"])
            c4.metric("Total Count", f"{row['total_count']:,}")

            # Enrichment details
            name_lower = name.lower()
            if name_lower in enrichments:
                enrich = enrichments[name_lower]
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Class", enrich.get("class", "—"))
                c6.metric("Heritage", enrich.get("heritage", "—"))
                c7.metric("Origin", enrich.get("origin_type", "—"))
                c8.metric("Sound", enrich.get("sound", "—"))

                tags = enrich.get("sentiment_tags", [])
                if tags:
                    st.markdown(
                        "**Sentiment:** "
                        + "  ".join(f"`{t}`" for t in tags)
                    )

            # Popularity chart
            name_data = get_name_series(name, row["sex"], df)
            if not name_data.empty:
                year_totals = df.groupby(["year", "sex"])["count"].sum().reset_index()
                year_totals.rename(columns={"count": "year_total"}, inplace=True)
                chart_data = name_data.merge(
                    year_totals, on=["year", "sex"]
                )
                chart_data["rate_per_million"] = (
                    chart_data["count"] / chart_data["year_total"] * 1_000_000
                )
                chart_df = chart_data.set_index("year")[["rate_per_million"]]
                st.line_chart(chart_df, y="rate_per_million",
                              y_label="Per million births", x_label="Year")


if __name__ == "__main__":
    main()
