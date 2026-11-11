import streamlit as st
import pandas as pd
import plotly.express as px
from lib.data import load_community_profiles, load_graph_parquet, ASPECTS

st.title("Community Insights")
st.caption("Louvain community profiles and hotel benchmarking")

try:
    profiles = load_community_profiles()
    nodes = load_graph_parquet("hotel_nodes.parquet")
except Exception as exc:
    st.error(str(exc))
    st.stop()

missing = [a for a in ASPECTS if a not in profiles.columns]
if missing:
    st.error(f"Community profile is missing aspect columns: {missing}")
    st.stop()

profiles.index = profiles.index.astype(int)
profiles.index.name = "community"

st.subheader("Aspect ratings relative to the overall hotel average")
heatmap_data = profiles[ASPECTS].copy()
heatmap_data.index = [
    f"C{idx} (n={int(profiles.loc[idx, 'n'])})"
    if "n" in profiles.columns else f"C{idx}"
    for idx in profiles.index
]
fig = px.imshow(
    heatmap_data,
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    text_auto="+.2f",
    aspect="auto",
    labels=dict(x="Hotel aspect", y="Louvain community", color="Difference"),
)
fig.update_layout(height=480)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Positive values indicate an aspect rating above the overall hotel average; "
    "negative values indicate below-average ratings."
)

st.subheader("Community comparison")
if "community" not in nodes.columns:
    st.error("hotel_nodes.parquet must contain a community column.")
    st.stop()

if "pct_positive" in nodes.columns:
    community_sentiment = (
        nodes.groupby("community", as_index=False)
        .agg(
            positive_pct=("pct_positive", "mean"),
            hotel_count=("hotel_id", "nunique")
        )
    )
    community_sentiment["positive_pct"] *= 100
    fig = px.bar(
        community_sentiment,
        x="community", y="positive_pct",
        text_auto=".1f",
        title="Average positive-review percentage by community",
        labels={
            "community": "Community",
            "positive_pct": "Average positive reviews (%)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        community_sentiment.sort_values("positive_pct", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "The exported node data does not contain pct_positive, "
        "so the community sentiment comparison is not available."
    )

st.subheader("Community aspect profile values")
st.dataframe(profiles, use_container_width=True)
