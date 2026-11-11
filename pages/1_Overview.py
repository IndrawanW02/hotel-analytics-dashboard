import streamlit as st
import pandas as pd
import plotly.express as px
from lib.data import load_parquet_dataset, ASPECTS, format_number

st.title("Hotel Performance Overview")
st.caption("Aggregated insights from the Gold layer")

try:
    hotels = load_parquet_dataset("hotel_performance")
    aspects = load_parquet_dataset("aspect_ratings")
    temporal = load_parquet_dataset("temporal_trends")
except Exception as exc:
    st.error(str(exc))
    st.stop()

required = [
    "hotel_url", "review_count", "avg_rating",
    "positive_reviews", "neutral_reviews", "negative_reviews",
    "positive_pct", "neutral_pct", "negative_pct"
]
missing = [c for c in required if c not in hotels.columns]
if missing:
    st.error(f"hotel_performance is missing columns: {missing}")
    st.stop()

total_reviews = int(hotels["review_count"].sum())
hotel_count = len(hotels)
positive = int(hotels["positive_reviews"].sum())
neutral = int(hotels["neutral_reviews"].sum())
negative = int(hotels["negative_reviews"].sum())
overall_rating = (
    (hotels["avg_rating"] * hotels["review_count"]).sum()
    / hotels["review_count"].sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hotels", format_number(hotel_count))
c2.metric("Reviews", format_number(total_reviews))
c3.metric("Weighted average rating", f"{overall_rating:.2f} / 5")
c4.metric("Positive reviews", f"{positive / total_reviews:.1%}")

st.divider()
left, right = st.columns(2)

with left:
    sentiment = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Reviews": [positive, neutral, negative],
    })
    fig = px.pie(
        sentiment, names="Sentiment", values="Reviews",
        hole=0.55, title="Overall review sentiment"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    rating_bins = hotels["avg_rating"].dropna()
    fig = px.histogram(
        x=rating_bins, nbins=25,
        labels={"x": "Average hotel rating", "y": "Number of hotels"},
        title="Distribution of average hotel ratings"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Aspect ratings")
if {"aspect", "avg_aspect_rating"}.issubset(aspects.columns):
    aspect_summary = (
        aspects.groupby("aspect", as_index=False)
        .agg(
            avg_aspect_rating=("avg_aspect_rating", "mean"),
            hotels=("hotel_url", "nunique")
        )
        .sort_values("avg_aspect_rating")
    )
    fig = px.bar(
        aspect_summary, x="avg_aspect_rating", y="aspect",
        orientation="h", text_auto=".2f",
        title="Average rating by hotel aspect",
        labels={
            "avg_aspect_rating": "Average aspect rating",
            "aspect": "Aspect",
        },
    )
    fig.update_xaxes(range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Aspect dataset does not contain the expected columns.")

st.subheader("Review trends")
if {"year", "month", "review_count"}.issubset(temporal.columns):
    temporal = temporal.copy()
    temporal["period"] = pd.to_datetime(
        temporal[["year", "month"]].assign(day=1)
    )
    temporal = temporal.sort_values("period")
    fig = px.line(
        temporal, x="period", y="review_count",
        title="Monthly review volume",
        labels={"period": "Month", "review_count": "Number of reviews"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if "avg_rating" in temporal.columns:
        fig = px.line(
            temporal, x="period", y="avg_rating",
            title="Monthly average rating",
            labels={"period": "Month", "avg_rating": "Average rating"}
        )
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Temporal dataset does not contain the expected columns.")

with st.expander("Inspect Gold data"):
    st.write("Hotel performance", hotels.head(10))
    st.write("Aspect ratings", aspects.head(10))
    st.write("Temporal trends", temporal.head(10))
