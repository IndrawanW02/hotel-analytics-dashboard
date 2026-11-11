import streamlit as st

st.set_page_config(
    page_title="Hotel Customer Experience Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏨 Hotel Customer Experience & Performance Analytics")
st.caption("Hotel review analytics, aspect-based sentiment prediction, and hotel similarity networks")

st.markdown("""
Use the pages in the sidebar to explore four parts of the project:

- **Overview:** hotel performance, review sentiment, aspect ratings, and temporal trends.
- **ABSA Analysis:** predict the sentiment of a review for a selected hotel aspect.
- **Hotel Network:** explore the 276-hotel similarity graph and Louvain communities.
- **Community Insights:** compare community-level aspect ratings and identify benchmarking opportunities.
""")

st.info(
    "This application reads precomputed Gold datasets and graph outputs. "
    "It does not rerun the original 50-million-review Spark pipeline."
)

st.subheader("Data coverage")
st.markdown("""
The dashboard uses curated Gold data, saved scikit-learn ABSA artifacts, and
the exported NetworkX graph. The Hotels-50K image collection is not required
for these analytical pages.
""")
