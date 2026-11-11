import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from lib.data import load_graph_parquet

st.title("Hotel Similarity Network")
st.caption("Interactive view of the exported 276-hotel similarity graph")

try:
    nodes = load_graph_parquet("hotel_nodes.parquet")
    edges = load_graph_parquet("hotel_edges.parquet")
except Exception as exc:
    st.error(str(exc))
    st.stop()

required_nodes = ["hotel_id", "community"]
required_edges = ["source", "target"]
missing_nodes = [c for c in required_nodes if c not in nodes.columns]
missing_edges = [c for c in required_edges if c not in edges.columns]
if missing_nodes or missing_edges:
    st.error(
        f"Graph schema mismatch. Missing node columns: {missing_nodes}. "
        f"Missing edge columns: {missing_edges}. "
        f"Node columns: {list(nodes.columns)}; edge columns: {list(edges.columns)}"
    )
    st.stop()

nodes = nodes.copy()
edges = edges.copy()
nodes["hotel_id"] = nodes["hotel_id"].astype(str)
edges["source"] = edges["source"].astype(str)
edges["target"] = edges["target"].astype(str)
nodes["community"] = nodes["community"].astype(int)

communities = sorted(nodes["community"].unique())
selected = st.multiselect(
    "Communities to display",
    communities,
    default=communities,
    format_func=lambda x: f"Community {x}",
)

visible_nodes = nodes[nodes["community"].isin(selected)].copy()
visible_ids = set(visible_nodes["hotel_id"])
visible_edges = edges[
    edges["source"].isin(visible_ids) & edges["target"].isin(visible_ids)
].copy()

st.metric("Visible hotels", len(visible_nodes))
st.caption(f"Visible connections: {len(visible_edges):,}")

# Build an interactive Plotly network from the saved coordinates generated
# deterministically with a spring layout. This avoids rebuilding Louvain.
import networkx as nx
G = nx.from_pandas_edgelist(
    visible_edges, source="source", target="target",
    edge_attr=True, create_using=nx.Graph()
)
G.add_nodes_from(visible_nodes["hotel_id"].tolist())
pos = nx.spring_layout(G, seed=67, k=0.45, iterations=80)

edge_x, edge_y = [], []
for source, target in G.edges():
    x0, y0 = pos[source]
    x1, y1 = pos[target]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=0.6, color="rgba(130,130,130,0.35)"),
    hoverinfo="none",
    name="Similarity links",
))

palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#637939",
]
for i, community in enumerate(communities):
    group = visible_nodes[visible_nodes["community"] == community]
    xs, ys, hover = [], [], []
    for _, row in group.iterrows():
        node_id = row["hotel_id"]
        x, y = pos[node_id]
        xs.append(x)
        ys.append(y)
        name = row.get("name", node_id)
        city = row.get("city", "")
        positive = row.get("pct_positive", None)
        hover.append(
            f"{name}<br>{city}<br>Community {community}"
            + (f"<br>Positive: {positive:.1%}" if pd.notna(positive) else "")
        )
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=9, color=palette[i % len(palette)]),
        text=hover, hoverinfo="text",
        name=f"Community {community}",
    ))

fig.update_layout(
    title="Hotel similarity network by Louvain community",
    showlegend=True,
    height=720,
    margin=dict(l=10, r=10, t=55, b=10),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Hotel details")
display_cols = [
    c for c in [
        "hotel_id", "name", "city", "community", "pct_positive",
        "service", "cleanliness", "value", "location", "rooms",
        "sleep_quality"
    ] if c in visible_nodes.columns
]
st.dataframe(
    visible_nodes[display_cols].sort_values("community"),
    use_container_width=True,
    hide_index=True,
)
