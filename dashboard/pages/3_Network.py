# dashboard/pages/3_Network.py

import sys
from pathlib import Path
import math

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SERVICE IMPORTS
# ============================================================

from optimizer_service import (
    get_graph,
    get_traffic_summary,
    get_traffic_records,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | Network",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "3_Network.css"

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


# ============================================================
# HELPER
# ============================================================

def html(content):
    st.html(content)


def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def get_node_id(node):
    return str(
        get_attr(
            node,
            "id",
            get_attr(node, "node_id", ""),
        )
    )


def get_edge_source(edge):
    source = get_attr(edge, "source", "")

    if isinstance(source, dict):
        return str(
            source.get(
                "id",
                source.get("node_id", ""),
            )
        )

    return str(
        get_attr(
            source,
            "id",
            source,
        )
    )


def get_edge_target(edge):
    target = get_attr(edge, "target", "")

    if isinstance(target, dict):
        return str(
            target.get(
                "id",
                target.get("node_id", ""),
            )
        )

    return str(
        get_attr(
            target,
            "id",
            target,
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        """
        <div class="network-sidebar-brand">

            <div class="network-sidebar-logo">
                ⚛
            </div>

            <div>
                <div class="network-sidebar-title">
                    Q-ROUTE
                </div>

                <div class="network-sidebar-subtitle">
                    Intelligent Traffic Optimization
                </div>
            </div>

        </div>
        """
    )

    st.markdown("---")

    st.markdown("### 🌐 Network Configuration")

    scenario = st.selectbox(
        "Traffic Scenario",
        [
            "Normal",
            "Morning Peak",
            "Heavy Congestion",
            "Evening Peak",
            "Accident",
            "Road Closure",
        ],
        index=0,
    )

    network_size = st.selectbox(
        "Network Size",
        [6, 50, 100, 250, 500],
        index=2,
        format_func=lambda x: f"{x} nodes",
    )

    edges_per_node = st.slider(
        "Edges per Node",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
    )

    st.caption(
        "The network graph is generated using the "
        "selected scale and connectivity."
    )


# ============================================================
# HEADER
# ============================================================

html(
    f"""
    <div class="network-header">

        <div class="network-status-row">

            <div class="network-status-ready">
                <span></span>
                NETWORK ONLINE
            </div>

            <div class="network-status-scenario">
                🚦 {scenario}
            </div>

            <div class="network-status-size">
                {network_size} NODES
            </div>

        </div>

        <div class="network-icon">
            🌐
        </div>

        <div class="network-eyebrow">
            TRANSPORTATION NETWORK MODEL
        </div>

        <h1>
            Network
            <span>Topology</span>
        </h1>

        <p>
            Explore the transportation graph used by
            Q-ROUTE for traffic-aware route optimization.
        </p>

    </div>
    """
)


# ============================================================
# GET GRAPH
# ============================================================

try:

    graph = get_graph(
        scenario=scenario,
        network_size=network_size,
        edges_per_node=edges_per_node,
        seed=42,
    )

except Exception as error:

    graph = None

    st.error(
        f"Unable to generate network: {error}"
    )


# ============================================================
# SERIALIZE GRAPH
# ============================================================

node_objects = []
edge_objects = []

if graph is not None:

    try:
        node_objects = list(
            get_attr(
                graph,
                "_nodes",
                [],
            )
        )
    except Exception:
        node_objects = []

    try:
        if hasattr(graph, "get_edges"):
            edge_objects = list(
                graph.get_edges()
            )
        else:
            edge_objects = list(
                get_attr(
                    graph,
                    "_edges",
                    [],
                )
            )
    except Exception:
        edge_objects = []


node_ids = [
    get_node_id(node)
    for node in node_objects
]

node_ids = [
    node_id
    for node_id in node_ids
    if node_id
]


# ============================================================
# NETWORK METRICS
# ============================================================

node_count = len(node_ids)

edge_count = len(edge_objects)

blocked_count = 0

total_distance = 0.0
total_travel_time = 0.0
total_congestion = 0.0

for edge in edge_objects:

    blocked = bool(
        get_attr(
            edge,
            "blocked",
            False,
        )
    )

    if blocked:
        blocked_count += 1

    total_distance += float(
        get_attr(
            edge,
            "distance",
            0,
        )
        or 0
    )

    total_travel_time += float(
        get_attr(
            edge,
            "travel_time",
            0,
        )
        or 0
    )

    total_congestion += float(
        get_attr(
            edge,
            "congestion",
            0,
        )
        or 0
    )


average_distance = (
    total_distance / edge_count
    if edge_count
    else 0
)

average_travel_time = (
    total_travel_time / edge_count
    if edge_count
    else 0
)

average_congestion = (
    total_congestion / edge_count
    if edge_count
    else 0
)


# ============================================================
# NETWORK OVERVIEW
# ============================================================

st.markdown("### 🌐 Network Overview")

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)


with metric_1:

    html(
        f"""
        <div class="network-metric-card network-green">

            <div class="network-metric-label">
                NODES
            </div>

            <div class="network-metric-value">
                {node_count}
            </div>

            <div class="network-metric-description">
                Network intersections
            </div>

        </div>
        """
    )


with metric_2:

    html(
        f"""
        <div class="network-metric-card network-yellow">

            <div class="network-metric-label">
                EDGES
            </div>

            <div class="network-metric-value">
                {edge_count}
            </div>

            <div class="network-metric-description">
                Directed road connections
            </div>

        </div>
        """
    )


with metric_3:

    html(
        f"""
        <div class="network-metric-card network-red">

            <div class="network-metric-label">
                BLOCKED
            </div>

            <div class="network-metric-value">
                {blocked_count}
            </div>

            <div class="network-metric-description">
                Unavailable roads
            </div>

        </div>
        """
    )


with metric_4:

    html(
        f"""
        <div class="network-metric-card network-green">

            <div class="network-metric-label">
                AVG TIME
            </div>

            <div class="network-metric-value">
                {average_travel_time:.2f}
            </div>

            <div class="network-metric-description">
                Travel time per edge
            </div>

        </div>
        """
    )


with metric_5:

    html(
        f"""
        <div class="network-metric-card network-yellow">

            <div class="network-metric-label">
                AVG CONGESTION
            </div>

            <div class="network-metric-value">
                {average_congestion * 100:.1f}%
            </div>

            <div class="network-metric-description">
                Network traffic level
            </div>

        </div>
        """
    )


# ============================================================
# NETWORK VISUALIZATION
# ============================================================

st.markdown("### 🗺️ Transportation Network")


def build_network_figure(
    node_ids,
    edge_objects,
    max_labels=100,
):

    fig = go.Figure()

    if not node_ids:
        return fig


    # --------------------------------------------------------
    # Circular coordinates
    # --------------------------------------------------------

    coordinates = {}

    total_nodes = len(node_ids)

    radius = 10

    for index, node_id in enumerate(node_ids):

        angle = (
            2
            * math.pi
            * index
            / total_nodes
        )

        coordinates[node_id] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
        )


    # --------------------------------------------------------
    # Background edges
    # --------------------------------------------------------

    edge_x = []
    edge_y = []

    for edge in edge_objects:

        source = get_edge_source(edge)
        target = get_edge_target(edge)

        if (
            source not in coordinates
            or target not in coordinates
        ):
            continue

        x1, y1 = coordinates[source]
        x2, y2 = coordinates[target]

        edge_x.extend(
            [
                x1,
                x2,
                None,
            ]
        )

        edge_y.extend(
            [
                y1,
                y2,
                None,
            ]
        )


    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(
                width=0.9,
            ),
            opacity=0.30,
            hoverinfo="none",
            name="Road Network",
        )
    )


    # --------------------------------------------------------
    # Blocked edges
    # --------------------------------------------------------

    blocked_x = []
    blocked_y = []

    for edge in edge_objects:

        if not bool(
            get_attr(
                edge,
                "blocked",
                False,
            )
        ):
            continue

        source = get_edge_source(edge)
        target = get_edge_target(edge)

        if (
            source not in coordinates
            or target not in coordinates
        ):
            continue

        x1, y1 = coordinates[source]
        x2, y2 = coordinates[target]

        blocked_x.extend(
            [
                x1,
                x2,
                None,
            ]
        )

        blocked_y.extend(
            [
                y1,
                y2,
                None,
            ]
        )


    if blocked_x:

        fig.add_trace(
            go.Scatter(
                x=blocked_x,
                y=blocked_y,
                mode="lines",
                line=dict(
                    width=4,
                ),
                name="Blocked Roads",
            )
        )


    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    node_x = []
    node_y = []

    for node_id in node_ids:

        x, y = coordinates[node_id]

        node_x.append(x)
        node_y.append(y)


    show_labels = (
        len(node_ids) <= max_labels
    )


    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode=(
                "markers+text"
                if show_labels
                else "markers"
            ),
            text=(
                node_ids
                if show_labels
                else None
            ),
            textposition="top center",
            marker=dict(
                size=(
                    8
                    if len(node_ids) <= 100
                    else 5
                ),
            ),
            customdata=node_ids,
            hovertemplate=(
                "Node: %{customdata}"
                "<extra></extra>"
            ),
            name="Intersections",
        )
    )


    # --------------------------------------------------------
    # Depot
    # --------------------------------------------------------

    if node_ids:

        depot = node_ids[0]

        x, y = coordinates[depot]

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=17,
                    symbol="star",
                ),
                name="Depot",
                hovertemplate=(
                    "Depot: "
                    + depot
                    + "<extra></extra>"
                ),
            )
        )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        height=690,

        margin=dict(
            l=15,
            r=15,
            t=25,
            b=15,
        ),

        showlegend=True,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),

        xaxis=dict(
            visible=False,
        ),

        yaxis=dict(
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),

        hovermode="closest",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(255,255,255,0.65)",
    )

    return fig


network_fig = build_network_figure(
    node_ids,
    edge_objects,
)


st.plotly_chart(
    network_fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
    },
)


# ============================================================
# NETWORK CHARACTERISTICS
# ============================================================

st.markdown("### 📐 Network Characteristics")

characteristic_col1, characteristic_col2 = st.columns(2)


with characteristic_col1:

    html(
        f"""
        <div class="network-info-card">

            <div class="network-info-title">
                🛣️ Road Characteristics
            </div>

            <div class="network-info-row">

                <span>
                    Average Distance
                </span>

                <strong>
                    {average_distance:.2f}
                </strong>

            </div>

            <div class="network-info-row">

                <span>
                    Average Travel Time
                </span>

                <strong>
                    {average_travel_time:.2f}
                </strong>

            </div>

            <div class="network-info-row">

                <span>
                    Average Congestion
                </span>

                <strong>
                    {average_congestion * 100:.1f}%
                </strong>

            </div>

        </div>
        """
    )


with characteristic_col2:

    density = (
        edge_count / node_count
        if node_count
        else 0
    )

    html(
        f"""
        <div class="network-info-card">

            <div class="network-info-title">
                🔗 Connectivity
            </div>

            <div class="network-info-row">

                <span>
                    Edges per Node
                </span>

                <strong>
                    {density:.2f}
                </strong>

            </div>

            <div class="network-info-row">

                <span>
                    Configured Connectivity
                </span>

                <strong>
                    {edges_per_node}
                </strong>

            </div>

            <div class="network-info-row">

                <span>
                    Blocked Roads
                </span>

                <strong>
                    {blocked_count}
                </strong>

            </div>

        </div>
        """
    )


# ============================================================
# EDGE DATA
# ============================================================

st.markdown("### 🛣️ Road Edge Data")

if edge_objects:

    edge_rows = []

    for index, edge in enumerate(
        edge_objects,
        start=1,
    ):

        source = get_edge_source(edge)
        target = get_edge_target(edge)

        distance = get_attr(
            edge,
            "distance",
            0,
        )

        travel_time = get_attr(
            edge,
            "travel_time",
            0,
        )

        congestion = get_attr(
            edge,
            "congestion",
            0,
        )

        blocked = get_attr(
            edge,
            "blocked",
            False,
        )

        edge_rows.append(
            {
                "Edge": f"E{index}",
                "Source": source,
                "Target": target,
                "Distance": float(
                    distance or 0
                ),
                "Travel Time": float(
                    travel_time or 0
                ),
                "Congestion": (
                    float(
                        congestion or 0
                    ) * 100
                ),
                "Blocked": bool(
                    blocked
                ),
            }
        )


    edge_df = pd.DataFrame(
        edge_rows
    )

    st.dataframe(
        edge_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No road edges are available."
    )


# ============================================================
# TRAFFIC EFFECT ON NETWORK
# ============================================================

st.markdown("### 🚦 Traffic Effect on Network")

try:

    traffic_summary = get_traffic_summary(
        scenario
    )

except Exception:

    traffic_summary = {}


traffic_average = traffic_summary.get(
    "average_congestion",
    average_congestion,
)

traffic_blocked = traffic_summary.get(
    "blocked_edges",
    blocked_count,
)


html(
    f"""
    <div class="traffic-effect-card">

        <div class="traffic-effect-header">

            <div>
                <div class="traffic-effect-title">
                    Active Traffic Scenario
                </div>

                <div class="traffic-effect-scenario">
                    🚦 {scenario}
                </div>
            </div>

            <div class="traffic-effect-indicator">
                {"⚠️ RESTRICTIONS ACTIVE"
                 if traffic_blocked > 0
                 else "✅ NETWORK AVAILABLE"}
            </div>

        </div>

        <div class="traffic-effect-grid">

            <div>
                <div class="traffic-effect-label">
                    AVERAGE CONGESTION
                </div>

                <div class="traffic-effect-value">
                    {float(traffic_average) * 100:.1f}%
                </div>
            </div>

            <div>
                <div class="traffic-effect-label">
                    BLOCKED EDGES
                </div>

                <div class="traffic-effect-value">
                    {traffic_blocked}
                </div>
            </div>

            <div>
                <div class="traffic-effect-label">
                    OPTIMIZATION INPUT
                </div>

                <div class="traffic-effect-value">
                    UPDATED GRAPH
                </div>
            </div>

        </div>

    </div>
    """
)


# ============================================================
# NETWORK → OPTIMIZATION PIPELINE
# ============================================================

st.markdown("### 🔄 Network to Optimization")

html(
    """
    <div class="network-pipeline">

        <div class="pipeline-card pipeline-red">

            <div class="pipeline-number">
                01
            </div>

            <div class="pipeline-icon">
                🚦
            </div>

            <div class="pipeline-title">
                Traffic
            </div>

            <div class="pipeline-text">
                Traffic conditions define road
                congestion and availability.
            </div>

        </div>


        <div class="pipeline-arrow">
            →
        </div>


        <div class="pipeline-card pipeline-yellow">

            <div class="pipeline-number">
                02
            </div>

            <div class="pipeline-icon">
                🌐
            </div>

            <div class="pipeline-title">
                Weighted Graph
            </div>

            <div class="pipeline-text">
                Roads become weighted edges using
                distance, travel time and congestion.
            </div>

        </div>


        <div class="pipeline-arrow">
            →
        </div>


        <div class="pipeline-card pipeline-green">

            <div class="pipeline-number">
                03
            </div>

            <div class="pipeline-icon">
                🧠
            </div>

            <div class="pipeline-title">
                Optimization
            </div>

            <div class="pipeline-text">
                Q-ROUTE searches the updated graph
                for feasible vehicle routes.
            </div>

        </div>

    </div>
    """
)


# ============================================================
# NETWORK MODEL
# ============================================================

st.markdown("### 📘 Network Model")

html(
    """
    <div class="network-model-card">

        <div class="network-model-title">
            Transportation Network Representation
        </div>

        <div class="network-model-equation">
            G = (V, E)
        </div>

        <div class="network-model-text">

            <strong>V</strong> represents network nodes or
            intersections, while <strong>E</strong> represents
            directed road connections.

            Each road contains optimization information such as
            distance, travel time, congestion and availability.

        </div>

        <div class="network-model-tags">

            <span>Distance</span>
            <span>Travel Time</span>
            <span>Congestion</span>
            <span>Blocked Status</span>

        </div>

    </div>
    """
)


# ============================================================
# TRAFFIC RECORDS
# ============================================================

with st.expander(
    "🚦 View Traffic Records"
):

    try:

        records = get_traffic_records(
            scenario
        )

    except Exception:

        records = []


    if records:

        traffic_df = pd.DataFrame(
            records
        )

        st.dataframe(
            traffic_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No traffic records available."
        )


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="network-footer">

        <div>
            <strong>Q-ROUTE</strong>
        </div>

        <div>
            Intelligent Transportation Network
        </div>

        <div>
            Smarter Routes · Greener Cities
        </div>

    </div>
    """
)