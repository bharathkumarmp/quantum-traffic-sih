# ============================================================
# Q-ROUTE
# dashboard/pages/4_Results.py
# Optimization Results
# ============================================================

import sys
from pathlib import Path
import html as html_lib

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | Results",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "4_Results.css"

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


# ============================================================
# SERVICE IMPORT
# ============================================================

try:
    from optimizer_service import (
        get_traffic_summary,
        get_traffic_records,
        get_scenario_info,
    )
except Exception:
    get_traffic_summary = None
    get_traffic_records = None
    get_scenario_info = None


# ============================================================
# HELPERS
# ============================================================

def render_html(content):
    """Render HTML safely through Streamlit."""
    st.html(content)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_number(value, decimals=2):
    return f"{safe_float(value):,.{decimals}f}"


def format_runtime(seconds):
    seconds = safe_float(seconds)

    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"

    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes = int(seconds // 60)
    remaining = seconds % 60

    return f"{minutes}m {remaining:.1f}s"


def format_percent(value):
    value = safe_float(value)

    # Congestion in the project is stored as 0.0 - 1.0.
    if value <= 1:
        return f"{value * 100:.1f}%"

    return f"{value:.1f}%"


def normalize_blocked(value):
    """Handle True, False, 'true', '1', etc."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "blocked",
        }

    return False


def get_result():
    """
    Results are normally stored by 1_Optimizer.py as
    st.session_state.optimizer_result.
    """

    result = st.session_state.get("optimizer_result")

    if isinstance(result, dict):
        return result

    # Compatibility with alternate session-state naming.
    result = st.session_state.get("optimization_result")

    if isinstance(result, dict):
        return result

    return None


def get_algorithm(result):
    return (
        result.get("algorithm")
        or result.get("algorithm_name")
        or result.get("optimizer")
        or "Unknown"
    )


def get_scenario(result):
    return (
        result.get("scenario")
        or result.get("traffic_scenario")
        or "Normal"
    )


def get_nodes(result):
    nodes = result.get("graph_nodes", [])

    if isinstance(nodes, dict):
        return list(nodes.values())

    if isinstance(nodes, list):
        return nodes

    return []


def get_edges(result):
    edges = result.get("graph_edges", [])

    if isinstance(edges, list):
        return edges

    return []


def get_blocked_edges(result):
    return [
        edge
        for edge in get_edges(result)
        if normalize_blocked(edge.get("blocked", False))
    ]


def get_solution(result):
    solution = result.get("solution", {})

    if isinstance(solution, dict):
        return solution

    return {}


def get_vehicle_results(result):
    vehicle_results = result.get("vehicle_results", {})

    if isinstance(vehicle_results, dict):
        return vehicle_results

    if isinstance(vehicle_results, list):

        output = {}

        for index, item in enumerate(vehicle_results):

            if not isinstance(item, dict):
                continue

            vehicle_id = (
                item.get("vehicle_id")
                or item.get("id")
                or f"Vehicle {index + 1}"
            )

            output[str(vehicle_id)] = item

        return output

    return {}


def get_physical_route(vehicle_result):
    """
    vehicle_results contains the physical Node route generated
    by VRPFitnessFunction.

    This is preferred over solution[vehicle_id], because the
    solution route contains customer IDs.
    """

    if not isinstance(vehicle_result, dict):
        return []

    route = vehicle_result.get("route", [])

    if not isinstance(route, list):
        return []

    result = []

    for node in route:

        if isinstance(node, dict):
            node_id = (
                node.get("id")
                or node.get("node_id")
                or node.get("name")
            )

            if node_id is not None:
                result.append(str(node_id))

        else:
            node_id = getattr(node, "id", None)

            if node_id is not None:
                result.append(str(node_id))
            else:
                result.append(str(node))

    return result


def get_customer_route(value):
    """
    Convert a vehicle solution into customer IDs.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    return [str(value)]


def node_id_from_node(node):
    if isinstance(node, dict):
        return str(
            node.get("id")
            or node.get("node_id")
            or node.get("name")
        )

    return str(
        getattr(node, "id", node)
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">⚛</div>

            <div>
                <div class="sidebar-title">Q-ROUTE</div>
                <div class="sidebar-subtitle">
                    Intelligent Traffic Optimization
                </div>
            </div>
        </div>
        """
    )

    st.markdown("---")

    st.markdown("### Navigation")

    try:
        st.page_link(
            "app.py",
            label="🏠 Dashboard",
        )
    except Exception:
        pass

    try:
        st.page_link(
            "pages/1_Optimizer.py",
            label="⚙️ Optimizer",
        )
    except Exception:
        pass

    try:
        st.page_link(
            "pages/2_Traffic.py",
            label="🚦 Traffic",
        )
    except Exception:
        pass

    try:
        st.page_link(
            "pages/3_Analytics.py",
            label="📊 Analytics",
        )
    except Exception:
        pass

    try:
        st.page_link(
            "pages/4_Results.py",
            label="🏆 Results",
        )
    except Exception:
        pass

    try:
        st.page_link(
            "pages/5_Benchmark.py",
            label="🧪 Benchmark",
        )
    except Exception:
        pass

    try:
        st.page_link(
            "pages/6_About.py",
            label="ℹ️ About",
        )
    except Exception:
        pass

    st.markdown("---")

    st.markdown("### System")

    render_html(
        """
        <div class="sidebar-system">
            <span class="system-dot"></span>
            Optimization Engine Ready
        </div>
        """
    )


# ============================================================
# LOAD RESULT
# ============================================================

result = get_result()


# ============================================================
# EMPTY STATE
# ============================================================

if result is None:

    render_html(
        """
        <div class="empty-results">

            <div class="empty-results-icon">⚛️</div>

            <div class="results-eyebrow">
                Q-ROUTE
            </div>

            <h1>
                No Optimization
                <span>Results</span>
            </h1>

            <p>
                Run an optimization from the Optimizer page
                to view routes, network information,
                convergence and performance metrics.
            </p>

            <div class="empty-action">
                Open <strong>Optimizer</strong> and run
                your first optimization.
            </div>

        </div>
        """
    )

    st.stop()


# ============================================================
# EXTRACT RESULT DATA
# ============================================================

algorithm = get_algorithm(result)
scenario = get_scenario(result)

fitness = safe_float(result.get("fitness", 0))
travel_time = safe_float(result.get("travel_time", 0))
distance = safe_float(result.get("distance", 0))
congestion = safe_float(result.get("congestion", 0))
runtime = safe_float(
    result.get(
        "execution_time",
        result.get("runtime", 0),
    )
)

valid = result.get("valid", True)

graph_nodes = get_nodes(result)
graph_edges = get_edges(result)
blocked_edges = get_blocked_edges(result)

solution = get_solution(result)
vehicle_results = get_vehicle_results(result)

number_of_customers = (
    result.get("customers")
    or result.get("number_of_customers")
    or len(
        {
            customer
            for route in solution.values()
            for customer in get_customer_route(route)
        }
    )
)

number_of_vehicles = (
    result.get("vehicles")
    or result.get("number_of_vehicles")
    or len(solution)
)

number_of_clusters = (
    result.get("number_of_clusters")
    or result.get("clusters")
    or 0
)


# ============================================================
# HEADER
# ============================================================

render_html(
    f"""
    <div class="results-hero">

        <div class="results-eyebrow">
            QUANTUM-INSPIRED OPTIMIZATION
        </div>

        <h1>
            Optimization
            <span>Results</span>
        </h1>

        <p>
            Intelligent vehicle routing results for dynamic
            urban traffic
        </p>

        <div class="hero-tags">

            <span class="hero-tag">
                🧠 {html_lib.escape(str(algorithm))}
            </span>

            <span class="hero-tag">
                🚦 {html_lib.escape(str(scenario))}
            </span>

            <span class="hero-tag">
                🌐 {len(graph_nodes)} nodes
            </span>

        </div>

    </div>
    """
)


# ============================================================
# STATUS
# ============================================================

status_class = "valid-status" if valid else "invalid-status"

status_text = (
    "✓ VALID SOLUTION"
    if valid
    else "⚠ INVALID SOLUTION"
)

render_html(
    f"""
    <div class="{status_class}">
        <span class="status-icon">
            {"✓" if valid else "!"}
        </span>

        <div>
            <div class="status-title">
                {status_text}
            </div>

            <div class="status-description">
                Optimization completed using
                <strong>{html_lib.escape(str(algorithm))}</strong>
                under the
                <strong>{html_lib.escape(str(scenario))}</strong>
                scenario.
            </div>
        </div>
    </div>
    """
)


# ============================================================
# OPTIMIZATION METRICS
# ============================================================

render_html(
    """
    <div class="section-heading">
        <div class="section-icon">🏆</div>

        <div>
            <div class="section-title">
                Optimization Results
            </div>

            <div class="section-subtitle">
                Performance of the selected optimization run
            </div>
        </div>
    </div>
    """
)


render_html(
    f"""
    <div class="results-metrics-grid">

        <div class="result-metric-card metric-fitness">
            <div class="result-metric-icon">↗</div>
            <div class="result-metric-label">FITNESS</div>
            <div class="result-metric-value">
                {fitness:,.4f}
            </div>
            <div class="result-metric-note">
                Lower is better
            </div>
        </div>

        <div class="result-metric-card">
            <div class="result-metric-icon">◷</div>
            <div class="result-metric-label">
                TRAVEL TIME
            </div>
            <div class="result-metric-value">
                {travel_time:,.2f}
            </div>
            <div class="result-metric-note">
                Route time
            </div>
        </div>

        <div class="result-metric-card">
            <div class="result-metric-icon">📍</div>
            <div class="result-metric-label">
                DISTANCE
            </div>
            <div class="result-metric-value">
                {distance:,.2f}
            </div>
            <div class="result-metric-note">
                Total route distance
            </div>
        </div>

        <div class="result-metric-card">
            <div class="result-metric-icon">▥</div>
            <div class="result-metric-label">
                CONGESTION
            </div>
            <div class="result-metric-value">
                {format_percent(congestion)}
            </div>
            <div class="result-metric-note">
                Network route congestion
            </div>
        </div>

        <div class="result-metric-card">
            <div class="result-metric-icon">⏱</div>
            <div class="result-metric-label">
                RUNTIME
            </div>
            <div class="result-metric-value">
                {format_runtime(runtime)}
            </div>
            <div class="result-metric-note">
                Optimization execution
            </div>
        </div>

    </div>
    """
)


# ============================================================
# NETWORK OVERVIEW
# ============================================================

render_html(
    """
    <div class="section-heading">
        <div class="section-icon">🗄️</div>

        <div>
            <div class="section-title">
                Network Overview
            </div>

            <div class="section-subtitle">
                Transportation network used by the optimizer
            </div>
        </div>
    </div>
    """
)


render_html(
    f"""
    <div class="network-overview-card">

        <div class="network-stat">
            <div class="network-stat-icon">🛣</div>
            <div class="network-stat-value">
                {len(graph_nodes)}
            </div>
            <div class="network-stat-label">
                Nodes
            </div>
        </div>

        <div class="network-stat">
            <div class="network-stat-icon">↔</div>
            <div class="network-stat-value">
                {len(graph_edges)}
            </div>
            <div class="network-stat-label">
                Edges
            </div>
        </div>

        <div class="network-stat">
            <div class="network-stat-icon">👥</div>
            <div class="network-stat-value">
                {number_of_customers}
            </div>
            <div class="network-stat-label">
                Customers
            </div>
        </div>

        <div class="network-stat">
            <div class="network-stat-icon">🚚</div>
            <div class="network-stat-value">
                {number_of_vehicles}
            </div>
            <div class="network-stat-label">
                Vehicles
            </div>
        </div>

        <div class="network-stat">
            <div class="network-stat-icon">🧩</div>
            <div class="network-stat-value">
                {number_of_clusters}
            </div>
            <div class="network-stat-label">
                Clusters
            </div>
        </div>

    </div>
    """
)


# ============================================================
# TRAFFIC STATE
# ============================================================

traffic_summary = None

if get_traffic_summary is not None:

    try:
        traffic_summary = get_traffic_summary(scenario)
    except Exception:
        traffic_summary = None


# Use actual graph result as the authoritative blocked-edge count.
# This is important because the result contains the exact graph
# used for the optimization.

actual_blocked_count = len(blocked_edges)

if traffic_summary:

    summary_congestion = safe_float(
        traffic_summary.get(
            "average_congestion",
            congestion,
        )
    )

    summary_blocked = traffic_summary.get(
        "blocked_edges",
        actual_blocked_count,
    )

else:

    summary_congestion = congestion
    summary_blocked = actual_blocked_count


# ============================================================
# TRAFFIC OVERVIEW
# ============================================================

render_html(
    """
    <div class="section-heading traffic-heading">
        <div class="section-icon">🚦</div>

        <div>
            <div class="section-title">
                Traffic Overview
            </div>

            <div class="section-subtitle">
                Current traffic conditions affecting routing
            </div>
        </div>
    </div>
    """
)


traffic_col1, traffic_col2, traffic_col3, traffic_col4 = st.columns(4)


with traffic_col1:

    render_html(
        f"""
        <div class="traffic-card traffic-green">

            <div class="traffic-card-icon">
                🚗
            </div>

            <div class="traffic-card-label">
                CONGESTION
            </div>

            <div class="traffic-card-value">
                {format_percent(summary_congestion)}
            </div>

        </div>
        """
    )


with traffic_col2:

    blocked_class = (
        "traffic-red"
        if actual_blocked_count > 0
        else "traffic-green"
    )

    render_html(
        f"""
        <div class="traffic-card {blocked_class}">

            <div class="traffic-card-icon">
                ⚠
            </div>

            <div class="traffic-card-label">
                BLOCKED EDGES
            </div>

            <div class="traffic-card-value">
                {actual_blocked_count}
            </div>

        </div>
        """
    )


with traffic_col3:

    render_html(
        f"""
        <div class="traffic-card traffic-blue">

            <div class="traffic-card-icon">
                🛣
            </div>

            <div class="traffic-card-label">
                NODES
            </div>

            <div class="traffic-card-value">
                {len(graph_nodes)}
            </div>

        </div>
        """
    )


with traffic_col4:

    render_html(
        f"""
        <div class="traffic-card traffic-yellow">

            <div class="traffic-card-icon">
                👥
            </div>

            <div class="traffic-card-label">
                SCENARIO
            </div>

            <div class="traffic-card-value scenario-value">
                {html_lib.escape(str(scenario))}
            </div>

        </div>
        """
    )


# ============================================================
# BLOCKED ROAD ALERT
# ============================================================

if actual_blocked_count > 0:

    blocked_text = "<br>".join(
        [
            f"🚫 {edge.get('source', '?')} → {edge.get('target', '?')}"
            for edge in blocked_edges[:10]
        ]
    )

    if actual_blocked_count > 10:
        blocked_text += (
            f"<br>+ {actual_blocked_count - 10} more blocked roads"
        )

    render_html(
        f"""
        <div class="blocked-alert">

            <div class="blocked-alert-title">
                🚫 Blocked Roads Detected
            </div>

            <div class="blocked-alert-description">
                The optimizer was given a network containing
                <strong>{actual_blocked_count}</strong>
                blocked edge(s).
            </div>

            <div class="blocked-road-list">
                {blocked_text}
            </div>

        </div>
        """
    )


# ============================================================
# MAIN NETWORK VISUALIZATION
# ============================================================

render_html(
    """
    <div class="section-heading">
        <div class="section-icon">🗺️</div>

        <div>
            <div class="section-title">
                Transportation Network
            </div>

            <div class="section-subtitle">
                Optimized routes over the transportation graph
            </div>
        </div>
    </div>
    """
)


if graph_nodes and graph_edges:

    G = nx.DiGraph()

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    for node in graph_nodes:

        if isinstance(node, dict):

            node_id = (
                node.get("id")
                or node.get("node_id")
                or node.get("name")
            )

        else:

            node_id = str(node)

        if node_id is not None:
            G.add_node(str(node_id))


    # --------------------------------------------------------
    # Add edges
    # --------------------------------------------------------

    for edge in graph_edges:

        source = edge.get("source")
        target = edge.get("target")

        if source is None or target is None:
            continue

        source = str(source)
        target = str(target)

        G.add_edge(
            source,
            target,
            distance=safe_float(
                edge.get("distance", 0)
            ),
            travel_time=safe_float(
                edge.get("travel_time", 0)
            ),
            congestion=safe_float(
                edge.get("congestion", 0)
            ),
            blocked=normalize_blocked(
                edge.get("blocked", False)
            ),
        )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    if len(G.nodes) <= 10:

        # Controlled demo / small graph.
        pos = nx.spring_layout(
            G,
            seed=42,
            k=1.6,
            iterations=100,
        )

    else:

        pos = nx.spring_layout(
            G,
            seed=42,
            k=0.35,
            iterations=80,
        )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = go.Figure()


    # ========================================================
    # NORMAL EDGES
    # ========================================================

    normal_x = []
    normal_y = []

    normal_hover = []

    # ========================================================
    # BLOCKED EDGES
    # ========================================================

    blocked_x = []
    blocked_y = []

    blocked_hover = []


    for source, target, data in G.edges(data=True):

        if source not in pos or target not in pos:
            continue

        x0, y0 = pos[source]
        x1, y1 = pos[target]

        if data.get("blocked", False):

            blocked_x.extend(
                [x0, x1, None]
            )

            blocked_y.extend(
                [y0, y1, None]
            )

            blocked_hover.extend(
                [
                    (
                        f"🚫 BLOCKED ROAD<br>"
                        f"{source} → {target}<br>"
                        f"Distance: "
                        f"{data.get('distance', 0):.2f}<br>"
                        f"Congestion: "
                        f"{data.get('congestion', 0) * 100:.1f}%"
                    ),
                    "",
                    "",
                ]
            )

        else:

            normal_x.extend(
                [x0, x1, None]
            )

            normal_y.extend(
                [y0, y1, None]
            )

            normal_hover.extend(
                [
                    (
                        f"{source} → {target}<br>"
                        f"Distance: "
                        f"{data.get('distance', 0):.2f}<br>"
                        f"Travel time: "
                        f"{data.get('travel_time', 0):.2f}"
                    ),
                    "",
                    "",
                ]
            )


    # --------------------------------------------------------
    # Available road layer
    # --------------------------------------------------------

    if normal_x:

        fig.add_trace(
            go.Scatter(
                x=normal_x,
                y=normal_y,
                mode="lines",
                line=dict(
                    width=1.2,
                ),
                hoverinfo="text",
                hovertext=normal_hover,
                name="Available Roads",
            )
        )


    # --------------------------------------------------------
    # BLOCKED road layer
    # --------------------------------------------------------

    if blocked_x:

        fig.add_trace(
            go.Scatter(
                x=blocked_x,
                y=blocked_y,
                mode="lines",
                line=dict(
                    width=5,
                    dash="dash",
                ),
                hoverinfo="text",
                hovertext=blocked_hover,
                name="🚫 Blocked Roads",
            )
        )


    # ========================================================
    # OPTIMIZED ROUTES
    # ========================================================

    route_colors = [
        "#2563eb",
        "#7c3aed",
        "#0891b2",
        "#ea580c",
        "#16a34a",
        "#db2777",
        "#9333ea",
        "#0284c7",
        "#ca8a04",
        "#dc2626",
    ]

    plotted_routes = 0


    for vehicle_index, (vehicle_id, vehicle_result) in enumerate(
        vehicle_results.items()
    ):

        physical_route = get_physical_route(
            vehicle_result
        )

        if len(physical_route) < 2:
            continue

        route_x = []
        route_y = []

        valid_route = True

        for node_id in physical_route:

            if node_id not in pos:
                valid_route = False
                break

            x, y = pos[node_id]

            route_x.append(x)
            route_y.append(y)

        if not valid_route:
            continue

        route_color = route_colors[
            vehicle_index % len(route_colors)
        ]

        fig.add_trace(
            go.Scatter(
                x=route_x,
                y=route_y,
                mode="lines+markers",
                line=dict(
                    width=4,
                    color=route_color,
                ),
                marker=dict(
                    size=6,
                    color=route_color,
                ),
                name=f"Vehicle {vehicle_id}",
                hovertemplate=(
                    f"<b>Vehicle {vehicle_id}</b>"
                    "<br>Optimized route"
                    "<extra></extra>"
                ),
            )
        )

        plotted_routes += 1


    # ========================================================
    # NETWORK NODES
    # ========================================================

    node_x = []
    node_y = []
    node_text = []

    for node_id in G.nodes():

        if node_id not in pos:
            continue

        x, y = pos[node_id]

        node_x.append(x)
        node_y.append(y)
        node_text.append(node_id)


    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text if len(G.nodes) <= 30 else None,
            textposition="top center",
            marker=dict(
                size=9 if len(G.nodes) > 30 else 12,
            ),
            name="Network Nodes",
            hovertemplate=(
                "<b>%{text}</b>"
                "<extra></extra>"
            ),
        )
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(
        height=620,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        hovermode="closest",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
        },
    )


    # --------------------------------------------------------
    # Legend explanation
    # --------------------------------------------------------

    legend_col1, legend_col2, legend_col3 = st.columns(3)

    with legend_col1:
        render_html(
            """
            <div class="map-legend-item">
                <span class="legend-line"></span>
                Available road
            </div>
            """
        )

    with legend_col2:
        render_html(
            """
            <div class="map-legend-item">
                <span class="legend-blocked"></span>
                🚫 Blocked road
            </div>
            """
        )

    with legend_col3:
        render_html(
            f"""
            <div class="map-legend-item">
                <span class="legend-route"></span>
                Optimized route
                ({plotted_routes} vehicles)
            </div>
            """
        )

else:

    render_html(
        """
        <div class="info-card">
            <strong>Network data unavailable</strong>
            <br>
            This optimization result does not contain
            serialized graph nodes and edges.
        </div>
        """
    )


# ============================================================
# BLOCKED EDGE TABLE
# ============================================================

if blocked_edges:

    render_html(
        """
        <div class="section-heading">
            <div class="section-icon">🚧</div>

            <div>
                <div class="section-title">
                    Blocked Road Details
                </div>

                <div class="section-subtitle">
                    Roads unavailable to the optimizer
                </div>
            </div>
        </div>
        """
    )

    blocked_rows = []

    for edge in blocked_edges:

        blocked_rows.append(
            {
                "Source": str(
                    edge.get("source", "")
                ),
                "Destination": str(
                    edge.get("target", "")
                ),
                "Status": "🚫 BLOCKED",
                "Distance": safe_float(
                    edge.get("distance", 0)
                ),
                "Travel Time": safe_float(
                    edge.get("travel_time", 0)
                ),
                "Congestion": format_percent(
                    edge.get("congestion", 0)
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(blocked_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# OPTIMIZED VEHICLE ROUTES
# ============================================================

render_html(
    """
    <div class="section-heading">
        <div class="section-icon">🚚</div>

        <div>
            <div class="section-title">
                Optimized Vehicle Routes
            </div>

            <div class="section-subtitle">
                Customer assignments and physical network routes
            </div>
        </div>
    </div>
    """
)


if solution:

    route_rows = []

    for vehicle_id, customer_route in solution.items():

        vehicle_id = str(vehicle_id)

        customer_ids = get_customer_route(
            customer_route
        )

        vehicle_result = vehicle_results.get(
            vehicle_id,
            {},
        )

        physical_route = get_physical_route(
            vehicle_result
        )

        route_rows.append(
            {
                "Vehicle": vehicle_id,
                "Customers": (
                    " → ".join(customer_ids)
                    if customer_ids
                    else "No customers"
                ),
                "Network Route": (
                    " → ".join(physical_route)
                    if physical_route
                    else "Not available"
                ),
                "Travel Time": format_number(
                    vehicle_result.get(
                        "travel_time",
                        0,
                    ),
                    2,
                ),
                "Distance": format_number(
                    vehicle_result.get(
                        "distance",
                        0,
                    ),
                    2,
                ),
                "Congestion": format_percent(
                    vehicle_result.get(
                        "congestion",
                        0,
                    )
                ),
            }
        )


    if route_rows:

        st.dataframe(
            pd.DataFrame(route_rows),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No vehicle routes are available."
        )

else:

    st.info(
        "No optimized solution is available."
    )


# ============================================================
# CONVERGENCE
# ============================================================

history = result.get("history", [])

if isinstance(history, list) and len(history) > 1:

    render_html(
        """
        <div class="section-heading">
            <div class="section-icon">📈</div>

            <div>
                <div class="section-title">
                    Convergence Analysis
                </div>

                <div class="section-subtitle">
                    Fitness progression during optimization
                </div>
            </div>
        </div>
        """
    )

    numeric_history = []

    for value in history:

        try:
            numeric_history.append(
                float(value)
            )

        except (TypeError, ValueError):
            continue


    if len(numeric_history) > 1:

        iterations_axis = list(
            range(len(numeric_history))
        )

        fig_conv = go.Figure()

        fig_conv.add_trace(
            go.Scatter(
                x=iterations_axis,
                y=numeric_history,
                mode="lines+markers",
                line=dict(
                    width=3,
                ),
                marker=dict(
                    size=5,
                ),
                name=str(algorithm),
            )
        )

        fig_conv.update_layout(
            height=390,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
            paper_bgcolor="white",
            plot_bgcolor="#f8fafc",
            xaxis_title="Iteration",
            yaxis_title="Fitness",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_conv,
            use_container_width=True,
        )


        initial_fitness = numeric_history[0]
        final_recorded_fitness = numeric_history[-1]

        if initial_fitness != 0:

            improvement = (
                (initial_fitness - final_recorded_fitness)
                / abs(initial_fitness)
            ) * 100

        else:

            improvement = 0


        c1, c2, c3 = st.columns(3)

        with c1:

            render_html(
                f"""
                <div class="small-stat">
                    <div>INITIAL FITNESS</div>
                    <strong>
                        {initial_fitness:,.4f}
                    </strong>
                </div>
                """
            )

        with c2:

            render_html(
                f"""
                <div class="small-stat">
                    <div>FINAL RECORDED</div>
                    <strong>
                        {final_recorded_fitness:,.4f}
                    </strong>
                </div>
                """
            )

        with c3:

            render_html(
                f"""
                <div class="small-stat">
                    <div>IMPROVEMENT</div>
                    <strong>
                        {improvement:.2f}%
                    </strong>
                </div>
                """
            )


# ============================================================
# DECOMPOSITION INFORMATION
# ============================================================

is_decomposed = (
    "decomposed"
    in str(algorithm).lower()
)

if is_decomposed:

    render_html(
        """
        <div class="section-heading">
            <div class="section-icon">🧩</div>

            <div>
                <div class="section-title">
                    Decomposition
                </div>

                <div class="section-subtitle">
                    Large VRP divided into smaller optimization
                    subproblems
                </div>
            </div>
        </div>
        """
    )


    repair_enabled = result.get(
        "repair_enabled",
        result.get(
            "repair",
            False,
        ),
    )

    repair_improvement = safe_float(
        result.get(
            "repair_improvement",
            0,
        )
    )

    repair_operations = result.get(
        "repair_operations",
        result.get(
            "repair_metrics",
            {},
        ),
    )

    if isinstance(repair_operations, dict):

        operations = repair_operations.get(
            "operations",
            result.get(
                "repair_operations",
                0,
            ),
        )

    else:

        operations = repair_operations


    render_html(
        f"""
        <div class="decomposition-card">

            <div class="decomp-stat">
                <span>Clusters</span>
                <strong>
                    {number_of_clusters}
                </strong>
            </div>

            <div class="decomp-stat">
                <span>Warm Start</span>
                <strong>
                    {"Enabled" if "warm" in str(algorithm).lower() else "N/A"}
                </strong>
            </div>

            <div class="decomp-stat">
                <span>Global Repair</span>
                <strong>
                    {"Enabled" if repair_enabled else "Disabled"}
                </strong>
            </div>

            <div class="decomp-stat">
                <span>Repair Improvement</span>
                <strong>
                    {repair_improvement:,.4f}
                </strong>
            </div>

        </div>
        """
    )


# ============================================================
# TRAFFIC DATA
# ============================================================

if get_traffic_records is not None:

    with st.expander("🚦 View Traffic Data"):

        try:

            records = get_traffic_records(
                scenario
            )

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

        except Exception as error:

            st.warning(
                f"Traffic data unavailable: {error}"
            )


# ============================================================
# SCENARIO INFORMATION
# ============================================================

if get_scenario_info is not None:

    with st.expander("ℹ️ Scenario Information"):

        try:

            info = get_scenario_info(
                scenario
            )

            if isinstance(info, dict):

                info_rows = []

                for key, value in info.items():

                    info_rows.append(
                        {
                            "Property": key.replace(
                                "_",
                                " ",
                            ).title(),
                            "Value": value,
                        }
                    )

                st.dataframe(
                    pd.DataFrame(info_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.write(info)

        except Exception as error:

            st.warning(
                f"Scenario information unavailable: {error}"
            )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="results-footer">

        <div class="footer-brand">
            ⚛️ Q-ROUTE
        </div>

        <div class="footer-text">
            Quantum-Inspired Intelligent Traffic
            Route Optimization
        </div>

        <div class="footer-status">
            ● OPTIMIZATION COMPLETE
        </div>

    </div>
    """
)