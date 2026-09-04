# ============================================================
# Q-ROUTE | OPTIMIZER PAGE
# dashboard/pages/1_Optimizer.py
# ============================================================

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SERVICE
# ============================================================

from optimizer_service import (
    run_optimizer,
    run_algorithm_comparison,
    get_traffic_summary,
    get_traffic_records,
    get_scenario_info,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | Optimizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "1_Optimizer.css"

if CSS_PATH.exists():
    st.html(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"
    )


# ============================================================
# HELPERS
# ============================================================

def html(content):
    st.html(content)


def number(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "0"


def percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def runtime(value):
    try:
        value = float(value)

        if value < 1:
            return f"{value * 1000:.0f} ms"

        if value < 60:
            return f"{value:.2f} s"

        minutes = int(value // 60)
        seconds = value % 60

        return f"{minutes}m {seconds:.1f}s"

    except Exception:
        return "—"


def safe_float(value, default=0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# SESSION STATE
# ============================================================

if "optimizer_result" not in st.session_state:
    st.session_state.optimizer_result = None

if "optimizer_config" not in st.session_state:
    st.session_state.optimizer_config = None

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    html(
        """
        <div class="brand-row">

            <div class="brand-icon">
                ⚛
            </div>

            <div>
                <div class="brand-name">
                    Q-ROUTE
                </div>

                <div class="brand-subtitle">
                    Intelligent Traffic Optimization
                </div>
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    html(
        """
        <div class="side-section-title">
            NAVIGATION
        </div>
        """
    )

    st.page_link(
        "pages/1_Optimizer.py",
        label="Optimizer",
        icon="🏠",
    )

    st.page_link(
        "pages/2_Traffic.py",
        label="Traffic Scenario",
        icon="🚦",
    )

    st.page_link(
        "pages/3_Network.py",
        label="Network Configuration",
        icon="🔗",
    )

    st.page_link(
        "pages/4_Results.py",
        label="Results & Analytics",
        icon="📊",
    )

    st.page_link(
        "pages/5_Benchmark.py",
        label="Benchmark",
        icon="⚡",
    )

    st.page_link(
        "pages/6_About.py",
        label="About",
        icon="ℹ️",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # QUOTE
    # --------------------------------------------------------

    html(
        """
        <div class="sidebar-quote">

            <div class="quote-icon">
                🌱
            </div>

            <div>
                <strong>
                    "Smarter Routes
                    <br>
                    Greener Cities"
                </strong>

                <small>
                    Q-ROUTE
                </small>
            </div>

        </div>
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    html(
        """
        <div class="location-info">

            <div>
                <span>📍</span>
                <b>Bengaluru, India</b>
            </div>

            <div>
                <span>📅</span>
                Aug 30, 2026
            </div>

            <div>
                <span>◷</span>
                09:42 AM
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # SIDEBAR TRAFFIC LIGHT
    # --------------------------------------------------------

    html(
        """
        <div class="sidebar-signal-wrap">

            <div class="traffic-signal small-signal">

                <div class="signal-light red-light"></div>
                <div class="signal-light yellow-light"></div>
                <div class="signal-light green-light"></div>

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # SIDEBAR FOOTER
    # --------------------------------------------------------

    html(
        """
        <div class="sidebar-bottom-message">

            STOP CONGESTION
            <br>
            START A BETTER TOMORROW

        </div>
        """
    )


# ============================================================
# GET TRAFFIC INFORMATION
# ============================================================

scenario_options = [
    "Normal",
    "Morning Peak",
    "Heavy Congestion",
    "Evening Peak",
    "Accident",
    "Road Closure",
]


# ============================================================
# MAIN TWO ZONES
# ============================================================

left_zone, right_zone = st.columns(
    [1.03, 0.97],
    gap="small",
)


# ============================================================
# YELLOW ZONE
# ============================================================

with left_zone:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    html(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                QUANTUM-INSPIRED OPTIMIZATION
            </div>

            <h1>
                Optimization
                <span>Control</span>
            </h1>

            <p>
                Intelligent vehicle routing for dynamic urban traffic
            </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # NETWORK SCALE
    # --------------------------------------------------------

    html(
        """
        <div class="section-card">

            <div class="card-title">
                <span class="card-icon blue">
                    🌐
                </span>

                Network Scale
            </div>

        </div>
        """
    )

    network_col1, network_col2 = st.columns(
        [1, 1],
        gap="medium",
    )

    with network_col1:

        network_size = st.selectbox(
            "Network Size",
            [6, 50, 100, 250, 500],
            index=2,
            format_func=lambda x: f"{x} nodes",
            key="network_size",
        )

    with network_col2:

        edges_per_node = st.slider(
            "Edges per Node",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            key="edges_per_node",
        )

    # --------------------------------------------------------
    # FLEET
    # --------------------------------------------------------

    html(
        """
        <div class="section-card fleet-card">

            <div class="card-title">
                <span class="card-icon red">
                    🚚
                </span>

                Fleet Configuration
            </div>

        </div>
        """
    )

    fleet_col1, fleet_col2 = st.columns(
        [1, 1],
        gap="medium",
    )

    with fleet_col1:

        customers = st.slider(
            "Customers",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            key="customers",
        )

    with fleet_col2:

        vehicles = st.slider(
            "Vehicles",
            min_value=2,
            max_value=20,
            value=10,
            step=1,
            key="vehicles",
        )

    # --------------------------------------------------------
    # ALGORITHM
    # --------------------------------------------------------

    html(
        """
        <div class="section-card">

            <div class="card-title">
                <span class="card-icon navy">
                    ⚙
                </span>

                Algorithm Settings
            </div>

        </div>
        """
    )

    alg_col1, alg_col2, alg_col3 = st.columns(
        [1.25, 0.75, 0.75],
        gap="small",
    )

    with alg_col1:

        algorithm = st.selectbox(
            "Algorithm",
            [
                "PSO",
                "QPSO",
                "Warm-QPSO",
                "Decomposed Warm-QPSO",
            ],
            index=3,
            key="algorithm",
        )

    with alg_col2:

        particles = st.slider(
            "Particles",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="particles",
        )

    with alg_col3:

        iterations = st.slider(
            "Iterations",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            key="iterations",
        )

    # --------------------------------------------------------
    # DECOMPOSITION
    # --------------------------------------------------------

    html(
        """
        <div class="section-card decomposition-card">

            <div class="card-title">
                <span class="card-icon purple">
                    🧩
                </span>

                Decomposition
            </div>

        </div>
        """
    )

    max_clusters = min(
        10,
        customers,
        vehicles,
    )

    if max_clusters >= 2:

        clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=max_clusters,
            value=max_clusters,
            step=1,
            key="clusters",
        )

    else:

        clusters = 1

        st.caption(
            "At least 2 customers and 2 vehicles "
            "are recommended."
        )

    effective_clusters = min(
        clusters,
        customers,
        vehicles,
    )

    st.caption(
        f"Selected: {clusters} clusters · "
        f"Effective: {effective_clusters} clusters"
    )

    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

    button_col1, button_col2 = st.columns(
        [1, 1],
        gap="small",
    )

    with button_col1:

        run_button = st.button(
            "▶ RUN OPTIMIZATION",
            type="primary",
            use_container_width=True,
            key="run_optimizer",
        )

    with button_col2:

        comparison_button = st.button(
            "▥ COMPARE ALGORITHMS",
            use_container_width=True,
            key="compare_algorithms",
        )

    # --------------------------------------------------------
    # LARGE NETWORK WARNING
    # --------------------------------------------------------

    if network_size >= 500:

        html(
            """
            <div class="warning-card">
                ◷ &nbsp;
                <b>500-node stress test</b>
                may take several minutes.
            </div>
            """
        )

    elif network_size >= 250:

        html(
            """
            <div class="warning-card">
                ◷ &nbsp;
                Large network selected.
                Optimization may take longer.
            </div>
            """
        )


# ============================================================
# GREEN ZONE
# ============================================================

with right_zone:

    # --------------------------------------------------------
    # TOP TRAFFIC ILLUSTRATION
    # --------------------------------------------------------

    html(
        """
        <div class="traffic-hero">

            <div class="ready-badge">
                <span></span>
                SYSTEM READY
            </div>

            <div class="traffic-message">
                <b>
                    SMART<br>
                    TRAFFIC<br>
                    BRIGHTER<br>
                    TOMORROW
                </b>
            </div>

            <div class="city-silhouette">

                <div class="building b1"></div>
                <div class="building b2"></div>
                <div class="building b3"></div>
                <div class="building b4"></div>
                <div class="building b5"></div>

            </div>

            <div class="hero-road"></div>

            <div class="traffic-signal big-signal">

                <div class="signal-light red-light"></div>
                <div class="signal-light yellow-light"></div>
                <div class="signal-light green-light"></div>

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # TRAFFIC SUMMARY
    # --------------------------------------------------------

    try:

        traffic_summary = get_traffic_summary(
            scenario
        )

    except Exception:

        traffic_summary = None

    if traffic_summary:

        average_congestion = safe_float(
            traffic_summary.get(
                "average_congestion",
                0,
            )
        )

        blocked_edges = traffic_summary.get(
            "blocked_edges",
            0,
        )

    else:

        average_congestion = 0
        blocked_edges = 0


    # --------------------------------------------------------
    # TRAFFIC SCENARIO
    # --------------------------------------------------------

    html(
        """
        <div class="green-card">

            <div class="traffic-title-row">

                <div class="traffic-icon">
                    🚦
                </div>

                <div>

                    <h2>
                        Traffic Scenario
                    </h2>

                    <p>
                        Traffic conditions and road status
                    </p>

                </div>

            </div>

        </div>
        """
    )

    scenario = st.selectbox(
        "Traffic Scenario",
        scenario_options,
        index=0,
        key="scenario",
    )


    # --------------------------------------------------------
    # TRAFFIC KPI CARDS
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small",
    )

    with k1:

        html(
            f"""
            <div class="traffic-kpi green-kpi">

                <div class="kpi-icon">
                    🚗
                </div>

                <div class="kpi-number">
                    {average_congestion * 100:.1f}%
                </div>

                <div class="kpi-label">
                    Congestion
                </div>

            </div>
            """
        )

    with k2:

        html(
            f"""
            <div class="traffic-kpi red-kpi">

                <div class="kpi-icon">
                    ⚠
                </div>

                <div class="kpi-number">
                    {blocked_edges}
                </div>

                <div class="kpi-label">
                    Blocked Edges
                </div>

            </div>
            """
        )

    with k3:

        html(
            f"""
            <div class="traffic-kpi blue-kpi">

                <div class="kpi-icon">
                    🛣
                </div>

                <div class="kpi-number">
                    {network_size}
                </div>

                <div class="kpi-label">
                    Nodes
                </div>

            </div>
            """
        )

    with k4:

        html(
            f"""
            <div class="traffic-kpi purple-kpi">

                <div class="kpi-icon">
                    👥
                </div>

                <div class="kpi-number">
                    {scenario}
                </div>

                <div class="kpi-label">
                    Scenario
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # OPTIMIZATION RESULTS
    # --------------------------------------------------------

    result = st.session_state.optimizer_result

    if result:

        result_fitness = result.get(
            "fitness",
            0,
        )

        result_time = result.get(
            "travel_time",
            0,
        )

        result_distance = result.get(
            "distance",
            0,
        )

        result_congestion = result.get(
            "congestion",
            0,
        )

        result_runtime = result.get(
            "execution_time",
            0,
        )

        result_valid = result.get(
            "valid",
            True,
        )

    else:

        result_fitness = 0
        result_time = 0
        result_distance = 0
        result_congestion = 0
        result_runtime = 0
        result_valid = False


    html(
        f"""
        <div class="results-card">

            <div class="results-header">

                <div class="results-title">
                    <span>
                        🏆
                    </span>

                    Optimization Results
                </div>

                <div class="valid-solution
                    {'valid' if result_valid else 'waiting'}">

                    <span>✓</span>

                    {
                        "VALID SOLUTION"
                        if result_valid
                        else "WAITING"
                    }

                </div>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------

    r1, r2, r3 = st.columns(
        3,
        gap="small",
    )

    with r1:

        html(
            f"""
            <div class="result-kpi pink-result">

                <div class="result-icon">
                    ↗
                </div>

                <div class="result-number">
                    {
                        number(result_fitness, 4)
                        if result
                        else "—"
                    }
                </div>

                <div class="result-label">
                    Fitness
                </div>

            </div>
            """
        )

    with r2:

        html(
            f"""
            <div class="result-kpi yellow-result">

                <div class="result-icon">
                    ◷
                </div>

                <div class="result-number">
                    {
                        number(result_time, 2)
                        if result
                        else "—"
                    }
                </div>

                <div class="result-label">
                    Travel Time
                </div>

            </div>
            """
        )

    with r3:

        html(
            f"""
            <div class="result-kpi green-result">

                <div class="result-icon">
                    📍
                </div>

                <div class="result-number">
                    {
                        number(result_distance, 2)
                        if result
                        else "—"
                    }
                </div>

                <div class="result-label">
                    Distance
                </div>

            </div>
            """
        )


    r4, r5 = st.columns(
        2,
        gap="small",
    )

    with r4:

        html(
            f"""
            <div class="result-kpi orange-result">

                <div class="result-icon">
                    ▥
                </div>

                <div class="result-number">
                    {
                        percent(result_congestion)
                        if result
                        else "—"
                    }
                </div>

                <div class="result-label">
                    Congestion
                </div>

            </div>
            """
        )

    with r5:

        html(
            f"""
            <div class="result-kpi cyan-result">

                <div class="result-icon">
                    ⏱
                </div>

                <div class="result-number">
                    {
                        runtime(result_runtime)
                        if result
                        else "—"
                    }
                </div>

                <div class="result-label">
                    Runtime
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # NETWORK OVERVIEW
    # --------------------------------------------------------

    html(
        """
        <div class="green-card overview-card">

            <div class="overview-title">
                🗄 &nbsp; Network Overview
            </div>

        </div>
        """
    )

    graph_nodes = (
        result.get("graph_nodes", [])
        if result
        else []
    )

    graph_edges = (
        result.get("graph_edges", [])
        if result
        else []
    )

    overview_values = [
        (
            len(graph_nodes)
            if graph_nodes
            else network_size,
            "Nodes",
        ),
        (
            len(graph_edges)
            if graph_edges
            else "—",
            "Edges",
        ),
        (
            result.get(
                "customers",
                customers,
            )
            if result
            else customers,
            "Customers",
        ),
        (
            result.get(
                "vehicles",
                vehicles,
            )
            if result
            else vehicles,
            "Vehicles",
        ),
        (
            result.get(
                "number_of_clusters",
                effective_clusters,
            )
            if result
            else effective_clusters,
            "Clusters",
        ),
    ]

    overview_cols = st.columns(
        5,
        gap="small",
    )

    for index, (value, label) in enumerate(
        overview_values
    ):

        with overview_cols[index]:

            html(
                f"""
                <div class="overview-kpi">

                    <div class="overview-number">
                        {value}
                    </div>

                    <div class="overview-label">
                        {label}
                    </div>

                </div>
                """
            )


    # --------------------------------------------------------
    # OPEN RESULTS
    # --------------------------------------------------------

    if result:

        if st.button(
            "📊 OPEN FULL RESULTS",
            use_container_width=True,
            key="open_results",
        ):

            st.switch_page(
                "pages/4_Results.py"
            )


    # --------------------------------------------------------
    # GREEN FOOTER
    # --------------------------------------------------------

    html(
        """
        <div class="green-footer">

            <b>
                Q-ROUTE
            </b>

            <span>|</span>

            <span>
                Smarter Routes
            </span>

            <span>
                •
            </span>

            <span>
                Greener Cities
            </span>

            <span>
                •
            </span>

            <span>
                Better Tomorrow
            </span>

            <span class="leaf">
                🌱
            </span>

        </div>
        """
    )


# ============================================================
# RUN OPTIMIZATION
# ============================================================

if run_button:

    config = {
        "scenario": scenario,
        "algorithm": algorithm,
        "customers": customers,
        "vehicles": vehicles,
        "network_size": network_size,
        "edges_per_node": edges_per_node,
        "particles": particles,
        "iterations": iterations,
        "clusters": effective_clusters,
    }

    with st.spinner(
        f"Running {algorithm} on "
        f"{network_size} nodes..."
    ):

        try:

            result = run_optimizer(
                scenario=scenario,
                algorithm_name=algorithm,
                customers=customers,
                vehicles=vehicles,
                network_size=network_size,
                edges_per_node=edges_per_node,
                particles=particles,
                iterations=iterations,
                clusters=effective_clusters,
            )

            st.session_state.optimizer_result = result

            st.session_state.optimizer_config = config

            st.session_state.comparison_result = None

            st.switch_page(
                "pages/4_Results.py"
            )

        except Exception as error:

            st.error(
                f"Optimization failed: {error}"
            )

            st.exception(error)


# ============================================================
# COMPARE ALGORITHMS
# ============================================================

if comparison_button:

    with st.spinner(
        "Comparing optimization algorithms..."
    ):

        try:

            comparison = run_algorithm_comparison(
                scenario=scenario,
                customers=customers,
                vehicles=vehicles,
                network_size=network_size,
                edges_per_node=edges_per_node,
                particles=particles,
                iterations=iterations,
                clusters=effective_clusters,
            )

            st.session_state.comparison_result = comparison

            st.success(
                "Algorithm comparison completed."
            )

        except Exception as error:

            st.error(
                f"Comparison failed: {error}"
            )

            st.exception(error)


# ============================================================
# COMPARISON RESULTS
# ============================================================

comparison = st.session_state.comparison_result

if comparison:

    st.markdown("---")

    st.markdown(
        "## ⚔️ Algorithm Comparison"
    )

    rows = []

    if isinstance(
        comparison,
        dict,
    ):

        for name, data in comparison.items():

            if not isinstance(
                data,
                dict,
            ):
                continue

            if "fitness" not in data:
                continue

            rows.append(
                {
                    "Algorithm": name,
                    "Fitness": data.get(
                        "fitness",
                        0,
                    ),
                    "Travel Time": data.get(
                        "travel_time",
                        0,
                    ),
                    "Distance": data.get(
                        "distance",
                        0,
                    ),
                    "Congestion": data.get(
                        "congestion",
                        0,
                    ),
                    "Runtime": data.get(
                        "execution_time",
                        0,
                    ),
                    "Valid": data.get(
                        "valid",
                        True,
                    ),
                }
            )

        if not rows:

            data_list = comparison.get(
                "results",
                [],
            )

            if isinstance(
                data_list,
                list,
            ):

                for data in data_list:

                    if not isinstance(
                        data,
                        dict,
                    ):
                        continue

                    rows.append(
                        {
                            "Algorithm": data.get(
                                "algorithm",
                                "Unknown",
                            ),
                            "Fitness": data.get(
                                "fitness",
                                0,
                            ),
                            "Travel Time": data.get(
                                "travel_time",
                                0,
                            ),
                            "Distance": data.get(
                                "distance",
                                0,
                            ),
                            "Congestion": data.get(
                                "congestion",
                                0,
                            ),
                            "Runtime": data.get(
                                "execution_time",
                                0,
                            ),
                            "Valid": data.get(
                                "valid",
                                True,
                            ),
                        }
                    )

    elif isinstance(
        comparison,
        list,
    ):

        for data in comparison:

            if not isinstance(
                data,
                dict,
            ):
                continue

            rows.append(
                {
                    "Algorithm": data.get(
                        "algorithm",
                        "Unknown",
                    ),
                    "Fitness": data.get(
                        "fitness",
                        0,
                    ),
                    "Travel Time": data.get(
                        "travel_time",
                        0,
                    ),
                    "Distance": data.get(
                        "distance",
                        0,
                    ),
                    "Congestion": data.get(
                        "congestion",
                        0,
                    ),
                    "Runtime": data.get(
                        "execution_time",
                        0,
                    ),
                    "Valid": data.get(
                        "valid",
                        True,
                    ),
                }
            )

    if rows:

        comparison_df = pd.DataFrame(rows)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # FITNESS
        # ----------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=comparison_df["Algorithm"],
                y=comparison_df["Fitness"],
                text=[
                    f"{safe_float(x):.2f}"
                    for x in comparison_df["Fitness"]
                ],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="Fitness Comparison",
            height=400,
            xaxis_title="Algorithm",
            yaxis_title="Fitness",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        runtime_fig = go.Figure()

        runtime_fig.add_trace(
            go.Bar(
                x=comparison_df["Algorithm"],
                y=comparison_df["Runtime"],
                text=[
                    runtime(x)
                    for x in comparison_df["Runtime"]
                ],
                textposition="auto",
            )
        )

        runtime_fig.update_layout(
            title="Execution Time Comparison",
            height=400,
            xaxis_title="Algorithm",
            yaxis_title="Seconds",
        )

        st.plotly_chart(
            runtime_fig,
            use_container_width=True,
        )


# ============================================================
# SIMULATED TRAFFIC DATA
# ============================================================

with st.expander(
    "🚦 View Simulated Traffic Data"
):

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

with st.expander(
    "ℹ️ Scenario Information"
):

    try:

        info = get_scenario_info(
            scenario
        )

        if isinstance(
            info,
            dict,
        ):

            for key, value in info.items():

                st.markdown(
                    f"**{key.replace('_', ' ').title()}:** "
                    f"{value}"
                )

        else:

            st.write(info)

    except Exception as error:

        st.warning(
            f"Scenario information unavailable: {error}"
        )