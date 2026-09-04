# ============================================================
# Q-ROUTE | BENCHMARK LAB
# dashboard/pages/5_Benchmark.py
# ============================================================

import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SERVICE IMPORT
# ============================================================

from optimizer_service import (
    run_algorithm_comparison,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | Benchmark Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "5_Benchmark.css"

if CSS_PATH.exists():

    st.html(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"
    )


# ============================================================
# HELPERS
# ============================================================

def html(content):
    st.html(content)


def safe_float(value, default=0.0):

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return float(default)


def safe_int(value, default=0):

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return int(default)


def format_number(
    value,
    decimals=2,
):

    try:
        return f"{float(value):,.{decimals}f}"

    except (
        TypeError,
        ValueError,
    ):
        return "—"


def format_seconds(value):

    value = safe_float(value)

    if value < 1:
        return f"{value * 1000:.1f} ms"

    if value < 60:
        return f"{value:.3f} s"

    minutes = int(value // 60)
    seconds = value % 60

    return f"{minutes}m {seconds:.1f}s"


def extract_history(result):

    if not isinstance(
        result,
        dict,
    ):
        return []

    history = result.get(
        "history",
        [],
    )

    if not isinstance(
        history,
        list,
    ):
        return []

    values = []

    for item in history:

        if isinstance(
            item,
            dict,
        ):

            item = item.get(
                "fitness",
                item.get(
                    "best_fitness",
                    None,
                ),
            )

        try:

            values.append(
                float(item)
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

    return values


def get_result_fitness(result):

    return safe_float(
        result.get(
            "fitness",
            float("inf"),
        ),
        float("inf"),
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                ⚛
            </div>

            <div>

                <div class="sidebar-title">
                    Q-ROUTE
                </div>

                <div class="sidebar-subtitle">
                    Intelligent Traffic Optimization
                </div>

            </div>

        </div>
        """
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="side-section-title">
            NAVIGATION
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.page_link(
        "pages/1_Optimizer.py",
        label="Optimizer",
        icon="⚛️",
    )

    st.page_link(
        "pages/2_Traffic.py",
        label="Traffic Scenario",
        icon="🚦",
    )

    st.page_link(
        "pages/3_Network.py",
        label="Network Configuration",
        icon="🌐",
    )

    st.page_link(
        "pages/4_Results.py",
        label="Results & Analytics",
        icon="📊",
    )

    st.page_link(
        "pages/5_Benchmark.py",
        label="Benchmark Lab",
        icon="⚡",
    )

    st.page_link(
        "pages/6_About.py",
        label="About",
        icon="ℹ️",
    )

    st.markdown("---")

    html(
        """
        <div class="sidebar-message">

            <div class="sidebar-message-icon">
                ⚡
            </div>

            <div>

                <strong>
                    PERFORMANCE LAB
                </strong>

                <br>

                <span>
                    Compare Optimization Methods
                </span>

            </div>

        </div>
        """
    )


# ============================================================
# HERO
# ============================================================

html(
    """
    <div class="benchmark-hero">

        <div class="benchmark-eyebrow">
            PERFORMANCE LAB
        </div>

        <div class="benchmark-icon">
            ⚡
        </div>

        <div class="benchmark-main-title">
            ALGORITHM PERFORMANCE ANALYSIS
        </div>

        <h1>
            Benchmark
            <span>Lab</span>
        </h1>

        <p>
            Compare classical and quantum-inspired
            optimization methods across solution quality,
            travel cost, congestion and execution time.
        </p>

    </div>
    """
)


# ============================================================
# CONFIGURATION
# ============================================================

st.markdown(
    "## ⚙️ Benchmark Configuration"
)


col1, col2, col3 = st.columns(3)


with col1:

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


with col2:

    network_size = st.selectbox(
        "Network Size",
        [
            6,
            50,
            100,
            250,
            500,
        ],
        index=2,
        format_func=lambda x: (
            f"{x} nodes"
        ),
    )


with col3:

    edges_per_node = st.slider(
        "Edges per Node",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
    )


col4, col5, col6 = st.columns(3)


with col4:

    customers = st.slider(
        "Customers",
        min_value=5,
        max_value=100,
        value=50,
        step=5,
    )


with col5:

    vehicles = st.slider(
        "Vehicles",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
    )


with col6:

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )


# ============================================================
# ALGORITHM CARDS
# ============================================================

st.markdown(
    "## 🧠 Algorithms Under Test"
)


a1, a2, a3, a4 = st.columns(4)


with a1:

    html(
        """
        <div class="algorithm-card classical">

            <div class="algorithm-icon">
                🔵
            </div>

            <div class="algorithm-name">
                PSO
            </div>

            <div class="algorithm-description">
                Classical Particle Swarm
                Optimization
            </div>

        </div>
        """
    )


with a2:

    html(
        """
        <div class="algorithm-card quantum">

            <div class="algorithm-icon">
                ⚛️
            </div>

            <div class="algorithm-name">
                QPSO
            </div>

            <div class="algorithm-description">
                Quantum-inspired Particle
                Swarm Optimization
            </div>

        </div>
        """
    )


with a3:

    html(
        """
        <div class="algorithm-card warm">

            <div class="algorithm-icon">
                🔥
            </div>

            <div class="algorithm-name">
                Warm-QPSO
            </div>

            <div class="algorithm-description">
                Classical PSO warm start
                followed by QPSO
            </div>

        </div>
        """
    )


with a4:

    html(
        """
        <div class="algorithm-card hybrid">

            <div class="algorithm-icon">
                🧩
            </div>

            <div class="algorithm-name">
                Decomposed Warm-QPSO
            </div>

            <div class="algorithm-description">
                Decomposition + warm start
                + global repair
            </div>

        </div>
        """
    )


# ============================================================
# ADVANCED SETTINGS
# ============================================================

with st.expander(
    "🔧 Advanced Benchmark Settings"
):

    adv1, adv2, adv3 = st.columns(3)


    with adv1:

        particles = st.slider(
            "Particles",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )


    with adv2:

        iterations = st.slider(
            "Iterations",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
        )


    with adv3:

        max_clusters = min(
            10,
            int(customers),
            int(vehicles),
        )


        if max_clusters >= 2:

            clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=max_clusters,
                value=max_clusters,
                step=1,
            )

        else:

            clusters = 1

            st.caption(
                "At least 2 customers and "
                "2 vehicles are recommended."
            )


# ============================================================
# BENCHMARK INFORMATION
# ============================================================

html(
    """
    <div class="benchmark-info">

        <div class="benchmark-info-icon">
            ℹ️
        </div>

        <div>

            <strong>
                Fair Comparison
            </strong>

            <div>
                All algorithms use the same network,
                traffic scenario, fleet configuration,
                particles, iterations and random seed.
            </div>

        </div>

    </div>
    """
)


# ============================================================
# RUN BUTTON
# ============================================================

st.markdown("")


run_col1, run_col2, run_col3 = st.columns(
    [1, 2, 1]
)


with run_col2:

    run_benchmark = st.button(
        "⚡ RUN COMPLETE BENCHMARK",
        type="primary",
        width="stretch",
    )


# ============================================================
# RUN BENCHMARK
# ============================================================

if run_benchmark:

    progress = st.progress(
        0
    )

    status = st.empty()


    try:

        status.markdown(
            "### ⚡ Running all four algorithms..."
        )


        benchmark_start = (
            time.perf_counter()
        )


        results = run_algorithm_comparison(

            scenario=scenario,

            customers=int(
                customers
            ),

            vehicles=int(
                vehicles
            ),

            network_size=int(
                network_size
            ),

            edges_per_node=int(
                edges_per_node
            ),

            particles=int(
                particles
            ),

            iterations=int(
                iterations
            ),

            clusters=int(
                clusters
            ),

            seed=int(
                seed
            ),
        )


        benchmark_runtime = (
            time.perf_counter()
            - benchmark_start
        )


        progress.progress(
            1.0
        )


        status.empty()


        if not isinstance(
            results,
            dict,
        ):

            raise ValueError(
                "Benchmark backend did not "
                "return a results dictionary."
            )


        st.session_state[
            "benchmark_results"
        ] = results


        st.session_state[
            "benchmark_configuration"
        ] = {

            "scenario": scenario,

            "customers": int(
                customers
            ),

            "vehicles": int(
                vehicles
            ),

            "network_size": int(
                network_size
            ),

            "edges_per_node": int(
                edges_per_node
            ),

            "particles": int(
                particles
            ),

            "iterations": int(
                iterations
            ),

            "clusters": int(
                clusters
            ),

            "seed": int(
                seed
            ),

            "benchmark_runtime": (
                benchmark_runtime
            ),
        }


        st.success(
            "Benchmark completed successfully."
        )


    except Exception as error:

        progress.empty()
        status.empty()

        st.error(
            f"Benchmark failed: {error}"
        )

        st.stop()


# ============================================================
# LOAD RESULTS
# ============================================================

benchmark_results = st.session_state.get(
    "benchmark_results",
    {},
)


if not benchmark_results:

    st.markdown("")


    html(
        """
        <div class="benchmark-ready">

            <div class="ready-icon">
                ⚡
            </div>

            <div class="ready-title">
                Benchmark Ready
            </div>

            <div class="ready-text">
                Configure the experiment from the
                sidebar and run the benchmark to compare
                Q-ROUTE optimization algorithms.
            </div>

        </div>
        """
    )

    st.stop()


# ============================================================
# RESULTS DATAFRAME
# ============================================================

rows = []


for algorithm, result in benchmark_results.items():

    if not isinstance(
        result,
        dict,
    ):
        continue


    rows.append(
        {
            "Algorithm": algorithm,

            "Fitness": safe_float(
                result.get(
                    "fitness",
                    0,
                )
            ),

            "Travel Time": safe_float(
                result.get(
                    "travel_time",
                    0,
                )
            ),

            "Distance": safe_float(
                result.get(
                    "distance",
                    0,
                )
            ),

            "Congestion (%)": (
                safe_float(
                    result.get(
                        "congestion",
                        0,
                    )
                )
                * 100
            ),

            "Runtime (s)": safe_float(
                result.get(
                    "execution_time",
                    0,
                )
            ),

            "Valid": result.get(
                "valid",
                False,
            ),
        }
    )


benchmark_df = pd.DataFrame(
    rows
)


# ============================================================
# RESULTS
# ============================================================

st.markdown(
    "## 🏆 Benchmark Results"
)


if benchmark_df.empty:

    st.warning(
        "No benchmark results were returned."
    )

    st.stop()


# ============================================================
# WINNERS
# ============================================================

valid_df = benchmark_df[
    benchmark_df["Valid"] == True
].copy()


if valid_df.empty:

    valid_df = benchmark_df[
        benchmark_df["Fitness"].notna()
    ].copy()


if not valid_df.empty:

    best_fitness = valid_df.loc[
        valid_df["Fitness"].idxmin()
    ]

    best_time = valid_df.loc[
        valid_df["Travel Time"].idxmin()
    ]

    best_distance = valid_df.loc[
        valid_df["Distance"].idxmin()
    ]

    best_congestion = valid_df.loc[
        valid_df["Congestion (%)"].idxmin()
    ]

    fastest = valid_df.loc[
        valid_df["Runtime (s)"].idxmin()
    ]


    w1, w2, w3, w4, w5 = st.columns(5)


    with w1:

        st.metric(
            "🏆 Best Fitness",
            best_fitness["Algorithm"],
            f"{best_fitness['Fitness']:.4f}",
        )


    with w2:

        st.metric(
            "⏱️ Best Travel Time",
            best_time["Algorithm"],
            f"{best_time['Travel Time']:.2f}",
        )


    with w3:

        st.metric(
            "🛣️ Best Distance",
            best_distance["Algorithm"],
            f"{best_distance['Distance']:.2f}",
        )


    with w4:

        st.metric(
            "🚦 Lowest Congestion",
            best_congestion["Algorithm"],
            f"{best_congestion['Congestion (%)']:.2f}%",
        )


    with w5:

        st.metric(
            "⚡ Fastest",
            fastest["Algorithm"],
            format_seconds(
                fastest["Runtime (s)"]
            ),
        )


# ============================================================
# COMPARISON TABLE
# ============================================================

st.markdown(
    "### 📋 Algorithm Comparison"
)


table_df = benchmark_df.copy()


table_df["Fitness"] = table_df[
    "Fitness"
].map(
    lambda x: (
        f"{x:.4f}"
        if pd.notna(x)
        else "—"
    )
)


table_df["Travel Time"] = table_df[
    "Travel Time"
].map(
    lambda x: (
        f"{x:.2f}"
        if pd.notna(x)
        else "—"
    )
)


table_df["Distance"] = table_df[
    "Distance"
].map(
    lambda x: (
        f"{x:.2f}"
        if pd.notna(x)
        else "—"
    )
)


table_df["Congestion (%)"] = table_df[
    "Congestion (%)"
].map(
    lambda x: (
        f"{x:.2f}%"
        if pd.notna(x)
        else "—"
    )
)


table_df["Runtime (s)"] = table_df[
    "Runtime (s)"
].map(
    lambda x: f"{x:.3f}"
)


table_df["Valid"] = table_df[
    "Valid"
].map(
    lambda x: (
        "✓ Valid"
        if x
        else "✗ Invalid"
    )
)


st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
)


# ============================================================
# FITNESS COMPARISON
# ============================================================

st.markdown(
    "### 🎯 Fitness Comparison"
)


fitness_fig = go.Figure()


fitness_fig.add_trace(
    go.Bar(
        x=benchmark_df[
            "Algorithm"
        ],

        y=benchmark_df[
            "Fitness"
        ],

        text=[
            f"{x:.2f}"
            for x in benchmark_df[
                "Fitness"
            ]
        ],

        textposition="auto",

        name="Fitness",
    )
)


fitness_fig.update_layout(

    title=(
        "Solution Quality by Algorithm"
    ),

    xaxis_title="Algorithm",

    yaxis_title=(
        "Fitness — Lower is Better"
    ),

    template="plotly_white",

    height=450,

    margin=dict(
        l=50,
        r=30,
        t=70,
        b=80,
    ),
)


st.plotly_chart(
    fitness_fig,
    width="stretch",
)


# ============================================================
# TRAVEL TIME AND DISTANCE
# ============================================================

st.markdown(
    "### 🛣️ Travel Cost Comparison"
)


cost_fig = go.Figure()


cost_fig.add_trace(
    go.Bar(
        x=benchmark_df[
            "Algorithm"
        ],

        y=benchmark_df[
            "Travel Time"
        ],

        name="Travel Time",
    )
)


cost_fig.add_trace(
    go.Bar(
        x=benchmark_df[
            "Algorithm"
        ],

        y=benchmark_df[
            "Distance"
        ],

        name="Distance",
    )
)


cost_fig.update_layout(

    title=(
        "Travel Time vs Distance"
    ),

    xaxis_title="Algorithm",

    yaxis_title="Cost",

    barmode="group",

    template="plotly_white",

    height=450,
)


st.plotly_chart(
    cost_fig,
    width="stretch",
)


# ============================================================
# CONGESTION
# ============================================================

st.markdown(
    "### 🚦 Congestion Comparison"
)


congestion_fig = go.Figure()


congestion_fig.add_trace(
    go.Bar(
        x=benchmark_df[
            "Algorithm"
        ],

        y=benchmark_df[
            "Congestion (%)"
        ],

        text=[
            f"{x:.1f}%"
            for x in benchmark_df[
                "Congestion (%)"
            ]
        ],

        textposition="auto",

        name="Congestion",
    )
)


congestion_fig.update_layout(

    title=(
        "Average Route Congestion"
    ),

    xaxis_title="Algorithm",

    yaxis_title="Congestion (%)",

    template="plotly_white",

    height=420,
)


st.plotly_chart(
    congestion_fig,
    width="stretch",
)


# ============================================================
# RUNTIME
# ============================================================

st.markdown(
    "### ⚡ Execution Time"
)


runtime_fig = go.Figure()


runtime_fig.add_trace(
    go.Bar(
        x=benchmark_df[
            "Algorithm"
        ],

        y=benchmark_df[
            "Runtime (s)"
        ],

        text=[
            f"{x:.3f}s"
            for x in benchmark_df[
                "Runtime (s)"
            ]
        ],

        textposition="auto",

        name="Runtime",
    )
)


runtime_fig.update_layout(

    title=(
        "Optimization Execution Time"
    ),

    xaxis_title="Algorithm",

    yaxis_title="Runtime (seconds)",

    template="plotly_white",

    height=420,
)


st.plotly_chart(
    runtime_fig,
    width="stretch",
)


# ============================================================
# CONVERGENCE
# ============================================================

st.markdown(
    "### 📈 Convergence Comparison"
)


convergence_fig = go.Figure()


has_history = False


for algorithm, result in (
    benchmark_results.items()
):

    history = extract_history(
        result
    )


    if not history:

        continue


    has_history = True


    convergence_fig.add_trace(
        go.Scatter(

            x=list(
                range(
                    len(history)
                )
            ),

            y=history,

            mode="lines",

            name=algorithm,

            line=dict(
                width=3,
            ),
        )
    )


if has_history:

    convergence_fig.update_layout(

        title=(
            "Fitness Convergence "
            "Across Algorithms"
        ),

        xaxis_title="Iteration",

        yaxis_title="Fitness",

        template="plotly_white",

        height=500,

        hovermode="x unified",
    )


    st.plotly_chart(
        convergence_fig,
        width="stretch",
    )

else:

    st.info(
        "Convergence history is not "
        "available for this benchmark."
    )


# ============================================================
# ALGORITHM DETAILS
# ============================================================

st.markdown(
    "### 🔍 Algorithm Details"
)


for algorithm, result in (
    benchmark_results.items()
):

    if not isinstance(
        result,
        dict,
    ):
        continue


    with st.expander(
        f"⚙️ {algorithm}"
    ):

        if result.get(
            "error"
        ):

            st.error(
                result["error"]
            )

            continue


        d1, d2, d3, d4 = st.columns(
            4
        )


        with d1:

            st.metric(
                "Fitness",
                format_number(
                    result.get(
                        "fitness",
                        0,
                    ),
                    4,
                ),
            )


        with d2:

            st.metric(
                "Travel Time",
                format_number(
                    result.get(
                        "travel_time",
                        0,
                    )
                ),
            )


        with d3:

            st.metric(
                "Distance",
                format_number(
                    result.get(
                        "distance",
                        0,
                    )
                ),
            )


        with d4:

            st.metric(
                "Congestion",
                (
                    f"{safe_float(result.get('congestion', 0)) * 100:.2f}%"
                ),
            )


        st.write(
            "**Runtime:**",
            format_seconds(
                result.get(
                    "execution_time",
                    0,
                )
            ),
        )


        st.write(
            "**Valid:**",
            (
                "✓ Yes"
                if result.get(
                    "valid",
                    False,
                )
                else "✗ No"
            ),
        )


        if algorithm == (
            "Decomposed Warm-QPSO"
        ):

            st.write(
                "**Clusters:**",
                result.get(
                    "effective_clusters",
                    result.get(
                        "number_of_clusters",
                        "—",
                    ),
                ),
            )

            st.write(
                "**Warm-start iterations:**",
                result.get(
                    "warm_iterations",
                    "—",
                ),
            )

            st.write(
                "**Warm-start particles:**",
                result.get(
                    "warm_particles",
                    "—",
                ),
            )

            st.write(
                "**Repair improvement:**",
                format_number(
                    result.get(
                        "repair_improvement",
                        0,
                    ),
                    4,
                ),
            )


# ============================================================
# EXPERIMENT DETAILS
# ============================================================

st.markdown(
    "### 🧪 Experiment Details"
)


config = st.session_state.get(
    "benchmark_configuration",
    {},
)


e1, e2, e3, e4 = st.columns(
    4
)


with e1:

    st.metric(
        "Traffic",
        config.get(
            "scenario",
            scenario,
        ),
    )


with e2:

    st.metric(
        "Network",
        f"{config.get('network_size', network_size)} nodes",
    )


with e3:

    st.metric(
        "Customers",
        config.get(
            "customers",
            customers,
        ),
    )


with e4:

    st.metric(
        "Vehicles",
        config.get(
            "vehicles",
            vehicles,
        ),
    )


e5, e6, e7, e8 = st.columns(
    4
)


with e5:

    st.metric(
        "Particles",
        config.get(
            "particles",
            particles,
        ),
    )


with e6:

    st.metric(
        "Iterations",
        config.get(
            "iterations",
            iterations,
        ),
    )


with e7:

    st.metric(
        "Clusters",
        config.get(
            "clusters",
            clusters,
        ),
    )


with e8:

    st.metric(
        "Seed",
        config.get(
            "seed",
            seed,
        ),
    )


# ============================================================
# FAIR COMPARISON NOTE
# ============================================================

html(
    """
    <div class="fair-comparison-note">

        <div class="fair-title">
            🔬 Experimental Note
        </div>

        <div class="fair-text">

            The benchmark compares algorithms on the
            same generated transportation network,
            traffic scenario and VRP configuration.
            Fitness is minimized, so a lower value
            represents a better solution under the
            configured multi-objective cost function.

        </div>

    </div>
    """
)


# ============================================================
# EXPORT
# ============================================================

st.markdown(
    "### 💾 Export Benchmark"
)


export_df = benchmark_df.copy()


csv_data = export_df.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Benchmark CSV",
    data=csv_data,
    file_name="qroute_benchmark.csv",
    mime="text/csv",
    width="stretch",
)


# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")


n1, n2 = st.columns(
    2
)


with n1:

    if st.button(
        "⚛️ Run Another Benchmark",
        width="stretch",
    ):

        st.rerun()


with n2:

    if st.button(
        "📊 View Latest Optimization Result",
        width="stretch",
    ):

        if st.session_state.get(
            "optimizer_result"
        ) is not None:

            st.switch_page(
                "pages/4_Results.py"
            )

        else:

            st.warning(
                "No individual optimization result "
                "is currently available."
            )


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="benchmark-footer">

        <div class="benchmark-footer-title">
            ⚛️ Q-ROUTE
        </div>

        <div class="benchmark-footer-subtitle">
            Quantum-Inspired Intelligent
            Traffic Route Optimization
        </div>

        <div class="benchmark-footer-message">
            Performance Lab
            &nbsp; • &nbsp;
            Smarter Routes
            &nbsp; • &nbsp;
            Greener Cities 🌱
        </div>

    </div>
    """
)