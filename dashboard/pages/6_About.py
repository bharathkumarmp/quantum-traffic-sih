# ============================================================
# Q-ROUTE | ABOUT PAGE
# dashboard/pages/6_About.py
# ============================================================

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | About",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "6_About.css"

if CSS_PATH.exists():

    st.html(
        f"""
        <style>
        {CSS_PATH.read_text(encoding="utf-8")}
        </style>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
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

    st.html(
        """
        <div class="sidebar-message">

            <div class="sidebar-message-icon">
                🌱
            </div>

            <div>
                <strong>
                    Smarter Routes
                </strong>

                <br>

                <span>
                    Greener Cities
                </span>
            </div>

        </div>
        """
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="about-hero">

        <div class="about-hero-icon">
            ⚛️
        </div>

        <div class="about-eyebrow">
            QUANTUM-INSPIRED INTELLIGENT TRANSPORTATION
        </div>

        <h1>
            Q-ROUTE
        </h1>

        <h2>
            Intelligent Traffic Route Optimization
        </h2>

        <p>
            Smart routing for dynamic urban traffic
            using quantum-inspired metaheuristic optimization.
        </p>

    </div>
    """
)


# ============================================================
# ABOUT Q-ROUTE
# ============================================================

st.markdown(
    "## 🚦 About Q-ROUTE"
)

st.write(
    """
    Q-ROUTE is a quantum-inspired intelligent traffic
    route optimization platform designed for vehicle
    routing problems in dynamic transportation networks.
    """
)

st.write(
    """
    The system models transportation networks as weighted
    graphs and searches for high-quality vehicle routes
    while considering travel time, distance, congestion
    and routing constraints.
    """
)


# ============================================================
# PROBLEM
# ============================================================

st.markdown(
    "## 🎯 Problem We Address"
)


problem1, problem2, problem3 = st.columns(3)


with problem1:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🚗
            </div>

            <h3>
                Traffic Congestion
            </h3>

            <p>
                Urban traffic conditions change continuously,
                making static routes inefficient.
            </p>

        </div>
        """
    )


with problem2:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🛣️
            </div>

            <h3>
                Routing Complexity
            </h3>

            <p>
                Vehicle Routing Problems become computationally
                difficult as the number of customers increases.
            </p>

        </div>
        """
    )


with problem3:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                ⏱️
            </div>

            <h3>
                Operational Cost
            </h3>

            <p>
                Poor routing can increase travel time,
                distance and transportation cost.
            </p>

        </div>
        """
    )


# ============================================================
# PROPOSED SOLUTION
# ============================================================

st.markdown(
    "## 💡 Proposed Solution"
)

st.write(
    """
    Q-ROUTE combines classical and quantum-inspired
    optimization techniques into a hybrid routing framework.
    """
)


st.code(
    """
Traffic Scenario
        ↓
Traffic Data
        ↓
Weighted Transportation Graph
        ↓
Vehicle Routing Problem
        ↓
Large-Scale Decomposition
        ↓
PSO Warm Start
        ↓
Quantum-Inspired QPSO
        ↓
Cluster Solution Merge
        ↓
Global Boundary Repair
        ↓
Optimized Vehicle Routes
        ↓
Visualization & Benchmarking
    """,
    language="text",
)


# ============================================================
# ALGORITHMS
# ============================================================

st.markdown(
    "## 🧠 Optimization Algorithms"
)


alg1, alg2 = st.columns(2)


with alg1:

    st.html(
        """
        <div class="about-card algorithm-about">

            <div class="about-card-icon">
                🔵
            </div>

            <h3>
                PSO
            </h3>

            <strong>
                Particle Swarm Optimization
            </strong>

            <p>
                A classical population-based metaheuristic
                used as a baseline and as a warm-start mechanism.
            </p>

        </div>
        """
    )


with alg2:

    st.html(
        """
        <div class="about-card algorithm-about">

            <div class="about-card-icon">
                ⚛️
            </div>

            <h3>
                QPSO
            </h3>

            <strong>
                Quantum Particle Swarm Optimization
            </strong>

            <p>
                A quantum-inspired variant of particle swarm
                optimization using probabilistic position updates
                and a mean-best attractor.
            </p>

        </div>
        """
    )


alg3, alg4 = st.columns(2)


with alg3:

    st.html(
        """
        <div class="about-card algorithm-about">

            <div class="about-card-icon">
                🔥
            </div>

            <h3>
                Warm-QPSO
            </h3>

            <p>
                A short classical PSO search generates an initial
                high-quality solution that is injected into QPSO
                to improve its starting point.
            </p>

        </div>
        """
    )


with alg4:

    st.html(
        """
        <div class="about-card algorithm-about">

            <div class="about-card-icon">
                🧩
            </div>

            <h3>
                Decomposed Warm-QPSO
            </h3>

            <p>
                Large vehicle routing problems are divided into
                smaller subproblems, solved independently and
                then improved through global boundary repair.
            </p>

        </div>
        """
    )


# ============================================================
# QUANTUM INSPIRED
# ============================================================

st.markdown(
    "## ⚛️ Where Is the Quantum-Inspired Part?"
)

st.write(
    """
    Q-ROUTE uses Quantum Particle Swarm Optimization (QPSO).
    Instead of using the conventional PSO velocity-position
    update, QPSO uses a probabilistic position update based
    on the particle's best position, the swarm's mean-best
    position and a logarithmic random step.
    """
)


st.latex(
    r"""
    x_{i,j}^{t+1}
    =
    p_{i,j}
    \pm
    \beta
    \left|
    mbest_j-x_{i,j}^{t}
    \right|
    \ln\left(\frac{1}{u}\right)
    """
)


st.info(
    "QPSO is quantum-inspired; the current prototype "
    "does not require quantum hardware."
)


# ============================================================
# WARM START
# ============================================================

st.markdown(
    "## 🔥 Warm-Start Strategy"
)


warm1, warm2, warm3 = st.columns(3)


with warm1:

    st.html(
        """
        <div class="step-card">

            <div class="step-number">
                1
            </div>

            <h3>
                Classical Search
            </h3>

            <p>
                Run a short PSO optimization to obtain
                a useful initial solution.
            </p>

        </div>
        """
    )


with warm2:

    st.html(
        """
        <div class="step-card">

            <div class="step-number">
                2
            </div>

            <h3>
                Encode Solution
            </h3>

            <p>
                Convert the customer ordering into
                a random-key representation.
            </p>

        </div>
        """
    )


with warm3:

    st.html(
        """
        <div class="step-card">

            <div class="step-number">
                3
            </div>

            <h3>
                QPSO Search
            </h3>

            <p>
                Inject the warm solution into the QPSO
                population and continue optimization.
            </p>

        </div>
        """
    )


# ============================================================
# DECOMPOSITION
# ============================================================

st.markdown(
    "## 🧩 Large-Scale Decomposition"
)

st.write(
    """
    Large vehicle routing problems can become expensive when
    optimized as a single problem. Q-ROUTE therefore supports
    decomposition into smaller customer groups.
    """
)


dec1, dec2 = st.columns(2)


with dec1:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🧩
            </div>

            <h3>
                Cluster
            </h3>

            <p>
                Customers are divided into smaller subproblems
                using deterministic demand-aware grouping
                in the current prototype.
            </p>

        </div>
        """
    )


with dec2:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🔧
            </div>

            <h3>
                Repair
            </h3>

            <p>
                After the cluster solutions are merged,
                global boundary customers can be moved or
                exchanged to improve the complete solution.
            </p>

        </div>
        """
    )


# ============================================================
# OBJECTIVE
# ============================================================

st.markdown(
    "## 📐 Optimization Objective"
)

st.write(
    """
    The routing objective combines multiple transportation
    criteria into a single fitness value.
    """
)


st.latex(
    r"""
    F(R)=w_tT+w_dD+w_cC+P
    """
)


obj1, obj2, obj3 = st.columns(3)


with obj1:

    st.html(
        """
        <div class="weight-card">

            <div class="weight-value">
                0.50
            </div>

            <div class="weight-label">
                Travel Time Weight
            </div>

        </div>
        """
    )


with obj2:

    st.html(
        """
        <div class="weight-card">

            <div class="weight-value">
                0.30
            </div>

            <div class="weight-label">
                Distance Weight
            </div>

        </div>
        """
    )


with obj3:

    st.html(
        """
        <div class="weight-card">

            <div class="weight-value">
                0.20
            </div>

            <div class="weight-label">
                Congestion Weight
            </div>

        </div>
        """
    )


# ============================================================
# TRAFFIC
# ============================================================

st.markdown(
    "## 🚦 Dynamic Traffic Adaptation"
)

st.write(
    """
    Traffic conditions are represented through changes in
    edge travel time, congestion and road availability.
    """
)


traffic1, traffic2, traffic3 = st.columns(3)


with traffic1:

    st.html(
        """
        <div class="traffic-card normal">

            <div class="traffic-icon">
                🟢
            </div>

            <h3>
                Normal
            </h3>

            <p>
                Standard traffic conditions.
            </p>

        </div>
        """
    )


with traffic2:

    st.html(
        """
        <div class="traffic-card incident">

            <div class="traffic-icon">
                🟡
            </div>

            <h3>
                Incident
            </h3>

            <p>
                Increased travel time and congestion
                on an affected road.
            </p>

        </div>
        """
    )


with traffic3:

    st.html(
        """
        <div class="traffic-card closure">

            <div class="traffic-icon">
                🔴
            </div>

            <h3>
                Road Closure
            </h3>

            <p>
                A blocked edge is removed from the
                feasible routing options.
            </p>

        </div>
        """
    )


# ============================================================
# EXPERIMENTAL VALIDATION
# ============================================================

st.markdown(
    "## 📊 Experimental Validation"
)


validation1, validation2 = st.columns(2)


with validation1:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🎲
            </div>

            <h3>
                Multi-Seed Benchmarking
            </h3>

            <p>
                Multiple random seeds are used to evaluate
                variation in solution quality and runtime.
            </p>

        </div>
        """
    )


with validation2:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                📈
            </div>

            <h3>
                Convergence Analysis
            </h3>

            <p>
                Fitness is tracked across optimization
                iterations to study search behavior.
            </p>

        </div>
        """
    )


validation3, validation4 = st.columns(2)


with validation3:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🧩
            </div>

            <h3>
                Decomposition Ablation
            </h3>

            <p>
                Different cluster counts can be compared
                to study scalability and solution quality.
            </p>

        </div>
        """
    )


with validation4:

    st.html(
        """
        <div class="about-card">

            <div class="about-card-icon">
                🎯
            </div>

            <h3>
                Exact Validation
            </h3>

            <p>
                Small VRP instances can be compared with
                an exact solver to measure optimality gaps.
            </p>

        </div>
        """
    )


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown(
    "## 🛠 Technology Stack"
)


tech1, tech2, tech3, tech4 = st.columns(4)


with tech1:

    st.html(
        """
        <div class="tech-card">

            <div class="tech-icon">
                🐍
            </div>

            <h3>
                Python
            </h3>

            <p>
                Optimization algorithms,
                graph modeling and experiments.
            </p>

        </div>
        """
    )


with tech2:

    st.html(
        """
        <div class="tech-card">

            <div class="tech-icon">
                🎛
            </div>

            <h3>
                Streamlit
            </h3>

            <p>
                Interactive dashboard,
                controls and visualization.
            </p>

        </div>
        """
    )


with tech3:

    st.html(
        """
        <div class="tech-card">

            <div class="tech-icon">
                📈
            </div>

            <h3>
                Plotly
            </h3>

            <p>
                Interactive convergence,
                benchmark and network charts.
            </p>

        </div>
        """
    )


with tech4:

    st.html(
        """
        <div class="tech-card">

            <div class="tech-icon">
                🕸
            </div>

            <h3>
                Graph Modeling
            </h3>

            <p>
                Weighted transportation networks
                and routing paths.
            </p>

        </div>
        """
    )


# ============================================================
# CURRENT PROTOTYPE
# ============================================================

st.markdown(
    "## ⚠️ Current Prototype"
)

st.warning(
    """
    The current prototype uses simulated traffic data and
    synthetic transportation networks. Large-scale clustering
    currently uses deterministic demand-aware/node-order grouping
    rather than live geographic or traffic-aware clustering.
    """
)


# ============================================================
# FUTURE WORK
# ============================================================

st.markdown(
    "## 🚀 Future Work"
)


future_items = [
    "Live traffic API integration",
    "Real road-network data",
    "Network-distance and traffic-aware clustering",
    "Predictive traffic modeling",
    "Hybrid quantum-classical optimization",
    "Larger vehicle routing instances",
    "Real-time route re-optimization",
]


for item in future_items:

    st.markdown(
        f"- {item}"
    )


# ============================================================
# RESEARCH POSITION
# ============================================================

st.markdown(
    "## 🔬 Research Position"
)

st.info(
    """
    Q-ROUTE does not assume that QPSO is universally better
    than classical optimization. The prototype evaluates PSO,
    QPSO, Warm-QPSO and Decomposed Warm-QPSO experimentally
    across different problem sizes and scenarios.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="about-footer">

        <div class="about-footer-title">
            ⚛️ Q-ROUTE
        </div>

        <div class="about-footer-subtitle">
            Quantum-Inspired Intelligent
            Traffic Route Optimization
        </div>

        <div class="about-footer-message">
            Smarter Routes &nbsp; • &nbsp;
            Greener Cities &nbsp; • &nbsp;
            Better Tomorrow 🌱
        </div>

    </div>
    """
)