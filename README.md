# Q-ROUTE 🚦⚛️

## Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems Using Metaheuristic Optimization

> A traffic-aware, multi-vehicle route optimization framework combining Vehicle Routing, PSO, QPSO, Warm-QPSO, decomposition, global boundary repair, and systematic algorithm benchmarking.

---

## 📌 Overview

**Q-ROUTE** is a traffic-aware multi-vehicle routing and optimization platform designed to solve **large-scale Vehicle Routing Problems (VRPs)** under changing traffic conditions.

The system models a transportation network as a **weighted graph**, where roads contain information such as:

- Distance
- Travel time
- Congestion
- Road availability

The routing problem considers:

- Multiple vehicles
- Multiple customers
- Customer demand
- Vehicle capacity
- Traffic conditions
- Road incidents
- Road blockages

The system searches for high-quality fleet-wide routing solutions using classical, quantum-inspired, and hybrid metaheuristic optimization techniques.

### Core Optimization Approaches

- **Dijkstra** — shortest-path baseline
- **PSO** — Particle Swarm Optimization
- **QPSO** — Quantum Particle Swarm Optimization
- **Warm-QPSO** — PSO warm-started QPSO
- **Decomposed Warm-QPSO** — scalable hybrid framework
- **Global Boundary Repair** — post-merge route improvement

The project also includes a **benchmarking framework** for comparing algorithms under identical experimental conditions.

---

# 🎯 Problem Statement

Urban transportation networks face several challenges:

- Traffic congestion
- Inefficient fleet routing
- Increasing delivery demand
- Road incidents
- Road blockages
- Vehicle capacity constraints
- Large routing search spaces
- Dynamic travel times

For a fleet with many vehicles and customers, the number of possible routing combinations grows extremely quickly.

Checking every possible solution is computationally impractical for large instances.

Therefore, Q-ROUTE investigates **metaheuristic and quantum-inspired optimization techniques** for finding high-quality solutions within practical computational limits.

---

# 💡 Core Idea

The central idea of Q-ROUTE is:

> **Traffic changes → road weights change → routing costs change → the optimization problem changes → the fleet can be re-optimized.**

Instead of treating routing as a fixed shortest-path problem, Q-ROUTE considers the **entire fleet**, its customers, vehicle capacities, and current transportation conditions.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Traffic Inputs    │
                    │ Customers / Vehicles│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Weighted Road      │
                    │      Graph          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    VRP Modeling     │
                    │ Demand + Capacity   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Fitness Function  │
                    │ Time + Distance +   │
                    │ Congestion + Penalty│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
           PSO              QPSO          Warm-QPSO
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                   Decomposed Warm-QPSO
                               │
                               ▼
                    Global Boundary Repair
                               │
                               ▼
                     Final Fleet Routes
                               │
                               ▼
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
           Fitness          Runtime        Convergence


# Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems Using Metaheuristic Optimization

<p align="center">

**A Traffic-Aware Multi-Vehicle Routing and Optimization Framework**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-orange?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-blue?logo=pandas)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Modeling-green)
![Pytest](https://img.shields.io/badge/Tests-Pytest-yellow?logo=pytest)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple)

</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Motivation](#-motivation)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Core Idea](#-core-idea)
- [System Architecture](#-system-architecture)
- [Transportation Network Model](#-transportation-network-model)
- [Dynamic Traffic Model](#-dynamic-traffic-model)
- [Vehicle Routing Problem](#-vehicle-routing-problem)
- [Mathematical Formulation](#-mathematical-formulation)
- [Fitness Function](#-fitness-function)
- [Optimization Algorithms](#-optimization-algorithms)
- [Dijkstra Baseline](#-dijkstra-baseline)
- [Particle Swarm Optimization](#-particle-swarm-optimization)
- [Quantum Particle Swarm Optimization](#-quantum-particle-swarm-optimization)
- [Warm-QPSO](#-warm-qpso)
- [Decomposed Warm-QPSO](#-decomposed-warm-qpso)
- [Global Boundary Repair](#-global-boundary-repair)
- [Dynamic Re-Routing](#-dynamic-re-routing)
- [Benchmarking Framework](#-benchmarking-framework)
- [Experimental Methodology](#-experimental-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [Preliminary Results](#-preliminary-results)
- [Validation](#-validation)
- [Convergence Analysis](#-convergence-analysis)
- [Scalability Analysis](#-scalability-analysis)
- [Ablation Study](#-ablation-study)
- [Dashboard](#-dashboard)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Running Tests](#-running-tests)
- [Running Experiments](#-running-experiments)
- [Research Contribution](#-research-contribution)
- [Innovation and Uniqueness](#-innovation-and-uniqueness)
- [Quantum vs Quantum-Inspired](#-quantum-vs-quantum-inspired)
- [Real-World Deployment](#-real-world-deployment)
- [Environmental and Economic Impact](#-environmental-and-economic-impact)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Research Direction](#-research-direction)
- [References](#-references)
- [Project Team](#-project-team)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

# 🚦 Overview

**Q-ROUTE** is a research-oriented traffic-aware vehicle routing and optimization platform designed to investigate classical, quantum-inspired, and hybrid metaheuristic algorithms for large-scale transportation optimization.

The system models a transportation network as a **weighted graph** and formulates fleet routing as a **Vehicle Routing Problem (VRP)**.

Each road contains information such as:

- Distance
- Travel time
- Congestion
- Availability
- Blockage status

The routing problem considers:

- Multiple vehicles
- Multiple customers
- Customer demand
- Vehicle capacity
- Road conditions
- Traffic scenarios
- Traffic incidents
- Road blockages

Q-ROUTE evaluates different optimization strategies to search for high-quality fleet-wide routes.

The main optimization approaches implemented in the project are:

1. Dijkstra
2. Particle Swarm Optimization (PSO)
3. Quantum Particle Swarm Optimization (QPSO)
4. Warm-QPSO
5. Decomposed Warm-QPSO

---

# 🎯 Problem Statement

Urban transportation systems face increasing challenges due to:

- Rapid urbanization
- Increasing vehicle density
- Traffic congestion
- Growing delivery demand
- Fleet management complexity
- Road incidents
- Dynamic travel times
- Infrastructure constraints

Traditional shortest-path algorithms are useful for finding a route between two points, but they do not directly solve the more complex fleet-level problem:

> **How can multiple vehicles serve multiple customers efficiently while satisfying vehicle constraints and minimizing transportation cost under changing traffic conditions?**

This is a **Vehicle Routing Problem (VRP)**.

For large instances, the number of possible routing configurations becomes extremely large.

Therefore, exhaustive search can become computationally expensive.

Q-ROUTE investigates metaheuristic and quantum-inspired optimization strategies to efficiently search this large solution space.

---

# 💡 Motivation

Consider a delivery fleet with:

```text
10 Vehicles
50 Customers
100+ Road Nodes
Hundreds of Road Connections
Changing Traffic Conditions
