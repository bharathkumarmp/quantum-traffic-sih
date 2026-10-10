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
