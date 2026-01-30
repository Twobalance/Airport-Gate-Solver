# Optimizing Airport Gate Allocation: An Integer Linear Programming Approach

## ✈️ Executive Summary
This project addresses the **Airport Gate Assignment Problem (AGAP)**, a critical logistical challenge in aviation management. By leveraging **Integer Linear Programming (ILP)**, we demonstrate a mathematical framework to optimize flight-to-gate assignments. The primary objective is to minimize total passenger walking distance while satisfying stringent operational constraints, such as gate availability and flight schedule overlaps.

**Key Findings:**
*   **Optimization Methodology:** Binary Integer Programming using the Branch-and-Cut algorithm.
*   **Performance Gain:** Achieved a **40.68% reduction** in total passenger walking distance compared to stochastic (random) assignment baselines.
*   **Operational Scalability:** Successfully resolved complex scheduling conflicts for a simulated high-density operational window.

---

## 📊 Performance Overview

| Metric | Baseline (Stochastic) | Optimized (ILP) | Delta (Improvement) |
| :--- | :--- | :--- | :--- |
| **Total Walking Distance** | 1,073,280 pax-m | 636,700 pax-m | **↓ 40.68%** |
| **Average Distance per Pax** | 536.6 m | 318.3 m | **↓ 218.3 m** |

![Performance Comparison](images/viz_comparison_bar.png)
*Figure 1: Comparison of total passenger walking distance between baseline and optimized scenarios.*

---


## 📈 Operational Insights

### Optimized Flight Schedule
The visualization below shows the gapless scheduling achieved by the solver. The system prioritizes high-capacity flights (darker bars) at gates with shorter terminal distances ($G1, G2$).

![Optimized Schedule](images/viz_gantt_schedule.png)
*Figure 2: Gantt chart depicting the optimized allocation of flights across available gates over time.*

### 3.D Volumetric Analysis
This chart visualizes the "operational pressure" on the gate system, combining time, gate ID, and passenger volume into a single 3D metric.

![3D Operations](images/viz_3d_enhanced.png)
*Figure 3: 3D Visualization of Gate Operations (Time × Gate × Passenger Volume).*

---

## 🚀 Implementation & Reproducibility

### Repository Structure
*   `notebooks/Airport-Gate-Solver.ipynb`: Comprehensive notebook with step-by-step mathematical derivation and analysis.
*   `run_model.py`: Lightweight production script for running the optimization and generating core results.
*   `generate_readme_assets.py`: Advanced visualization suite used to generate academic-grade charts.
*   `images/` folder containing all high-resolution analytical visualizations.

### Installation

1.  **Initialize Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Execute Analysis:**
    ```bash
    python3 run_model.py
    ```

---
