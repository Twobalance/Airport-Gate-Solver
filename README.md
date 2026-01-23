# Airport Gate Solver ✈️

**A Data Science & Operations Research Portfolio Project**

## 📖 Executive Summary
This project demonstrates how **Integer Linear Programming (ILP)** can solve complex logistical challenges in aviation. By using Python and the `pulp` library, we optimize the assignment of flights to airport gates to **minimize passenger walking distance**.

**Key Results:**
-   **Method:** Binary Optimization ($0$ or $1$ decisions).
-   **Metric:** Total Passenger-Meters Walked.
-   **Outcome:** ~40-60% reduction in walking distance compared to random/naive assignment.

---

## 🚀 How to Run this Project

### Option 1: The Jupyter Notebook (Recommended)
1.  Open `Airport-Gate-Solver.ipynb` in VS Code or Jupyter Lab.
2.  Click **"Run All"**.
3.  The notebook will guide you through the Data Generation, Math Logic, and 3D Visualizations.

### Option 2: The Standalone Script
If you want to generate the charts immediately without opening a notebook:
```bash
# 1. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install pandas pulp matplotlib numpy seaborn

# 3. Run the analysis
python3 run_model.py
```
This will verify the math and save the following images to your folder:
-   `viz_3d_operations.png`: A 3D view of gate intensity.
-   `viz_cumulative_impact.png`: A graph showing how savings accumulate.

---

## 🧠 The "Ultimate" Step-by-Step Explanation

### Step 1: The Business Problem
Airports are like giant puzzles. You have **Flights** (Demand) and **Gates** (Supply).
-   Some gates are close to the exit (User-friendly).
-   Some gates are far away (User-hostile).
-   **Goal:** Put the biggest planes at the closest gates.
-   **Constraint:** TWO planes cannot be at the same gate at the same time.

### Step 2: The Data
We simulate a busy day:
-   **Flights:** 15 random arrivals with different passenger counts (50-250 pax).
-   **Gates:** 5 gates, ranked by distance (100m to 1000m).

### Step 3: The Conflict Matrix
Before optimizing, we must know the rules. We compare every flight against every other flight.
-   *If Flight A lands at 10:00 and leaves at 11:30...*
-   *And Flight B lands at 11:00...*
-   **Overlap!** They cannot share a gate. We record this pair as a "Conflict".

### Step 4: The Solver (The "Magic")
We don't guess. We use **Integer Linear Programming**.
We create a variable $x_{i,j}$ which is either 1 (Assign) or 0 (Don't Assign).

**The Equation:**
$$ \text{Minimize } \sum (x_{i,j} \times \text{Passengers}_i \times \text{Distance}_j) $$

**The Rules:**
1.  Every flight must go somewhere ($\sum x = 1$).
2.  Conflicting flights cannot go to the same place ($x_A + x_B \le 1$).

### Step 5: Visual Proof
We use `matplotlib` to prove our solution works:
-   **Gantt Chart:** Shows the schedule. No bars overlap!
-   **3D Chart:** Shows the volume of operations.
