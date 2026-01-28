import pandas as pd
import pulp
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
SAVE_PLOTS = True

print("--- Starting Airport Gate Solver ---")

# --- 1. Data Generation ---
print("Step 1: Generating Synthetic Data...")
# Gates
gates_dict = {
    'Gate_ID': ['G1', 'G2', 'G3', 'G4', 'G5'],
    'Distance_m': [100, 300, 500, 700, 1000]
}
df_gates = pd.DataFrame(gates_dict)

# Flights
num_flights = 15
flights_data = []
for i in range(num_flights):
    flight_id = f"F{101+i}"
    arrival = random.randint(480, 1200) # 08:00 to 20:00
    duration = random.randint(45, 120)
    departure = arrival + duration
    passengers = random.randint(50, 250)
    flights_data.append({
        'Flight_ID': flight_id,
        'Arrival': arrival,
        'Departure': departure,
        'Duration': duration,
        'Passengers': passengers
    })
df_flights = pd.DataFrame(flights_data)
print(f"Generated {num_flights} flights and {len(df_gates)} gates.")

# --- 2. Conflict Detection ---
print("Step 2: Detecting Variable Conflcits...")
def check_overlap(row_a, row_b):
    buffer = 15
    start_a, end_a = row_a['Arrival'], row_a['Departure'] + buffer
    start_b, end_b = row_b['Arrival'], row_b['Departure'] + buffer
    return (start_a < end_b) and (start_b < end_a)

incompatible_pairs = []
for i in range(len(df_flights)):
    for j in range(i + 1, len(df_flights)):
        flight_a = df_flights.iloc[i]
        flight_b = df_flights.iloc[j]
        if check_overlap(flight_a, flight_b):
            incompatible_pairs.append( (flight_a['Flight_ID'], flight_b['Flight_ID']) )

print(f"Found {len(incompatible_pairs)} conflicting pairs.")

# --- 3. Optimization (ILP) ---
print("Step 3: Solving Integer Linear Pricing Model...")
prob = pulp.LpProblem("Airport_Gate_Solver", pulp.LpMinimize)

# Variables
flights = df_flights['Flight_ID'].tolist()
gates = df_gates['Gate_ID'].tolist()
x = pulp.LpVariable.dicts("x", (flights, gates), cat='Binary')

# Objective
objective_terms = []
for idx, f_row in df_flights.iterrows():
    f_id = f_row['Flight_ID']
    pax = f_row['Passengers']
    for _, g_row in df_gates.iterrows():
        g_id = g_row['Gate_ID']
        dist = g_row['Distance_m']
        objective_terms.append(x[f_id][g_id] * pax * dist)
prob += pulp.lpSum(objective_terms)

# Constraints
for f in flights:
    prob += pulp.lpSum([x[f][g] for g in gates]) == 1
for (fa, fb) in incompatible_pairs:
    for g in gates:
        prob += x[fa][g] + x[fb][g] <= 1

prob.solve()
status = pulp.LpStatus[prob.status]
optimized_score = pulp.value(prob.objective)
print(f"Solver Status: {status}")
print(f"Optimized Distance: {optimized_score:,.0f} pax-m")

# --- 4. Impact Analysis ---
avg_gate_dist = df_gates['Distance_m'].mean()
total_pax = df_flights['Passengers'].sum()
random_score = total_pax * avg_gate_dist
pct_improvement = ((random_score - optimized_score) / random_score) * 100
print(f"Baseline Score: {random_score:,.0f} pax-m")
print(f"Efficiency Improvement: {pct_improvement:.2f}%")

# --- 5. Visualization Export ---
print("Step 5: Generating Visualizations...")

# Prepare Schedule Data
schedule = []
for f in flights:
    for g in gates:
        if pulp.value(x[f][g]) == 1:
            row = df_flights[df_flights['Flight_ID'] == f].iloc[0]
            schedule.append({
                'Flight': f, 'Gate': g, 'Arrival': row['Arrival'],
                'Duration': row['Duration'], 'Passengers': row['Passengers']
            })
df_schedule = pd.DataFrame(schedule)
df_schedule['Gate_Index'] = df_schedule['Gate'].apply(lambda x: int(x[1:]) - 1)

# Plot 1: 3D Visualization
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
for _, row in df_schedule.iterrows():
    x_pos = row['Gate_Index']
    y_pos = row['Arrival']
    z_pos = 0
    dx = 0.4
    dy = row['Duration']
    dz = row['Passengers']
    color = plt.cm.viridis(row['Gate_Index'] / 5)
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=color, alpha=0.8, edgecolor='black')
ax.set_xlabel('Gate ID'); ax.set_ylabel('Time'); ax.set_zlabel('Passengers')
ax.set_title("3D Airport Operations Volume")
plt.savefig('images/viz_3d_operations.png')
print("Saved images/viz_3d_operations.png")

# Plot 2: Cumulative Savings
df_viz = df_schedule.sort_values('Arrival').reset_index(drop=True)
df_viz['Optimized_Cost'] = df_viz['Passengers'] * df_viz['Gate'].apply(lambda g: df_gates[df_gates['Gate_ID']==g]['Distance_m'].values[0])
df_viz['Cum_Optimized'] = df_viz['Optimized_Cost'].cumsum()
df_viz['Cum_Random'] = (df_viz['Passengers'] * avg_gate_dist).cumsum()

plt.figure(figsize=(12, 6))
plt.plot(df_viz['Flight'], df_viz['Cum_Random'], 'r--', label='Random Baseline')
plt.plot(df_viz['Flight'], df_viz['Cum_Optimized'], 'g-', label='Optimized')
plt.fill_between(df_viz['Flight'], df_viz['Cum_Optimized'], df_viz['Cum_Random'], color='green', alpha=0.1)
plt.title("Cumulative Walking Distance Savings")
plt.legend()
plt.savefig('images/viz_cumulative_impact.png')
print("Saved images/viz_cumulative_impact.png")

print("--- Analysis Complete ---")
