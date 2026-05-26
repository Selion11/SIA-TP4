import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Load and Prepare the Data
# ==========================================
# Make sure 'europe.csv' is in the same folder as your script
try:
    df = pd.read_csv('../data/europe.csv')
except FileNotFoundError:
    print("Error: Please ensure 'europe.csv' is in the current directory.")
    exit()

# Extract the country names and the numeric features
countries = df['Country'].values
# Drop 'Country' to keep only numeric data (Area, GDP, Inflation, etc.)
data = df.drop('Country', axis=1).values

# Neural Networks require normalized/scaled data to work properly, 
# otherwise large numbers (like GDP) will dominate small numbers (like Pop.growth).
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# ==========================================
# 2. Implement the Kohonen Network
# ==========================================
# We have 28 countries. A 5x5 grid gives us 25 neurons, 
# or a 6x6 gives us 36 neurons, which is a good size to spread them out.
grid_rows = 6
grid_cols = 6

# Initialize the Self-Organizing Map
som = MiniSom(x=grid_rows, y=grid_cols, 
              input_len=data_scaled.shape[1], 
              sigma=1.0, 
              learning_rate=0.5, 
              random_seed=42) # Seed for reproducibility

# Initialize weights and train the network
som.random_weights_init(data_scaled)
print("Training the Kohonen Network...")
som.train_random(data_scaled, num_iteration=2000)
print("Training complete!")

# ==========================================
# 3. Generate Required Graphs
# ==========================================

# Setup the figure for multiple subplots
fig = plt.figure(figsize=(18, 5))

# --- Graph 1: Association of Countries (Map of Results) ---
# "Asociar países que posean las mismas características... Realizar al menos un gráfico"
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title("1. Country Map (Similar countries are closer)")

# Plot an empty grid
ax1.pcolor(np.zeros((grid_rows, grid_cols)), cmap='Greys', edgecolors='k', alpha=0) 

# Place each country on its "winning" neuron
for i, x in enumerate(data_scaled):
    w = som.winner(x)  # Get the coordinates of the winning neuron for this country
    # Add a bit of random noise (jitter) to coordinates so names don't perfectly overlap
    jitter_x = np.random.uniform(-0.3, 0.3)
    jitter_y = np.random.uniform(-0.3, 0.3)
    ax1.text(w[0] + 0.5 + jitter_x, w[1] + 0.5 + jitter_y, countries[i], 
             ha='center', va='center', fontsize=8, 
             bbox=dict(facecolor='white', alpha=0.5, lw=0))

ax1.set_xlim([0, grid_rows])
ax1.set_ylim([0, grid_cols])
ax1.set_xticks(np.arange(grid_rows))
ax1.set_yticks(np.arange(grid_cols))
ax1.grid(True, linestyle=':', alpha=0.6)

# --- Graph 2: Distance Plot (U-Matrix) ---
# "Realizar un gráfico que muestre las distancias promedio entre neuronas vecinas."
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("2. U-Matrix (Average Distance between neighbors)")

# distance_map() returns the average distance of a neuron to its neighbors
u_matrix = som.distance_map()
# Transpose the U-matrix to match the x-y orientation of the plot
cax = ax2.pcolor(u_matrix.T, cmap='viridis', edgecolors='k') 
fig.colorbar(cax, ax=ax2, label='Distance (Dark=Clusters, Light=Boundaries)')

# --- Graph 3: Number of Elements per Neuron ---
# "Analizar la cantidad de elementos que fueron asociados a cada neurona."
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title("3. Hits per Neuron (Count of associated elements)")

# activation_response() counts how many times each neuron was the winner
hit_map = som.activation_response(data_scaled)
cax3 = ax3.pcolor(hit_map.T, cmap='Blues', edgecolors='k')
fig.colorbar(cax3, ax=ax3, label='Number of Countries')

# Add the exact number as text inside the boxes
for i in range(grid_rows):
    for j in range(grid_cols):
        count = int(hit_map[i, j])
        if count > 0:
            ax3.text(i + 0.5, j + 0.5, str(count), 
                     ha='center', va='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig("Kohonen_Network_Results.png", dpi=300)  # Save the figure
# Print text analysis for the final")

# Print text analysis for the final point
print("\n--- Analysis of Elements per Neuron ---")
print(f"Total number of neurons: {grid_rows * grid_cols}")
print(f"Number of neurons with at least 1 country associated: {np.count_nonzero(hit_map)}")
print(f"Maximum number of countries in a single neuron: {int(np.max(hit_map))}")