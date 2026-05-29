import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter

# phase 1: creating a fake field with realistic nematode clusters
# we use a gaussian mixture model because nematodes dont just scatter randomly
# clump together in hotspots which is what we see in real agriculture
np.random.seed(42)  

# grid param
GRID_SIZE = 100
x = np.linspace(0, 100, GRID_SIZE)
y = np.linspace(0, 100, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# define 4 gaussian components to represent the infection spots
# component parameters (center, standard deviation, population proportion)
components = [
    {'center': (30, 25), 'sigma': 8, 'weight': 0.4},    # hotspot 1 (largest hotspot)
    {'center': (70, 35), 'sigma': 7, 'weight': 0.25},   # hotspot 2
    {'center': (50, 70), 'sigma': 6, 'weight': 0.2},    # hotspot 3
    {'center': (80, 80), 'sigma': 5, 'weight': 0.15},   # hotspot 4 (smallest hotspot)
]

# base nematode density map
nematode_density = np.zeros((GRID_SIZE, GRID_SIZE))

for comp in components:
    cx, cy = comp['center']
    sigma = comp['sigma']
    weight = comp['weight']
    
    # 2d gaussian formula
    gaussian = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    nematode_density += weight * gaussian

# normalising to realistic scale: 0-500 nematodes / 100cm^3
nematode_density = (nematode_density / nematode_density.max()) * 500

# adding a lil bit of noise to simulate natural diversity
noise = np.random.normal(0, 15, (GRID_SIZE, GRID_SIZE))
nematode_density = np.clip(nematode_density + noise, 0, 550)

n_samples = 50 # sample from 50 random locations
sample_indices = np.random.choice(GRID_SIZE * GRID_SIZE, n_samples, replace=False)
sample_rows = sample_indices // GRID_SIZE
sample_cols = sample_indices % GRID_SIZE
sample_coords = np.column_stack([sample_rows, sample_cols])
sample_values = nematode_density[sample_rows, sample_cols]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# first plot, baseline nematode density map
im1 = axes[0].contourf(X, Y, nematode_density, levels=15, cmap='YlOrRd')
axes[0].scatter(sample_cols, sample_rows, c='blue', s=50, marker='x', 
                label='Sample locations (n=50)', linewidth=2)
axes[0].set_xlabel('Distance (m)')
axes[0].set_ylabel('Distance (m)')
axes[0].set_title('Baseline - Nematode Density Distribution\n' +
                  'Gaussian Mixture Model (4 components)')
axes[0].legend()
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Nematode density (per 100cm^3)')

# second plot, heatmap with grid
im2 = axes[1].imshow(nematode_density, extent=[0, 100, 0, 100], 
                     origin='lower', cmap='YlOrRd', aspect='auto')
axes[1].scatter(sample_cols, sample_rows, c='blue', s=50, marker='x', 
                label='Sample locations (n=50)', linewidth=2)
axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Distance (m)')
axes[1].set_title('Baseline - Heatmap')
axes[1].legend()
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('Nematode density (per 100cm^3)')

plt.tight_layout()
plt.savefig('phase1_baseline.png', dpi=300, bbox_inches='tight')
plt.show()

with open("output.txt", "w", encoding="utf-8") as f:
    print("VERIFY THIS!!!", file=f)
    print(f"\nGrid info:", file=f)
    print(f"    dimensions: {GRID_SIZE} x {GRID_SIZE} cells", file=f)
    print(f"    total points: {GRID_SIZE ** 2}", file=f)
    print(f"    spatial resolution: 1 meter per cell", file=f)
    print(f"\nNematode distribution model:", file=f)
    print(f"    method: gaussian mixture model (k=4 components)", file=f)
    print(f"    rationale: reflects clumped distribution in literature", file=f)
    print(f"\nHotspot parameters:", file=f)
    for i, comp in enumerate(components, 1):
        print(
            f"{i}. centre={comp['center']}, "
            f"sigma={comp['sigma']}m, weight={comp['weight']}",
            file=f
        )

    print(f"\nNematode density range:", file=f)
    print(f"    min: {nematode_density.min():.2f} per 100cm^3", file=f)
    print(f"    max: {nematode_density.max():.2f} per 100cm^3", file=f)
    print(f"    mean: {nematode_density.mean():.2f} per 100cm^3", file=f)
    print(f"    standard dev: {nematode_density.std():.2f} per 100cm^3", file=f)

    print(f"\nSampling strategy:", file=f)
    print(f"    number of sampling locations: {n_samples}", file=f)
    print(
        f"    sampling density: {(n_samples / (GRID_SIZE ** 2)) * 100:.2f}%",
        file=f
    )
    print(
        f"    sample values range: "
        f"[{sample_values.min():.2f}, {sample_values.max():.2f}]",
        file=f
    )

    print(f"\nRandom seed: 42", file=f)
