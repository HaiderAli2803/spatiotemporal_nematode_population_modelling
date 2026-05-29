import numpy as np
import matplotlib.pyplot as plt

# sampling efficiency curve
# shows detection rate vs number of samples for both strategies
# answers how many samples do i actually need

np.random.seed(42)

# setup: same simulation as main model
grid_res = 250
field_dim = 223
cols, rows = 5, 2
num_fields = 10
master_w, master_h = cols * field_dim, rows * field_dim
K = 5000
expansion_rate = 0.75
growth_rate = 0.4
years = 10

mx = np.linspace(0, master_w, grid_res)
my = np.linspace(0, master_h, grid_res)
MX, MY = np.meshgrid(mx, my)

class Hotspot:
    def __init__(self, cx, cy, sigma, init_age):
        self.cx, self.cy = cx, cy
        self.sigma_initial = sigma
        self.init_age = init_age

    def render(self, X, Y, year):
        ey = year + self.init_age
        sigma = self.sigma_initial + (expansion_rate * ey)
        d = K / (1 + np.exp(-growth_rate * (ey - 2)))
        return d * np.exp(-((X - self.cx)**2 + (Y - self.cy)**2) / (2 * sigma**2))

# generate field
truth = np.zeros((grid_res, grid_res))
for f_idx in range(num_fields):
    r, c = divmod(f_idx, cols)
    x_off, y_off = c * field_dim, (rows - 1 - r) * field_dim
    n_hotspots = max(1, np.random.poisson(1.5))
    for _ in range(n_hotspots):
        cx = x_off + np.random.uniform(20, field_dim - 20)
        cy = y_off + np.random.uniform(20, field_dim - 20)
        hs = Hotspot(cx, cy, np.random.uniform(2, 5), np.random.uniform(0, 5))
        truth += hs.render(MX, MY, years)

truth = np.clip(truth, 0, K)

# total number of detectable cells (ground truth)
total_detectable = (truth > 500).sum()

# detection functions
def grid_detections(n_samples):
    # count detections using uniform grid sampling
    side = max(2, int(np.sqrt(n_samples)))
    gx = np.linspace(0, master_w, side)
    gy = np.linspace(0, master_h, side)
    GX_s, GY_s = np.meshgrid(gx, gy)
    pts = np.column_stack([GX_s.ravel(), GY_s.ravel()])[:n_samples]
    
    ix = (np.clip(pts[:, 0], 0, master_w - 1) / master_w * (grid_res - 1)).astype(int)
    iy = (np.clip(pts[:, 1], 0, master_h - 1) / master_h * (grid_res - 1)).astype(int)
    return int(np.sum(truth[iy, ix] > 500))

def targeted_detections(n_samples):
    # count detections using frontier-targeted sampling
    y_i, x_i = np.where((truth > 100) & (truth < 2500))
    
    if len(x_i) == 0:
        return 0
    
    actual_samples = min(n_samples, len(x_i))
    s_i = np.random.choice(len(x_i), actual_samples, replace=False)
    pts = np.column_stack([mx[x_i[s_i]], my[y_i[s_i]]])
    
    ix = (np.clip(pts[:, 0], 0, master_w - 1) / master_w * (grid_res - 1)).astype(int)
    iy = (np.clip(pts[:, 1], 0, master_h - 1) / master_h * (grid_res - 1)).astype(int)
    return int(np.sum(truth[iy, ix] > 500))

# run efficiency analysis
sample_counts = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
n_runs = 10  # average over multiple runs for stability

grid_results = []
target_results = []

print("Running efficiency analysis...")
for n in sample_counts:
    g_runs = []
    t_runs = []
    for _ in range(n_runs):
        g_runs.append(grid_detections(n))
        t_runs.append(targeted_detections(n))
    grid_results.append(np.mean(g_runs))
    target_results.append(np.mean(t_runs))
    print(f"  {n} samples: Grid={np.mean(g_runs):.0f}, Targeted={np.mean(t_runs):.0f}")

# visualization 1: detection count vs sample count
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(sample_counts, grid_results, 'o-', color='#3498db', linewidth=2.5,
         markersize=8, label='Traditional Grid')
ax1.plot(sample_counts, target_results, 's-', color='#e74c3c', linewidth=2.5,
         markersize=8, label='Targeted (Frontier)')
ax1.fill_between(sample_counts, grid_results, target_results, alpha=0.15, color='green',
                 label='Detection advantage')

ax1.set_xlabel('Number of Samples', fontweight='bold', fontsize=12)
ax1.set_ylabel('Hotspots Detected', fontweight='bold', fontsize=12)
ax1.set_title('Detection Count vs Sample Budget', fontweight='bold', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# visualization 2: efficiency ratio
efficiency_ratio = [t / max(g, 1) for t, g in zip(target_results, grid_results)]

ax2.bar([str(n) for n in sample_counts], efficiency_ratio, color='#2ecc71', edgecolor='black')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Equal performance')

for i, (n, ratio) in enumerate(zip(sample_counts, efficiency_ratio)):
    ax2.text(i, ratio + 0.1, f'{ratio:.1f}×', ha='center', fontweight='bold', fontsize=9)

ax2.set_xlabel('Number of Samples', fontweight='bold', fontsize=12)
ax2.set_ylabel('Targeted / Grid Detection Ratio', fontweight='bold', fontsize=12)
ax2.set_title('Efficiency Multiplier (Targeted vs Grid)', fontweight='bold', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure A4: Sampling Efficiency Analysis', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.show()

# visualization 3: cost-effectiveness (assuming £25 per sample)
cost_per_sample = 25  # approximate uk lab cost

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

grid_cost_per_detection = [(n * cost_per_sample) / max(d, 1) for n, d in zip(sample_counts, grid_results)]
target_cost_per_detection = [(n * cost_per_sample) / max(d, 1) for n, d in zip(sample_counts, target_results)]

ax.plot(sample_counts, grid_cost_per_detection, 'o-', color='#3498db', linewidth=2.5,
        markersize=8, label='Traditional Grid')
ax.plot(sample_counts, target_cost_per_detection, 's-', color='#e74c3c', linewidth=2.5,
        markersize=8, label='Targeted (Frontier)')

ax.set_xlabel('Number of Samples', fontweight='bold', fontsize=12)
ax.set_ylabel('Cost per Detection (£)', fontweight='bold', fontsize=12)
ax.set_title('Cost-Effectiveness: £ per Hotspot Detected', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle('Figure A5: Cost Analysis (assuming £25 per sample)',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# summary table
print(f"\n{'=' * 80}")
print("SAMPLING EFFICIENCY SUMMARY")
print(f"{'=' * 80}")
print(f"\n{'Samples':<10} {'Grid Hits':<12} {'Target Hits':<14} {'Ratio':<10} {'Grid £/det':<12} {'Target £/det':<12}")
print("-" * 70)

for i, n in enumerate(sample_counts):
    g = grid_results[i]
    t = target_results[i]
    ratio = t / max(g, 1)
    g_cost = (n * cost_per_sample) / max(g, 1)
    t_cost = (n * cost_per_sample) / max(t, 1)
    print(f"{n:<10} {g:<12.0f} {t:<14.0f} {ratio:<10.1f}× £{g_cost:<11.0f} £{t_cost:<11.0f}")

# find optimal sample count (where targeted detection starts to plateau)
for i in range(1, len(target_results)):
    marginal_gain = (target_results[i] - target_results[i-1]) / (sample_counts[i] - sample_counts[i-1])
    if marginal_gain < 0.1:
        print(f"\n→ Optimal sample count (targeted): ~{sample_counts[i]} samples")
        print(f"  Beyond this point, additional samples yield diminishing returns")
        break

print(f"\n{'=' * 80}")
