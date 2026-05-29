import numpy as np
import matplotlib.pyplot as plt

# sensitivity analysis
# tests how changing key parameters affects model outputs
# shows assessors you understand what drives the model

grid_res = 200  # lower res for speed (running many simulations)
field_dim = 223
cols, rows = 5, 2
num_fields = 10
master_w, master_h = cols * field_dim, rows * field_dim
K = 5000
years = 10
year_steps = list(range(0, years + 1, 2))

mx = np.linspace(0, master_w, grid_res)
my = np.linspace(0, master_h, grid_res)
MX, MY = np.meshgrid(mx, my)

class Hotspot:
    def __init__(self, cx, cy, sigma, init_age, exp_rate, gr_rate, k):
        self.cx, self.cy = cx, cy
        self.sigma_initial = sigma
        self.init_age = init_age
        self.exp_rate = exp_rate
        self.gr_rate = gr_rate
        self.k = k

    def render(self, X, Y, year):
        ey = year + self.init_age
        sigma = self.sigma_initial + (self.exp_rate * ey)
        d = self.k / (1 + np.exp(-self.gr_rate * (ey - 2)))
        return d * np.exp(-((X - self.cx)**2 + (Y - self.cy)**2) / (2 * sigma**2))

def run_simulation(exp_rate, gr_rate, lam_val, k_val):
    # run full simulation with given parameters, return final density + stats
    np.random.seed(42)  # reproducible for comparison
    
    history = {yr: np.zeros((grid_res, grid_res)) for yr in year_steps}
    
    for f_idx in range(num_fields):
        r, c = divmod(f_idx, cols)
        x_off, y_off = c * field_dim, (rows - 1 - r) * field_dim
        n_hotspots = max(1, np.random.poisson(lam_val))
        
        for _ in range(n_hotspots):
            cx = x_off + np.random.uniform(20, field_dim - 20)
            cy = y_off + np.random.uniform(20, field_dim - 20)
            init_age = np.random.uniform(0, 5)
            sigma = np.random.uniform(2, 5)
            hs = Hotspot(cx, cy, sigma, init_age, exp_rate, gr_rate, k_val)
            for yr in year_steps:
                history[yr] += hs.render(MX, MY, yr)
    
    for yr in year_steps:
        history[yr] = np.clip(history[yr], 0, k_val)
    
    # calculate metrics for each year
    metrics = {}
    for yr in year_steps:
        data = history[yr]
        infested_pct = (data > 100).sum() / data.size * 100
        max_density = data.max()
        avg_infested = data[data > 100].mean() if (data > 100).any() else 0
        metrics[yr] = {
            'infested_pct': infested_pct,
            'max_density': max_density,
            'avg_density': avg_infested
        }
    
    return history, metrics

# test 1: expansion rate sensitivity
print("Running expansion rate sensitivity...")
exp_rates = [0.25, 0.5, 0.75, 1.0, 1.5]
exp_results = {}
for er in exp_rates:
    _, metrics = run_simulation(er, 0.4, 1.5, 5000)
    exp_results[er] = metrics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for er in exp_rates:
    years_list = list(exp_results[er].keys())
    infested = [exp_results[er][yr]['infested_pct'] for yr in years_list]
    ax1.plot(years_list, infested, 'o-', label=f'v = {er} m/yr', linewidth=2)

ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Infested Area (%)', fontweight='bold')
ax1.set_title('Effect of Expansion Rate on Infested Area', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# bar chart: year 10 comparison
year10_infested = [exp_results[er][10]['infested_pct'] for er in exp_rates]
bars = ax2.bar([str(er) for er in exp_rates], year10_infested, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'])
ax2.set_xlabel('Expansion Rate (m/yr)', fontweight='bold')
ax2.set_ylabel('Infested Area at Year 10 (%)', fontweight='bold')
ax2.set_title('Year 10 Infested Area by Expansion Rate', fontweight='bold')
for bar, val in zip(bars, year10_infested):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Figure A3a: Effect of Expansion Rate on Infested Area', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# test 2: growth rate sensitivity
print("Running growth rate sensitivity...")
gr_rates = [0.1, 0.2, 0.4, 0.6, 0.8]
gr_results = {}
for gr in gr_rates:
    _, metrics = run_simulation(0.75, gr, 1.5, 5000)
    gr_results[gr] = metrics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for gr in gr_rates:
    years_list = list(gr_results[gr].keys())
    max_d = [gr_results[gr][yr]['max_density'] for yr in years_list]
    ax1.plot(years_list, max_d, 'o-', label=f'r = {gr}/season', linewidth=2)

ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Max Cyst Density (per hectare)', fontweight='bold')
ax1.set_title('Effect of Growth Rate on Peak Density', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

year10_max = [gr_results[gr][10]['max_density'] for gr in gr_rates]
bars = ax2.bar([str(gr) for gr in gr_rates], year10_max, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'])
ax2.set_xlabel('Growth Rate (per season)', fontweight='bold')
ax2.set_ylabel('Max Density at Year 10', fontweight='bold')
ax2.set_title('Year 10 Peak Density by Growth Rate', fontweight='bold')
for bar, val in zip(bars, year10_max):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:.0f}', ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Figure A3b: Effect of Growth Rate on Peak Density', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# test 3: lambda (hotspot count) sensitivity
print("Running lambda sensitivity...")
lambdas = [0.5, 1.0, 1.5, 2.5, 4.0]
lam_results = {}
for l in lambdas:
    _, metrics = run_simulation(0.75, 0.4, l, 5000)
    lam_results[l] = metrics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for l in lambdas:
    years_list = list(lam_results[l].keys())
    infested = [lam_results[l][yr]['infested_pct'] for yr in years_list]
    ax1.plot(years_list, infested, 'o-', label=f'λ = {l}', linewidth=2)

ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Infested Area (%)', fontweight='bold')
ax1.set_title('Effect of Initial Hotspot Count on Spread', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

year10_infested = [lam_results[l][10]['infested_pct'] for l in lambdas]
bars = ax2.bar([str(l) for l in lambdas], year10_infested, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'])
ax2.set_xlabel('Poisson λ (avg hotspots per field)', fontweight='bold')
ax2.set_ylabel('Infested Area at Year 10 (%)', fontweight='bold')
ax2.set_title('Year 10 Infested Area by Lambda', fontweight='bold')
for bar, val in zip(bars, year10_infested):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Figure A3c: Effect of Initial Hotspot Count on Spread', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# test 4: soil type impact (via expansion rate modifier)
print("Running soil type sensitivity...")
soil_types = {'Sandy (1.3×)': 0.75 * 1.3, 'Loam (1.0×)': 0.75 * 1.0, 'Clay (0.7×)': 0.75 * 0.7}
soil_results = {}
for name, er in soil_types.items():
    _, metrics = run_simulation(er, 0.4, 1.5, 5000)
    soil_results[name] = metrics

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for name in soil_types:
    years_list = list(soil_results[name].keys())
    infested = [soil_results[name][yr]['infested_pct'] for yr in years_list]
    ax.plot(years_list, infested, 'o-', label=name, linewidth=2.5)

ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Infested Area (%)', fontweight='bold')
ax.set_title('Effect of Soil Type on Nematode Spread', fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.suptitle('Figure A3d: Effect of Soil Type on Nematode Spread', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# summary table
print(f"\n{'=' * 70}")
print("SENSITIVITY ANALYSIS SUMMARY")
print(f"{'=' * 70}")
print(f"\n{'Parameter':<25} {'Value':<15} {'Year 10 Infested %':<20} {'Year 10 Max Density':<20}")
print("-" * 80)

for er in exp_rates:
    print(f"{'Expansion Rate':<25} {er:<15} {exp_results[er][10]['infested_pct']:<20.1f} {exp_results[er][10]['max_density']:<20.0f}")

print()
for gr in gr_rates:
    print(f"{'Growth Rate':<25} {gr:<15} {gr_results[gr][10]['infested_pct']:<20.1f} {gr_results[gr][10]['max_density']:<20.0f}")

print()
for l in lambdas:
    print(f"{'Lambda':<25} {l:<15} {lam_results[l][10]['infested_pct']:<20.1f} {lam_results[l][10]['max_density']:<20.0f}")

print()
for name, er in soil_types.items():
    print(f"{'Soil Type':<25} {name:<15} {soil_results[name][10]['infested_pct']:<20.1f} {soil_results[name][10]['max_density']:<20.0f}")
