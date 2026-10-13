import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator

# mode 2: risk-based prediction (no prior nematode data)
# this mode generates a probability surface for likely nematode introduction
# points using field characteristics instead of test results

print("=" * 70)
print("MODE 2: RISK-BASED NEMATODE PREDICTION")
print("No prior soil test data required")
print("=" * 70)

use_defaults = input("\nUse demo defaults? (yes/no): ").strip().lower()

if use_defaults in ['yes', 'y', 'default']:
    field_w = 316          # ~10 hectares (316m × 316m)
    field_h = 316
    entrance_x = 0         # gate at left edge, midway up
    entrance_y = 158
    rotation_years = 4     # years of continuous potato
    neighbour_infested = 'east'  # which boundary has infested neighbour
    soil_type = 'loam'     # sandy, loam, or clay
    years_since_test = 5   # years since last soil test
    uk_baseline = 0.65     # 65% of uk ware land infested (adas)
    
    print("\n✓ Using demo defaults")
else:
    print("\n--- FIELD DIMENSIONS ---")
    hectares = float(input("Field size (hectares, e.g., 5, 10, 20): ") or "10")
    field_w = int(np.sqrt(hectares * 10000))
    field_h = field_w
    
    print("\n--- FIELD ENTRANCE ---")
    print("Where is the main gate/entrance?")
    print("  Options: 'north', 'south', 'east', 'west', or exact coordinates 'x,y'")
    entrance_input = input("Entrance location: ").strip().lower()
    
    if entrance_input == 'north':
        entrance_x, entrance_y = field_w // 2, field_h
    elif entrance_input == 'south':
        entrance_x, entrance_y = field_w // 2, 0
    elif entrance_input == 'east':
        entrance_x, entrance_y = field_w, field_h // 2
    elif entrance_input == 'west':
        entrance_x, entrance_y = 0, field_h // 2
    elif ',' in entrance_input:
        parts = entrance_input.split(',')
        entrance_x, entrance_y = int(parts[0]), int(parts[1])
    else:
        entrance_x, entrance_y = 0, field_h // 2
    
    print("\n--- CROPPING HISTORY ---")
    rotation_years = int(input("Years of continuous potato cropping (e.g., 2, 4, 6): ") or "4")
    
    print("\n--- NEIGHBOURING FIELDS ---")
    print("Are any neighbouring fields known to be infested?")
    print("  Options: 'none', 'north', 'south', 'east', 'west', 'multiple'")
    neighbour_infested = input("Infested neighbour: ").strip().lower()
    
    print("\n--- SOIL TYPE ---")
    print("  Options: 'sandy', 'loam', 'clay'")
    soil_type = input("Soil type: ").strip().lower() or "loam"
    
    print("\n--- TESTING HISTORY ---")
    years_since_test = int(input("Years since last soil test (0 if never tested): ") or "5")
    
    uk_baseline = 0.65

# risk factor calculations
grid_res = 400
mx = np.linspace(0, field_w, grid_res)
my = np.linspace(0, field_h, grid_res)
MX, MY = np.meshgrid(mx, my)

# factor 1: entrance risk (gaussian decay from gate)
# equipment contamination is highest at entrance, decays with distance
entrance_sigma = field_w * 0.25  # influence radius = 25% of field width
entrance_risk = np.exp(-((MX - entrance_x)**2 + (MY - entrance_y)**2) / (2 * entrance_sigma**2))

# factor 2: neighbour risk (gaussian decay from shared boundary)
neighbour_risk = np.zeros_like(MX)
neighbour_sigma = field_w * 0.15  # influence radius = 15% of field width

if neighbour_infested == 'north' or neighbour_infested == 'multiple':
    neighbour_risk += np.exp(-((MY - field_h)**2) / (2 * neighbour_sigma**2))
if neighbour_infested == 'south' or neighbour_infested == 'multiple':
    neighbour_risk += np.exp(-((MY - 0)**2) / (2 * neighbour_sigma**2))
if neighbour_infested == 'east' or neighbour_infested == 'multiple':
    neighbour_risk += np.exp(-((MX - field_w)**2) / (2 * neighbour_sigma**2))
if neighbour_infested == 'west' or neighbour_infested == 'multiple':
    neighbour_risk += np.exp(-((MX - 0)**2) / (2 * neighbour_sigma**2))

if neighbour_infested != 'none':
    neighbour_risk = neighbour_risk / neighbour_risk.max()

# factor 3: rotation multiplier
# more continuous potato years = higher overall risk
# 1 year = 1.0x, 3 years = 1.5x, 5+ years = 2.0x
rotation_multiplier = min(1.0 + (rotation_years - 1) * 0.25, 2.0)

# factor 4: soil type modifier
# sandy soil = faster movement, higher risk spread
# clay soil = slower movement, more contained
soil_modifiers = {'sandy': 1.3, 'loam': 1.0, 'clay': 0.7}
soil_modifier = soil_modifiers.get(soil_type, 1.0)

# factor 5: testing gap uncertainty
# longer since last test = more uncertainty = wider risk zones
# 0 years = 1.0x, 5 years = 1.5x, 10+ years = 2.0x
test_gap_modifier = min(1.0 + (years_since_test * 0.1), 2.0)

# factor 6: uk baseline prior
# 65% of uk ware potato land is infested (adas, 2022)
baseline_prior = uk_baseline

# combined risk surface
# weight for each factor
weights = {
    'entrance': 0.35,      # equipment contamination is primary vector
    'neighbour': 0.30,     # cross field spread is significant
    'rotation': 0.15,      # cropping history (how often crops were alternated between)
    'soil': 0.10,          # soil type effect
    'test_gap': 0.10       # uncertainty from last test
}

# combine spatial risks; baseline is applied later as a multiplier
combined_risk = (
    weights['entrance'] * entrance_risk +
    weights['neighbour'] * neighbour_risk
)

# scale risk using modifiers (baseline is a multiplier, not a minimum)
combined_risk *= rotation_multiplier * soil_modifier * test_gap_modifier * uk_baseline

# normalise to 0-1 probability range
if combined_risk.max() > 0:
    combined_risk = combined_risk / combined_risk.max()

# priority zone classification
priority_map = np.zeros_like(combined_risk)
priority_map[combined_risk > 0.6] = 3    # critical - test immediately
priority_map[(combined_risk > 0.3) & (combined_risk <= 0.6)] = 2  # high - test soon
priority_map[(combined_risk > 0.1) & (combined_risk <= 0.3)] = 1  # moderate - monitor
# below 0.1 = low risk, left as 0

# generate recommended sampling points
num_samples = 30  # typical initial survey

# weight sample placement by risk probability
flat_risk = combined_risk.ravel()
flat_risk_norm = flat_risk / flat_risk.sum()

# sample indices weighted by risk
sample_indices = np.random.choice(len(flat_risk), size=num_samples, replace=False, p=flat_risk_norm)
sample_rows = sample_indices // grid_res
sample_cols = sample_indices % grid_res
sample_x = mx[sample_cols]
sample_y = my[sample_rows]

# visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# plot 1: risk probability surface
ax1 = axes[0]
c1 = ax1.contourf(MX, MY, combined_risk, levels=20, cmap='RdYlGn_r')
ax1.plot(entrance_x, entrance_y, 's', color='blue', markersize=12, label='Entrance', zorder=20)
ax1.set_title('Risk Probability Surface', fontweight='bold', fontsize=13)
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Distance (m)')
ax1.set_aspect('equal')
ax1.legend(loc='upper right')
fig.colorbar(c1, ax=ax1, label='Risk Probability (0-1)')

# plot 2: priority zones
ax2 = axes[1]
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # green, yellow, orange, red
from matplotlib.colors import ListedColormap
priority_cmap = ListedColormap(colors)
c2 = ax2.contourf(MX, MY, priority_map, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=priority_cmap)
ax2.plot(entrance_x, entrance_y, 's', color='blue', markersize=12, zorder=20)
ax2.set_title('Priority Testing Zones', fontweight='bold', fontsize=13)
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Distance (m)')
ax2.set_aspect('equal')

cbar2 = fig.colorbar(c2, ax=ax2, ticks=[0, 1, 2, 3])
cbar2.ax.set_yticklabels(['Low', 'Moderate', 'High', 'Critical'])

# plot 3: recommended sample points
ax3 = axes[2]
ax3.contourf(MX, MY, combined_risk, levels=20, cmap='RdYlGn_r', alpha=0.3)
ax3.scatter(sample_x, sample_y, c='red', s=60, marker='o', edgecolors='black',
            linewidths=0.5, zorder=15, label=f'{num_samples} Recommended Samples')
ax3.plot(entrance_x, entrance_y, 's', color='blue', markersize=12, zorder=20)
ax3.set_title('Recommended Initial Sampling', fontweight='bold', fontsize=13)
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('Distance (m)')
ax3.set_aspect('equal')
ax3.legend(loc='upper right')

plt.suptitle('Figure 2: Risk Surface Assessment', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print(f"\n{'=' * 70}")
print("MODE 2: RISK ANALYSIS REPORT")
print(f"{'=' * 70}")

print(f"\n--- FIELD CHARACTERISTICS ---")
print(f"  Field size: {field_w}m × {field_h}m ({field_w * field_h / 10000:.1f} hectares)")
print(f"  Entrance: ({entrance_x}m, {entrance_y}m)")
print(f"  Soil type: {soil_type} (modifier: {soil_modifier}×)")
print(f"  Rotation: {rotation_years} years continuous potato (modifier: {rotation_multiplier}×)")
print(f"  Neighbour infested: {neighbour_infested}")
print(f"  Years since last test: {years_since_test} (uncertainty modifier: {test_gap_modifier}×)")
print(f"  UK baseline probability: {uk_baseline * 100}%")

print(f"\n--- RISK FACTOR WEIGHTS ---")
for factor, weight in weights.items():
    print(f"  {factor}: {weight * 100}%")

print(f"\n--- RISK ZONE BREAKDOWN ---")
total_cells = combined_risk.size
critical_pct = (priority_map == 3).sum() / total_cells * 100
high_pct = (priority_map == 2).sum() / total_cells * 100
moderate_pct = (priority_map == 1).sum() / total_cells * 100
low_pct = (priority_map == 0).sum() / total_cells * 100

print(f"  CRITICAL (test immediately): {critical_pct:.1f}% of field")
print(f"  HIGH (test soon):            {high_pct:.1f}% of field")
print(f"  MODERATE (monitor):          {moderate_pct:.1f}% of field")
print(f"  LOW (safe for now):          {low_pct:.1f}% of field")

print(f"\n--- RECOMMENDED SAMPLING PLAN ---")
print(f"  Total samples recommended: {num_samples}")
critical_samples = sum(1 for i in range(num_samples) if priority_map[sample_rows[i], sample_cols[i]] == 3)
high_samples = sum(1 for i in range(num_samples) if priority_map[sample_rows[i], sample_cols[i]] == 2)
moderate_samples = sum(1 for i in range(num_samples) if priority_map[sample_rows[i], sample_cols[i]] == 1)
low_samples = sum(1 for i in range(num_samples) if priority_map[sample_rows[i], sample_cols[i]] == 0)

print(f"  In CRITICAL zones: {critical_samples}")
print(f"  In HIGH zones:     {high_samples}")
print(f"  In MODERATE zones: {moderate_samples}")
print(f"  In LOW zones:      {low_samples}")

print(f"\n--- NEXT STEPS ---")
print(f"  1. Take {num_samples} samples at recommended locations")
print(f"  2. If hotspots found → switch to Mode 1 (expansion prediction)")
print(f"  3. If no hotspots → re-test in {max(1, 3 - years_since_test)} years")
print(f"  4. Consider resistant varieties if rotation > 4 years")

print(f"\n{'=' * 70}")
print("END OF MODE 2 REPORT")
print(f"{'=' * 70}")
