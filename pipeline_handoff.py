import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# mode 2 to mode 1 handoff demonstration
# shows the full pipeline:
# 1 mode 2: risk map generated (no prior data)
# 2 farmer takes samples in priority zones
# 3 hotspots discovered
# 4 mode 1: expansion prediction from discovered hotspots
# 5 future sampling guided by expansion model

np.random.seed(123)

# field setup
field_dim = 316  # ~10 hectares
grid_res = 300
K = 5000
expansion_rate = 0.75
growth_rate = 0.4

mx = np.linspace(0, field_dim, grid_res)
my = np.linspace(0, field_dim, grid_res)
MX, MY = np.meshgrid(mx, my)

# stage 1: mode 2 - risk assessment (no nematode data)
# field characteristics
entrance_x, entrance_y = 0, field_dim // 2  # gate on west side
neighbour_side = 'east'  # infested neighbour to the east

# build risk surface
entrance_sigma = field_dim * 0.25
entrance_risk = np.exp(-((MX - entrance_x)**2 + (MY - entrance_y)**2) / (2 * entrance_sigma**2))

neighbour_sigma = field_dim * 0.15
neighbour_risk = np.exp(-((MX - field_dim)**2) / (2 * neighbour_sigma**2))

uk_baseline = 0.65
rotation_multiplier = 1.5  # 3 years continuous potato

# combined risk (no baseline floor - used as scalar)
combined_risk = (0.35 * entrance_risk + 0.30 * neighbour_risk)
combined_risk *= rotation_multiplier * uk_baseline
if combined_risk.max() > 0:
    combined_risk = combined_risk / combined_risk.max()

# generate recommended initial samples (weighted by risk)
num_initial_samples = 30
flat_risk = combined_risk.ravel()
flat_risk_norm = flat_risk / flat_risk.sum()
sample_indices = np.random.choice(len(flat_risk), size=num_initial_samples, replace=False, p=flat_risk_norm)
sample_rows = sample_indices // grid_res
sample_cols = sample_indices % grid_res
sample_x = mx[sample_cols]
sample_y = my[sample_rows]

# stage 2: simulated discovery (farmer takes samples, finds hotspots)
# hidden truth: there are hotspots in this field (farmer doesn't know yet)
# placed where mode 2 predicts high risk: near entrance and near infested neighbour
true_hotspots = [
    {'cx': 25, 'cy': 158, 'sigma': 25},    # directly at entrance - large established hotspot
    {'cx': 290, 'cy': 200, 'sigma': 20},    # near infested neighbour (east) - large
    {'cx': 300, 'cy': 100, 'sigma': 15},    # secondary spot near east boundary
]

# check which samples hit a hotspot (threshold: >500 cysts/ha at that point)
true_density = np.zeros((grid_res, grid_res))
for hs in true_hotspots:
    d = K * 0.8  # strong established hotspot (80% of carrying capacity)
    true_density += d * np.exp(-((MX - hs['cx'])**2 + (MY - hs['cy'])**2) / (2 * hs['sigma']**2))
true_density = np.clip(true_density, 0, K)

# determine which samples detected something
detections = []
for i in range(num_initial_samples):
    ix = int(np.clip(sample_x[i] / field_dim * (grid_res - 1), 0, grid_res - 1))
    iy = int(np.clip(sample_y[i] / field_dim * (grid_res - 1), 0, grid_res - 1))
    density_at_point = true_density[iy, ix]
    detected = density_at_point > 500
    detections.append({
        'x': sample_x[i], 'y': sample_y[i],
        'density': density_at_point, 'detected': detected
    })

detected_pts = [d for d in detections if d['detected']]
clean_pts = [d for d in detections if not d['detected']]

# stage 3: mode 1 - expansion prediction (from discovered hotspots)
# use detected points to estimate hotspot centers
# simple: cluster detected points to find centers
discovered_centers = []
if detected_pts:
    pts = np.array([[d['x'], d['y']] for d in detected_pts])
    # simple clustering: group points within 30m of each other
    used = set()
    for i, p in enumerate(pts):
        if i in used:
            continue
        cluster = [p]
        for j, q in enumerate(pts):
            if j != i and j not in used and np.linalg.norm(p - q) < 30:
                cluster.append(q)
                used.add(j)
        used.add(i)
        center = np.mean(cluster, axis=0)
        discovered_centers.append({'cx': center[0], 'cy': center[1], 'sigma': 4})

# predict expansion over 5 years from discovered centers
prediction_years = [0, 1, 2, 3, 5]
predictions = {}
for yr in prediction_years:
    pred = np.zeros((grid_res, grid_res))
    for hs in discovered_centers:
        sigma = hs['sigma'] + (expansion_rate * yr)
        d = K / (1 + np.exp(-growth_rate * (yr - 2)))
        pred += d * np.exp(-((MX - hs['cx'])**2 + (MY - hs['cy'])**2) / (2 * sigma**2))
    predictions[yr] = np.clip(pred, 0, K)

# visualization: 4-panel pipeline
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# panel 1: mode 2 risk map
ax1 = axes[0, 0]
ax1.contourf(MX, MY, combined_risk, levels=20, cmap='RdYlGn_r')
ax1.plot(entrance_x, entrance_y, 's', color='blue', markersize=12, zorder=20)
ax1.set_title('Stage 1: Mode 2 Risk Assessment\n(No nematode data)', fontweight='bold')
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Distance (m)')
ax1.set_aspect('equal')
ax1.annotate('Gate', (entrance_x + 5, entrance_y), fontweight='bold', color='blue')

# panel 2: initial sampling results
ax2 = axes[0, 1]
ax2.contourf(MX, MY, combined_risk, levels=20, cmap='RdYlGn_r', alpha=0.3)

# plot clean samples
for d in clean_pts:
    ax2.scatter(d['x'], d['y'], c='grey', marker='x', s=50, zorder=15)

# plot detected samples
for d in detected_pts:
    ax2.scatter(d['x'], d['y'], c='red', marker='o', s=80, edgecolors='black',
                linewidths=1.5, zorder=15)

ax2.plot(entrance_x, entrance_y, 's', color='blue', markersize=12, zorder=20)
ax2.set_title(f'Stage 2: Initial Sampling Results\n({len(detected_pts)} detections from {num_initial_samples} samples)',
              fontweight='bold')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Distance (m)')
ax2.set_aspect('equal')

# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
           markeredgecolor='black', label='Detected'),
    Line2D([0], [0], marker='x', color='grey', linestyle='None', markersize=10,
           label='Clean')
]
ax2.legend(handles=legend_elements, loc='upper right')

# panel 3: mode 1 expansion prediction (year 0 vs year 5)
ax3 = axes[1, 0]
# year 0 (current)
ax3.contourf(MX, MY, predictions[0], levels=20, cmap='YlOrRd', vmin=0, vmax=K, alpha=0.5)
# year 5 prediction overlay (contour lines)
cs = ax3.contour(MX, MY, predictions[5], levels=[100, 500, 1500, 3000],
                 colors='black', linewidths=1.5, linestyles=['-', '--', '-.', ':'])
ax3.clabel(cs, fmt='%d', fontsize=8)

# mark discovered centers
for hs in discovered_centers:
    ax3.plot(hs['cx'], hs['cy'], 'r*', markersize=15, zorder=20)

ax3.set_title('Stage 3: Mode 1 Expansion Prediction\n(Current density + Year 5 contours)',
              fontweight='bold')
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('Distance (m)')
ax3.set_aspect('equal')

# panel 4: recommended follow-up sampling
ax4 = axes[1, 1]

# priority zones from expansion prediction
pred_yr5 = predictions[5]
priority = np.zeros_like(pred_yr5)
priority[(pred_yr5 > 100) & (pred_yr5 < 2500)] = 2  # frontier = high
priority[pred_yr5 >= 2500] = 1  # established = medium

colors = ['#d4edda', '#ffc107', '#dc3545']  # green, yellow, red
priority_cmap = ListedColormap(colors)
ax4.contourf(MX, MY, priority, levels=[-0.5, 0.5, 1.5, 2.5], cmap=priority_cmap)

# generate follow-up sample points in frontier zones
frontier_mask = (pred_yr5 > 100) & (pred_yr5 < 2500)
fy, fx = np.where(frontier_mask)
if len(fx) > 20:
    follow_idx = np.random.choice(len(fx), 20, replace=False)
    follow_x = mx[fx[follow_idx]]
    follow_y = my[fy[follow_idx]]
    ax4.scatter(follow_x, follow_y, c='black', marker='^', s=60,
                zorder=15, label='Follow-up samples (n=20)')
    ax4.legend(loc='upper right')
else:
    ax4.text(field_dim/2, field_dim/2, 'No frontier zones detected\nField may be clean',
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax4.set_title('Stage 4: Recommended Follow-Up Sampling\n(Based on Year 5 expansion prediction)',
              fontweight='bold')
ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Distance (m)')
ax4.set_aspect('equal')

plt.suptitle('Figure 3: Risk Model -> Discovery -> Expansion Model -> Follow-Up', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print(f"\n{'=' * 70}")
print("FULL PIPELINE SUMMARY")
print(f"{'=' * 70}")

print(f"\n--- STAGE 1: MODE 2 RISK ASSESSMENT ---")
print(f"  Field: {field_dim}m × {field_dim}m ({field_dim**2/10000:.1f} hectares)")
print(f"  Entrance: ({entrance_x}m, {entrance_y}m)")
print(f"  Infested neighbour: {neighbour_side}")
print(f"  Risk zones generated: YES")

print(f"\n--- STAGE 2: INITIAL SAMPLING ---")
print(f"  Samples taken: {num_initial_samples}")
print(f"  Detections: {len(detected_pts)}")
print(f"  Detection rate: {len(detected_pts)/num_initial_samples*100:.1f}%")
print(f"  Hotspot clusters found: {len(discovered_centers)}")

print(f"\n--- STAGE 3: MODE 1 EXPANSION PREDICTION ---")
for hs in discovered_centers:
    yr5_sigma = 4 + (expansion_rate * 5)
    print(f"  Hotspot at ({hs['cx']:.0f}m, {hs['cy']:.0f}m)")
    print(f"    Current radius: ~{hs['sigma']:.0f}m")
    print(f"    Predicted Year 5 radius: ~{yr5_sigma:.1f}m")

print(f"\n--- STAGE 4: FOLLOW-UP RECOMMENDATIONS ---")
frontier_area = frontier_mask.sum() / priority.size * 100
print(f"  Frontier zone coverage: {frontier_area:.1f}% of field")
print(f"  Follow-up samples recommended: 20 (in frontier zones)")
print(f"  Next test date: 1-2 years from initial detection")

print(f"\n{'=' * 70}")
print("PIPELINE COMPLETE: Mode 2 → Mode 1 handoff demonstrated")
print(f"{'=' * 70}")
