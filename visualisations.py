import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# black and white compatible visualizations
# generates report-ready figures that work when printed in greyscale
# uses contour lines hatching and pattern fills instead of colour alone

# setup: reuse your existing simulation engine
np.random.seed(42)
grid_res = 350
field_dim = 223  # ~5 hectares per field
cols, rows = 5, 2
num_fields = 10
master_w, master_h = cols * field_dim, rows * field_dim
K = 5000
lam = 1.5
expansion_rate = 0.75
growth_rate = 0.4
years = 10
year_steps = list(range(0, years + 1, 2))

mx = np.linspace(0, master_w, grid_res)
my = np.linspace(0, master_h, grid_res)
MX, MY = np.meshgrid(mx, my)

class Hotspot:
    def __init__(self, cx, cy, initial_sigma=3.0):
        self.cx, self.cy = cx, cy
        self.sigma_initial = initial_sigma
        self.init_age = np.random.uniform(0, 5)

    def render(self, X, Y, year):
        effective_year = year + self.init_age
        sigma = self.sigma_initial + (expansion_rate * effective_year)
        d = K / (1 + np.exp(-growth_rate * (effective_year - 2)))
        return d * np.exp(-((X - self.cx)**2 + (Y - self.cy)**2) / (2 * sigma**2))

# generate fields
master_history = {yr: np.zeros((grid_res, grid_res)) for yr in year_steps}
for f_idx in range(num_fields):
    r, c = divmod(f_idx, cols)
    x_off, y_off = c * field_dim, (rows - 1 - r) * field_dim
    n_hotspots = max(1, np.random.poisson(lam))
    for _ in range(n_hotspots):
        cx = x_off + np.random.uniform(20, field_dim - 20)
        cy = y_off + np.random.uniform(20, field_dim - 20)
        hs = Hotspot(cx, cy, np.random.uniform(2, 5))
        for yr in year_steps:
            master_history[yr] += hs.render(MX, MY, yr)

for yr in year_steps:
    master_history[yr] = np.clip(master_history[yr], 0, K)

def draw_field_grid(ax):
    for f_idx in range(num_fields):
        r, c = divmod(f_idx, cols)
        ax.add_patch(patches.Rectangle(
            (c * field_dim, (rows - 1 - r) * field_dim),
            field_dim, field_dim, linewidth=1.5,
            edgecolor='black', linestyle='--', facecolor='none', zorder=10))

# figure 1: b&w density progression (report-safe)
bw_cmap = LinearSegmentedColormap.from_list('bw', ['white', '#cccccc', '#888888', '#444444', 'black'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, yr in enumerate(year_steps[:6]):
    r_idx, c_idx = divmod(i, 3)
    ax = axes[r_idx, c_idx]
    data = master_history[yr]

    # filled contour in greyscale
    cf = ax.contourf(MX, MY, data, levels=[0, 500, 1500, 2500, 3500, 5000],
                     cmap=bw_cmap)

    # contour lines with labels (readable in b&w)
    cs = ax.contour(MX, MY, data, levels=[500, 1500, 2500, 3500],
                    colors='black', linewidths=0.8, linestyles=['-', '--', '-.', ':'])
    ax.clabel(cs, fmt='%d', fontsize=7)

    draw_field_grid(ax)
    ax.set_title(f'Year {yr}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.set_aspect('equal')

# colorbar
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=bw_cmap, norm=plt.Normalize(0, K))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Cyst Density (per hectare)', fontweight='bold')

plt.suptitle('Figure A1: Nematode Density Progression',
             fontsize=14, fontweight='bold', y=0.98)
fig.subplots_adjust(right=0.90, hspace=0.3, wspace=0.3)
plt.show()

# figure 2: b&w priority zones with hatching
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, yr in enumerate(year_steps[:6]):
    r_idx, c_idx = divmod(i, 3)
    ax = axes[r_idx, c_idx]
    data = master_history[yr]

    # create priority zones
    priority = np.zeros_like(data)
    priority[(data > 100) & (data < 2500)] = 2  # frontier
    priority[data >= 2500] = 1                    # established

    # plot each zone with different hatching patterns
    # clean areas (no fill)
    ax.contourf(MX, MY, (priority == 0).astype(float), levels=[0.5, 1.5],
                colors=['white'], alpha=0.3)

    # established hotspots (dense dots)
    ax.contourf(MX, MY, (priority == 1).astype(float), levels=[0.5, 1.5],
                colors=['#aaaaaa'], hatches=['...'])

    # frontier zones (diagonal lines)
    ax.contourf(MX, MY, (priority == 2).astype(float), levels=[0.5, 1.5],
                colors=['#dddddd'], hatches=['///'])

    # contour boundaries
    ax.contour(MX, MY, data, levels=[100, 2500], colors='black',
               linewidths=1.2, linestyles=['--', '-'])

    draw_field_grid(ax)
    ax.set_title(f'Year {yr}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.set_aspect('equal')

# manual legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='Clean (Low Priority)'),
    Patch(facecolor='#dddddd', edgecolor='black', hatch='///', label='Frontier (HIGH Priority)'),
    Patch(facecolor='#aaaaaa', edgecolor='black', hatch='...', label='Established (Medium Priority)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 0))

plt.suptitle('Figure A2: Priority Sampling Zones',
             fontsize=14, fontweight='bold', y=0.98)
fig.subplots_adjust(right=0.95, hspace=0.3, wspace=0.3, bottom=0.08)
plt.show()

# figure 3: b&w strategy comparison
truth = master_history[years]

# grid sampling
side = int(np.sqrt(250))
gx = np.linspace(0, master_w, side)
gy = np.linspace(0, master_h, side)
GX_s, GY_s = np.meshgrid(gx, gy)
grid_pts = np.column_stack([GX_s.ravel(), GY_s.ravel()])

# targeted sampling
y_i, x_i = np.where((truth > 100) & (truth < 2500))
if len(x_i) > 250:
    s_i = np.random.choice(len(x_i), 250, replace=False)
    target_pts = np.column_stack([mx[x_i[s_i]], my[y_i[s_i]]])
else:
    target_pts = np.column_stack([mx[x_i], my[y_i]])

def count_hits(pts):
    ix = (np.clip(pts[:, 0], 0, master_w - 1) / master_w * (grid_res - 1)).astype(int)
    iy = (np.clip(pts[:, 1], 0, master_h - 1) / master_h * (grid_res - 1)).astype(int)
    return np.sum(truth[iy, ix] > 500)

g_hits = count_hits(grid_pts)
t_hits = count_hits(target_pts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

for ax, pts, hits, title, marker in zip(
        [ax1, ax2], [grid_pts, target_pts], [g_hits, t_hits],
        ['Traditional Grid', 'Targeted Strategy'],
        ['x', 'o']):

    # background density in greyscale
    ax.contourf(MX, MY, truth, levels=[0, 500, 1500, 2500, 3500, 5000],
                cmap=bw_cmap, alpha=0.3)
    ax.contour(MX, MY, truth, levels=[500, 2500], colors='black',
               linewidths=0.5, linestyles='--')

    # sample points
    ax.scatter(pts[:, 0], pts[:, 1], c='black', marker=marker, s=30,
               alpha=0.7, zorder=15)

    draw_field_grid(ax)
    ax.set_title(f'{title}: {hits} Detections', fontweight='bold', fontsize=13)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.set_xlim(0, master_w)
    ax.set_ylim(0, master_h)
    ax.set_aspect('equal')

improvement = ((t_hits - g_hits) / max(g_hits, 1)) * 100
plt.suptitle(f'Figure 1: Sampling Strategy Comparison (+{improvement:.1f}% improvement)',
             fontsize=14, fontweight='bold', y=0.98)
fig.subplots_adjust(wspace=0.2)
plt.show()

print(f"\nGrid detections: {g_hits}")
print(f"Targeted detections: {t_hits}")
print(f"Improvement: +{improvement:.1f}%")
