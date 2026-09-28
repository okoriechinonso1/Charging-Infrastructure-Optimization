'DP Algorithm Framework: Figure 5'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
import pandas as pd
import geopandas as gpd
import contextily as cx

os.makedirs('Figures', exist_ok=True)

"Dynamic Programming illustration: Figure 5"

P_VALS = list(range(6))
S_VALS = list(range(2, 9))
E_VALS = [6, 8, 10, 12]

n_p = len(P_VALS)
n_s = len(S_VALS)
n_e = len(E_VALS)

FLEET_COLORS = {
    0: '#FFA500',  # fleet 6  — orange
    1: '#228B22',  # fleet 8  — green
    2: '#1E90FF',  # fleet 10 — blue
    3: '#DC143C',  # fleet 12 — red
}

sp_p   = 4.0
sp_s   = 1.0
sp_e   = 1.0
angle  = np.radians(35)
v_p   = np.array([1.0,  0.0])
v_s   = np.array([0.0,  1.0])
v_e   = np.array([np.cos(angle), np.sin(angle)])

def proj(p, s, ei):
    si = s - S_VALS[0]
    return p*sp_p*v_p + si*sp_s*v_s + ei*sp_e*v_e

EDGE_LW  = 0.9
NODE_R   = 0.17
EDGE_COL = '#aaaaaa'

NODE_ORIGIN = proj(0, S_VALS[0], 0)
AXIS_OFFSET = np.array([-0.2, -0.4])
O = NODE_ORIGIN + AXIS_OFFSET - np.array([sp_p, 0.0])

P0_NODE = (0, S_VALS[0], 0)

OPT_PATH = [
    (0, S_VALS[0], 0),
    (1, S_VALS[0], 1),
    (2, S_VALS[2], 1),
    (3, S_VALS[3], 2),
    (4, S_VALS[3], 2),
    (5, S_VALS[5], 3),
]

OPT_SET = set(OPT_PATH)

# ── per-period "what changed" annotations ─────────────────────────────────
PERIOD_NOTES = {
    1: 'One new ET',
    2: '2 new\nstations',
    3: '2 new ETs and\n2 new stations',
    4: 'No change',
    5: '2 new ETs and\n3 new stations',
}

fig, ax = plt.subplots(figsize=(32, 20), facecolor='white')
ax.set_aspect('equal')
ax.axis('off')

def representative_targets(s0, ei0):
    targets = set()
    s_idx0 = S_VALS.index(s0)
    if ei0 + 1 < n_e:
        targets.add((s0, ei0 + 1))
    if s_idx0 + 1 < n_s:
        targets.add((S_VALS[s_idx0 + 1], ei0))
    if s_idx0 + 1 < n_s and ei0 + 1 < n_e:
        targets.add((S_VALS[s_idx0 + 1], ei0 + 1))
    targets.add((s0, ei0))
    if s_idx0 + 2 < n_s and s_idx0 % 2 == 0:
        targets.add((S_VALS[s_idx0 + 2], min(ei0 + 1, n_e - 1)))
    return targets

fan_segs = []
for p in range(n_p - 1):
    if p == 0:
        s0, ei0 = P0_NODE[1], P0_NODE[2]
        for (s1, ei1) in representative_targets(s0, ei0):
            fan_segs.append([proj(p, s0, ei0), proj(p+1, s1, ei1)])
        for s1 in [S_VALS[1], S_VALS[3], S_VALS[5]]:
            for ei1 in [0, n_e-1]:
                fan_segs.append([proj(p, s0, ei0), proj(p+1, s1, ei1)])
    else:
        for s0 in S_VALS:
            for ei0 in range(n_e):
                for (s1, ei1) in representative_targets(s0, ei0):
                    fan_segs.append([proj(p, s0, ei0), proj(p+1, s1, ei1)])

ax.add_collection(LineCollection(fan_segs,
    colors=EDGE_COL, linewidths=EDGE_LW, alpha=0.45, zorder=1))

for p in reversed(P_VALS):
    zb = (n_p - p) * 30

    c0 = proj(p, S_VALS[0],  0)
    c1 = proj(p, S_VALS[0],  n_e - 1)
    c2 = proj(p, S_VALS[-1], n_e - 1)
    c3 = proj(p, S_VALS[-1], 0)

    if p != 0:
        poly = mpatches.Polygon([c0, c1, c2, c3], closed=True,
                                 facecolor='#c8c4dc', edgecolor='none',
                                 alpha=0.12, zorder=zb)
        ax.add_patch(poly)

    # plane_segs = []
    # if p != 0:
    #     plane_segs += [[c0,c1],[c1,c2],[c2,c3],[c3,c0]]
    #     for si, s in enumerate(S_VALS):
    #         if si % 2 == 0:
    #             for ei in range(n_e - 1):
    #                 plane_segs.append([proj(p, s, ei), proj(p, s, ei+1)])
    #     for ei in range(n_e):
    #         for si in range(n_s - 1):
    #             if si % 2 == 0:
    #                 plane_segs.append([proj(p, S_VALS[si],   ei),
    #                                    proj(p, S_VALS[si+1], ei)])
    #     ax.add_collection(LineCollection(plane_segs,
    #         colors=EDGE_COL, linewidths=EDGE_LW, alpha=0.50, zorder=zb+2))

    s_range = [S_VALS[0]] if p == 0 else S_VALS
    e_range = [0]         if p == 0 else range(n_e)
    for s in s_range:
        for ei in e_range:
            c = proj(p, s, ei)
            is_path_node = (p, s, ei) in OPT_SET
            if is_path_node:
                fill = '#5B0EA6' #'black'
                edge_color = '#5B0EA6' #'black'
                edge_lw = 1.0
            else:
                fill = FLEET_COLORS[ei]
                edge_color = '#222'
                edge_lw = 0.8
            circ = mpatches.Circle(c, radius=NODE_R,
                                    facecolor=fill, edgecolor=edge_color,
                                    linewidth=edge_lw,
                                    zorder=950 if is_path_node else zb+3)
            ax.add_patch(circ)

    # x tick number
    bot = proj(p, S_VALS[0], 0)
    ax.text(bot[0], O[1] - 0.15, f'${p}$',
            ha='center', va='top', fontsize=30, color='#333', zorder=zb+4)

    # ── per-period change annotation: downward arrow + label ─────────────
    if p in PERIOD_NOTES:
        # find the fleet color of the path node(s) passing through this period
        path_nodes_here = [n for n in OPT_PATH if n[0] == p]
        last_ei_here = path_nodes_here[-1][2]   # color by the exit node's fleet
        arrow_color = FLEET_COLORS[last_ei_here]

        arrow_top    = (bot[0], O[1] - 0.55)
        arrow_bottom = (bot[0], O[1] - 1.35)
        ax.annotate('', xy=arrow_bottom, xytext=arrow_top,
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                    lw=3.2, mutation_scale=18),
                    zorder=zb+4)
        ax.text(bot[0], O[1] - 1.55, PERIOD_NOTES[p],
                ha='center', va='top', fontsize=25, color='#222',
                zorder=zb+4, style='italic')

ax.text(O[0], O[1] - 0.15, '$-1$',
        ha='center', va='top', fontsize=30, color='#333', zorder=999)

# ── OPTIMAL PATH EDGES ──────────────────────────────────────────────────────
for i in range(len(OPT_PATH) - 1):
    a = proj(*OPT_PATH[i])
    b = proj(*OPT_PATH[i+1])
    ax.plot([a[0],b[0]], [a[1],b[1]],
            color='#5B0EA6', lw=4.5, alpha=0.5, zorder=895, #'black'
            solid_capstyle='round')

aw = dict(arrowstyle='->', color='#000', lw=2.2, mutation_scale=18)

tip_p = np.array([proj(n_p-1, S_VALS[0], 0)[0] + sp_p*0.8, O[1]])
ax.annotate('', xy=tip_p, xytext=O, arrowprops=aw, zorder=999)
mid_p = np.array([(O[0] + tip_p[0])/2, O[1]])
ax.text(mid_p[0], mid_p[1] - 3.2,
        'DP Stage  (Planning Period)',
        ha='center', va='top', fontsize=30)

tip_e = O + (n_e - 1 + 0.8) * sp_e * v_e
ax.annotate('', xy=tip_e, xytext=O, arrowprops=aw, zorder=999)
rot_e = np.degrees(np.arctan2(v_e[1], v_e[0]))
tip_e_label = O + (n_e - 1 + 1.1) * sp_e * v_e
ax.text(tip_e_label[0] + 0.1, tip_e_label[1] + 0.05,
        'Fleet Size ($\\eta^p$)',
        ha='left', va='bottom', fontsize=30, rotation=rot_e)

tip_s = O + (n_s - 1 + 0.8) * sp_s * v_s
ax.annotate('', xy=tip_s, xytext=O, arrowprops=aw, zorder=999)
mid_s = O + (n_s - 1)/2 * sp_s * v_s
ax.text(mid_s[0] - 0.4, mid_s[1],
        'Charging Station Config. ($|S^p|$)',
        ha='right', va='center', fontsize=30, rotation=90)

ax.autoscale_view()
xl = ax.get_xlim(); yl = ax.get_ylim()
ax.set_xlim(xl[0] - 4.0, xl[1] + 2.5)
ax.set_ylim(yl[0] - 6.5, yl[1] + 2.5)   # extra bottom room for annotations

plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    plt.savefig(f'Figures/Dynamic_Programming_illustration_v4.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
print("Done.")




"Impact of fleet size and driving range: Figure 6"

# ---- Data ----
# Driving range R (miles)
R = [80, 70, 60, 50, 40, 30]

# Total cost is identical across eta = 2,3,4,5 (per the provided data)
total_cost = [272.7, 272.7, 280.7, 289.6, 302.9, 419.6]

# Number of ETs used - constant at 2 across all eta and all R
ets_used = 2

# Computation time (right-hand y-axis), same order as R
computation_time = [627, 627, 1255, 2919, 8, 7]

# R = 40 and R = 30 have comparatively tiny computation times that would
# flatten the larger values on the right axis, so exclude them from the
# computation-time series (the total-cost series still uses the full R).
R_comp_time = R[:4]
computation_time = computation_time[:4]

fleet_sizes = [2, 3]

# Explicit legend labels (the second line represents eta = 3, 4, and 5,
# since their total-cost values are identical to eta = 3's).
# Use explicit mathtext spaces ("\ ") around "or" since mathtext collapses
# regular spaces.
legend_labels = [r'$\eta = 2$', r'$\eta = 3,\ 4\ or\ 5$']

# Solid markers and colors, matching the style of Fig. 11
markers = ['^', 's', '', 'D']
colors = ['red', 'purple'] #'green', 'orange',

fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Plot one line per fleet size; since values are identical, lines overlap.
# Apply a small horizontal jitter (in R-units) purely for marker visibility,
# since R is now the x-axis.
offset_step = 0.0
for i, eta in enumerate(fleet_sizes):
    jitter = (i - (len(fleet_sizes) - 1) / 2) * offset_step
    R_jittered = [r + jitter for r in R]
    ax.plot(
        R_jittered, total_cost,
        marker=markers[i], color=colors[i], linestyle='-',
        linewidth=1.6, markersize=8, markerfacecolor=colors[i],
        markeredgecolor=colors[i], markeredgewidth=1.0, alpha=0.9,
        label=legend_labels[i], zorder=3
    )

# Annotate that only 2 ETs are ever used, regardless of eta or R
for r, c in zip(R, total_cost):
    ax.annotate(
        f'{ets_used} ETs',
        xy=(r, c), xytext=(0, 8), textcoords='offset points',
        ha='center', va='bottom', fontsize=12, color='black', alpha=0.7
    )

ax.set_xlabel('Driving Range ($R$)', fontsize=14)
ax.set_ylabel('Total Cost', fontsize=14)

# ---- Right-hand y-axis: computation time ----
ax2 = ax.twinx()
comp_time_line, = ax2.plot(
    R_comp_time, computation_time,
    marker='o', color='orange', linestyle='-',
    linewidth=1.6, markersize=8, markerfacecolor='orange',
    markeredgecolor='orange', markeredgewidth=1.0, alpha=0.9,
    label='Computation Time', zorder=3
)
ax2.set_ylabel('Computation Time (sec)', fontsize=14, color='black')
ax2.tick_params(axis='y', colors='black')
ax2.set_ylim(bottom=0, top=max(computation_time) * 1.1)

# ---- Axis limits (set BEFORE tick/annotation placement, since both depend
# on the limits). Left and top are padded slightly beyond the data range so
# that the "2 ETs" annotation on the R=30 / cost=419.6 point (which sits
# right at the data extremes) stays fully inside the plotting area instead
# of spilling outside the axes box.
ax.set_xlim(left=27, right=85)
ax.set_ylim(bottom=260, top=445)

# ---- Major ticks (labeled) ----
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

ax.tick_params(which='major', direction='in', length=5)

# ---- Vertical dashed reference lines at each R value (in place of a full
# grid), matching the style of Fig. 14: dashed gray lines spanning the
# plot height at each x tick.
for r in R:
    ax.axvline(x=r, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=0)

# Keep all four spines visible so the plot area is fully boxed, matching the reference figure
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.0)

# Legend, using default box styling (matches reference figure) with explicit
# handle ordering, following the same pattern used in the alpha-sensitivity plot.
handles, labels = ax.get_legend_handles_labels()
desired_order = list(legend_labels)
ordered_handles = [handles[labels.index(lbl)] for lbl in desired_order]
ordered_handles.append(comp_time_line)
desired_order.append('Computation Time')
ax.legend(ordered_handles, desired_order, fontsize=9, title_fontsize=9, loc='upper right')
#title='Fleet Size',

plt.tight_layout()
plt.savefig('Figures/fleet_size_analysis.pdf', bbox_inches='tight')
print("Saved plots.")




'Tallahassee Region Plot: Figure 7'

# Filepaths for all ten periods
DATA_DIR = 'DP_FullScale_data'
PERIOD_NAMES = ['first', 'second', 'third', 'fourth', 'fifth',
                'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
period_filepaths = {
    k: f'{DATA_DIR}/{name}_period/{name}_period_nodes.csv'
    for k, name in enumerate(PERIOD_NAMES, start=1)
}

# 1. Read all ten period CSVs.
# Checked against your actual files: each period is a strict superset of the
# last (60, 70, 80, ... 150 V-nodes), every Node ID has identical coordinates
# in every period it appears in, and there's a single consistent depot
# (Node ID 151) and 14 charging nodes shared identically across all ten
# files. Because of that, period 10 alone is a complete, single source of
# truth for geometry -- we only need periods 1-9 to work out WHICH Node IDs
# count as "existing" vs "future" below.
period_dfs = {k: pd.read_csv(v) for k, v in period_filepaths.items()}


def v_ids(df):
    return set(df.loc[df['Label'] == 'V', 'Node ID'].astype(int))


# "Existing" pool = every community that has appeared by period 5.
# "Future" pool   = the FULL periods 6-10 universe (150 communities -- since
#                    the periods nest, this is just period 10's complete set)
#                    minus whichever specific nodes end up drawn for the
#                    existing sample below (not the whole existing pool --
#                    only the ones actually picked). This is a deliberate
#                    choice: it widens the future pool from 50 (genuinely-new
#                    only) to up to 150 - N_EXISTING_SAMPLE, which is what
#                    makes an equal 60/60 split possible. The tradeoff: a
#                    community landing in the "future" sample may actually
#                    have existed since periods 1-5 -- it just wasn't one of
#                    the ones drawn into the existing group. See the
#                    [check] print after sampling for exactly how many of the
#                    sampled "future" nodes this affects.
existing_pool = set()
for k in range(1, 6):
    existing_pool |= v_ids(period_dfs[k])

full_future_universe = set()
for k in range(6, 11):
    full_future_universe |= v_ids(period_dfs[k])

print(f"[check] existing pool (periods 1-5): {len(existing_pool)} communities")
print(f"[check] full periods 6-10 universe: {len(full_future_universe)} communities")

SAMPLE_SEED = 42
N_EXISTING_SAMPLE = 50
N_FUTURE_SAMPLE = 50
MIN_SEPARATION_METERS = 1000

if N_EXISTING_SAMPLE > len(existing_pool):
    raise ValueError(
        f"N_EXISTING_SAMPLE={N_EXISTING_SAMPLE} is larger than the existing pool "
        f"({len(existing_pool)} communities available)."
    )
_max_future = len(full_future_universe) - N_EXISTING_SAMPLE
if N_FUTURE_SAMPLE > _max_future:
    raise ValueError(
        f"N_FUTURE_SAMPLE={N_FUTURE_SAMPLE} is larger than what's available once "
        f"N_EXISTING_SAMPLE={N_EXISTING_SAMPLE} communities are drawn from the "
        f"{len(full_future_universe)}-community periods 6-10 universe (at most "
        f"{_max_future} remain) -- lower N_FUTURE_SAMPLE or N_EXISTING_SAMPLE."
    )

# Coordinates (in meters) for every community, used only to keep the sample
# spatially spread out -- built from period 10, the single source of truth.
_terminal_v = period_dfs[10][period_dfs[10]['Label'] == 'V'].copy()
_terminal_v_proj = gpd.GeoDataFrame(
    _terminal_v,
    geometry=gpd.points_from_xy(_terminal_v['X Coordinate'], _terminal_v['Y Coordinate']),
    crs="EPSG:4326"
).to_crs(epsg=3857)
coords_lookup_m = {
    int(nid): (pt.x, pt.y) for nid, pt in zip(_terminal_v_proj['Node ID'], _terminal_v_proj.geometry)
}


def spatially_thinned_sample(pool_ids, quota, rng, accepted_coords, label):
    """Greedily draw up to `quota` ids from pool_ids (in random order),
    rejecting any candidate within MIN_SEPARATION_METERS of an already
    -accepted point. `accepted_coords` is shared across calls so existing and
    future samples stay mutually separated too, not just within their own
    group. If there aren't enough spatially-separated candidates left to
    reach `quota`, the shortfall is filled from the rejected leftovers (so
    the function always returns exactly `quota` ids) and a warning names
    which node(s) were included despite being too close."""
    shuffled = list(pool_ids)
    rng.shuffle(shuffled)
    accepted, rejected = [], []
    for nid in shuffled:
        pt = coords_lookup_m[nid]
        if all(((pt[0] - a[0]) ** 2 + (pt[1] - a[1]) ** 2) ** 0.5 >= MIN_SEPARATION_METERS
               for a in accepted_coords):
            accepted.append(nid)
            accepted_coords.append(pt)
        else:
            rejected.append(nid)
        if len(accepted) == quota:
            break
    shortfall = quota - len(accepted)
    if shortfall > 0:
        fill = rejected[:shortfall]
        for nid in fill:
            accepted.append(nid)
            accepted_coords.append(coords_lookup_m[nid])
        print(f"[check] WARNING: only {quota - shortfall} of {quota} {label} communities "
              f"could be kept at least {MIN_SEPARATION_METERS} m from every other sampled "
              f"point -- {shortfall} closer node(s) had to be included anyway to hit the "
              f"quota: {fill}. Lower N_{label.upper()}_SAMPLE or MIN_SEPARATION_METERS to avoid this.")
    return accepted


rng = np.random.default_rng(SAMPLE_SEED)
_accepted_coords = []  # shared across both calls below

# Existing is drawn first from its own pool...
sampled_existing_ids = set(
    spatially_thinned_sample(sorted(existing_pool), N_EXISTING_SAMPLE, rng, _accepted_coords, 'existing')
)

# ...then future is drawn from the full periods-6-10 universe MINUS whichever
# specific nodes just got drawn as existing (not the whole existing pool).
future_draw_pool = full_future_universe - sampled_existing_ids
sampled_future_ids = set(
    spatially_thinned_sample(sorted(future_draw_pool), N_FUTURE_SAMPLE, rng, _accepted_coords, 'future')
)

sampled_ids = sampled_existing_ids | sampled_future_ids
n_total = len(sampled_ids)

print(f"[check] sampled existing: {len(sampled_existing_ids)}, sampled future: "
      f"{len(sampled_future_ids)}, total sampled: {n_total}")
print(f"[check] largest sampled community Node ID: {max(sampled_ids)} -- "
      f"{'exceeds 119, looks like a genuine sample' if max(sampled_ids) > 119 else 'WARNING: does NOT exceed 119, reroll the seed'}")

# Semantic caveat check: how many of the sampled "future" (purple) nodes are
# actually communities that already existed in periods 1-5, just not drawn
# into the existing group? These will be colored/labeled as "future" in the
# figure despite having existed since periods 1-5.
_future_but_actually_existing = sampled_future_ids & existing_pool
print(f"[check] of the {len(sampled_future_ids)} sampled 'future' nodes, "
      f"{len(_future_but_actually_existing)} were already present in periods 1-5 "
      f"(not genuinely new): {sorted(_future_but_actually_existing)}")

# 3. Build the plotting frame from period 10 (the complete terminal dataset)
# filtered down to just the sampled community IDs, plus the depot and the
# charging nodes (charging nodes are structural, not part of the community
# sample, so all of them are kept here -- same as before).
terminal_df = period_dfs[10].copy()

v_rows = terminal_df[
    (terminal_df['Label'] == 'V') & (terminal_df['Node ID'].isin(sampled_ids))
].copy()
v_rows['color'] = v_rows['Node ID'].apply(lambda nid: 'orange' if nid in sampled_existing_ids else 'orange')
v_rows['marker'] = 'o'

non_v_rows = terminal_df[terminal_df['Label'] != 'V'].copy()


def assign_other_style(row):
    if row['Label'] in ['C+', 'C-']:
        return 'red', 'o'
    if row['Label'] == 'D':
        return 'blue', 'D'
    return 'purple', 'o'


non_v_rows['color'], non_v_rows['marker'] = zip(*non_v_rows.apply(assign_other_style, axis=1))

df = pd.concat([v_rows, non_v_rows], ignore_index=True)

# 4. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["X Coordinate"], df["Y Coordinate"]),
    crs="EPSG:4326"
)

# 5. Load & buffer the city boundary
boundaries = gpd.read_file("Figures/Tallahassee_City_Limit.geojson").to_crs("EPSG:4326")
boundaries_proj = boundaries.to_crs(epsg=3857)

# 5a. Buffer by BUFFER_METERS. Bump this if community nodes are getting
# dropped by the boundary filter below -- see the [check] diagnostic, which
# tells you exactly which node(s) and how far outside the unbuffered
# boundary they sit.
# Checked against your actual geojson + all 150 communities: the single
# farthest community from the raw boundary (Node ID 126) sits ~4515 m out,
# and it can land in the sample depending on the seed. 4600 m clears it (and
# every other community) with headroom, regardless of reseeding.
BUFFER_METERS = 4600
boundaries_unbuffered_proj = boundaries_proj.copy()  # kept for the diagnostic below
boundaries_proj["geometry"] = boundaries_proj.buffer(BUFFER_METERS)
boundaries = boundaries_proj.to_crs("EPSG:4326")

# 6. Spatial join to keep only points within the (buffered) city
joined = gpd.sjoin(gdf, boundaries, how="inner", predicate="within")
gdf_points = joined[joined.geometry.type == "Point"].copy()
gdf_points.reset_index(drop=True, inplace=True)

# --- Sanity check: confirm the boundary/city-limit filter didn't drop any
# sampled community nodes ---
n_final_community = (gdf_points['Label'] == 'V').sum()
print(f"[check] sampled community node count after boundary filter: {n_final_community}")
if n_final_community != n_total:
    dropped_ids = sampled_ids - set(gdf_points.loc[gdf_points['Label'] == 'V', 'Node ID'])
    print(f"[check] WARNING: {len(dropped_ids)} sampled community node(s) were dropped by "
          f"the boundary/city-limit spatial join (BUFFER_METERS={BUFFER_METERS}).")
    dropped_pts = gdf[gdf['Node ID'].isin(dropped_ids)].to_crs(epsg=3857)
    for _, r in dropped_pts.iterrows():
        dist = boundaries_unbuffered_proj.distance(r.geometry).min()
        print(f"[check]   Node ID {int(r['Node ID'])} is ~{dist:.0f} m outside the "
              f"unbuffered city boundary -- set BUFFER_METERS above that to keep it.")

# 6a. Only keep charging nodes (C+/C-) that are part of a charging-arc pair
# in join_pairs; any C+/C- node not part of a pair is dropped from the plot.
# NOTE: these are the new charging Node IDs (152-171). Matched by coordinate
# to your old (126-149) IDs -- (152,153) is at the exact same location as the
# old in-service pair (134,135), so it stays the one solid/non-hollow pair.
join_pairs = [(152, 153), (158, 159), (160, 161),
              (162, 163), (164, 165), (166, 167), (170, 171)]
join_node_ids = {nid for pair in join_pairs for nid in pair}

gdf_points = gdf_points[
    ~(gdf_points['Label'].isin(['C+', 'C-']) & ~gdf_points['Node ID'].isin(join_node_ids))
].reset_index(drop=True)

# 6b. Flag which charging nodes belong to the "hollow" subset of arcs
# These are the join_pairs (node ID pairs) whose markers should render
# as hollow (unfilled, red-edged) circles instead of solid red.
hollow_pairs = [(158, 159), (160, 161), (162, 163), (164, 165), (166, 167), (170, 171)]
hollow_node_ids = {nid for pair in hollow_pairs for nid in pair}

gdf_points['is_hollow'] = gdf_points['Node ID'].isin(hollow_node_ids)

# 7. Begin plotting
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

# 7a. Draw the thick black connection lines *behind* the nodes
for a, b in join_pairs:
    pa = gdf_points[gdf_points["Node ID"] == a]
    pb = gdf_points[gdf_points["Node ID"] == b]
    if not pa.empty and not pb.empty:
        x1, y1 = pa.geometry.iloc[0].x, pa.geometry.iloc[0].y
        x2, y2 = pb.geometry.iloc[0].x, pb.geometry.iloc[0].y
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=3, zorder=1)

# 7b. Plot solid (non-hollow) style groups (nodes on top with zorder=3)
solid_points = gdf_points[~gdf_points['is_hollow']]
for (col, mkr), subset in solid_points.groupby(["color", "marker"]):
    subset.plot(ax=ax, color=col, marker=mkr, markersize=50, linewidth=0, zorder=3)

# 7c. Plot hollow markers for the specified charging arcs
hollow_points = gdf_points[gdf_points['is_hollow']]
if not hollow_points.empty:
    ax.scatter(
        hollow_points.geometry.x,
        hollow_points.geometry.y,
        facecolors='none',
        edgecolors='red',
        marker='o',
        s=50,
        linewidths=1.5,
        zorder=3
    )

# Label every SAMPLED community node with its TRUE original Node ID, and the
# depot as "0". Charging nodes (C+/C-) are not labeled.
for _, row in gdf_points.iterrows():
    if row["Label"] == 'V':
        label = str(int(row["Node ID"]))
    elif row["Label"] == 'D':
        label = "0"
    else:
        continue
    ax.text(
        row.geometry.x, row.geometry.y + 0.0005,
        label,
        fontsize=8,
        ha='center', va='bottom',
        color='black',
        zorder=4
    )

# 8. Add the Stadia Maps basemap
cx.add_basemap(
    ax,
    crs=gdf_points.crs,
    zoom=14,
    source=(
        "https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png"
        f"?api_key={os.environ['STADIA_API_KEY']}"
    )
)

# 9. Build the legend
legend_handles = [
    # mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=8,
    #               label='Existing communities'),
    mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=8,
                  label='Communities'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8,
                  label='Charging nodes (Existing)'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8,
                  markerfacecolor='none', label='Charging nodes (Planned)'),
    mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=8,
                  label='Depot')
]
ax.legend(
    handles=legend_handles,
    loc='lower left',
    fontsize='small',
    markerscale=0.6,
    labelspacing=0.3,
    handlelength=1.5,
    handletextpad=0.4
)

# 10. Final formatting & save
ax.set_axis_off()
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig("Figures/Tallahassee_region.svg", bbox_inches='tight')








