import os
import pandas as pd
import pandas as pd
import networkx as nx
from pathlib import Path
import pandas as pd
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import itertools
import numpy as np
from DP_FullScale_scripts import generate_travel_parameters, save_travel_data
from DP_FullScale_scripts import generate_and_save_service_times, generate_and_save_waste_demands


def generate_network_links(nodes_csv_path: str, output_links_csv_path: str) -> None:
    """
    Generate all possible directed edges (excluding self-loops) from a nodes CSV
    and save them into a links CSV.

    Inputs:
      - nodes_csv_path: path to CSV with columns ['Node ID', 'X Coordinate', 'Y Coordinate', 'Label']
      - output_links_csv_path: destination CSV path

    Output CSV columns:
      ['from_node ID', 'to_node ID', 'start_lat', 'start_lon', 'end_lat', 'end_lon']
    """
    # Load network nodes data
    nodes_df = pd.read_csv(nodes_csv_path)

    # Standardize column names if needed
    nodes_df.columns = ['Node ID', 'X Coordinate', 'Y Coordinate', 'Label']

    # Create mapping from Node ID to coordinates (lat, lon)
    node_coord = {
        row['Node ID']: (row['X Coordinate'], row['Y Coordinate'])
        for _, row in nodes_df.iterrows()
    }

    # Generate all possible directed edges (excluding self-loops)
    all_edges = []
    for from_node, to_node in itertools.permutations(node_coord.keys(), 2):
        start_lat, start_lon = node_coord[from_node]
        end_lat, end_lon = node_coord[to_node]
        all_edges.append({
            'from_node ID': from_node,
            'to_node ID': to_node,
            'start_lat': round(start_lat, 6),
            'start_lon': round(start_lon, 6),
            'end_lat': round(end_lat, 6),
            'end_lon': round(end_lon, 6)
        })

    # Save
    edges_df = pd.DataFrame(all_edges)
    edges_df.to_csv(output_links_csv_path, index=False)



'Codes to generate complete network version of my case study data'
def generate_complete_network(
    nodes_data_file,
    links_data_file,
    travel_data_file,
    charging_links_file,
    open_charging_link_ids,
    save_modified_nodes,
    save_modified_links,
    save_modified_travel_data
):
    # Load node and travel data
    nodes_df = pd.read_csv(nodes_data_file)
    travel_data_df = pd.read_csv(travel_data_file)
    charging_links_df = pd.read_csv(charging_links_file)

    # Identify customer (V) and depot (D) nodes
    V_nodes = nodes_df[nodes_df['Label'] == 'V']['Node ID'].tolist()
    D_node = nodes_df[nodes_df['Label'] == 'D']['Node ID'].iloc[0]

    # Filter allowed charging links
    allowed_charging_links_df = charging_links_df[
        charging_links_df['Link ID'].isin(open_charging_link_ids)
    ]
    allowed_charging_pairs = set(
        zip(
            allowed_charging_links_df['from_node ID'].astype(str),
            allowed_charging_links_df['to_node ID'].astype(str)
        )
    )
    allowed_charging_node_ids = set(allowed_charging_links_df['from_node ID']).union(
        set(allowed_charging_links_df['to_node ID'])
    )

    # Build graph from travel data
    G = nx.DiGraph()
    G.add_nodes_from(nodes_df['Node ID'])
    for _, row in travel_data_df.iterrows():
        G.add_edge(row['from_node ID'], row['to_node ID'],
                   weight=row['Distance (miles)'], time=row['Time (minutes)'])

    node_coords = {
        row['Node ID']: (row['Y Coordinate'], row['X Coordinate']) for _, row in nodes_df.iterrows()
    }

    modified_links = []
    modified_travel_data = []

    # 1. Add shortest paths between all V-V (excluding depot)
    G_without_depot = G.copy()
    G_without_depot.remove_node(D_node)
    for i, v1 in enumerate(V_nodes):
        for v2 in V_nodes[i+1:]:
            if nx.has_path(G_without_depot, v1, v2):
                dist = nx.shortest_path_length(G_without_depot, v1, v2, weight='weight')
                time = nx.shortest_path_length(G_without_depot, v1, v2, weight='time')
                for a, b in [(v1, v2), (v2, v1)]:
                    modified_links.append({
                        'from_node ID': a,
                        'to_node ID': b,
                        'start_lat': node_coords[a][0],
                        'start_lon': node_coords[a][1],
                        'end_lat': node_coords[b][0],
                        'end_lon': node_coords[b][1]
                    })
                    modified_travel_data.append({
                        'from_node ID': a,
                        'to_node ID': b,
                        'Distance (miles)': round(dist, 2),
                        'Time (minutes)': round(time, 2)
                    })

    # 2. Add D ↔ V paths
    for v in V_nodes:
        if nx.has_path(G, D_node, v):
            dist = nx.shortest_path_length(G, D_node, v, weight='weight')
            time = nx.shortest_path_length(G, D_node, v, weight='time')
            for a, b in [(D_node, v), (v, D_node)]:
                modified_links.append({
                    'from_node ID': a,
                    'to_node ID': b,
                    'start_lat': node_coords[a][0],
                    'start_lon': node_coords[a][1],
                    'end_lat': node_coords[b][0],
                    'end_lon': node_coords[b][1]
                })
                modified_travel_data.append({
                    'from_node ID': a,
                    'to_node ID': b,
                    'Distance (miles)': round(dist, 2),
                    'Time (minutes)': round(time, 2)
                })

    # 3. Add C+ → C- links only if in allowed_charging_pairs
    for from_node, to_node in allowed_charging_pairs:
        from_node = int(from_node)
        to_node = int(to_node)
        row = travel_data_df[(travel_data_df['from_node ID'] == from_node) &
                             (travel_data_df['to_node ID'] == to_node)]
        if not row.empty:
            distance = row['Distance (miles)'].iloc[0]
            time = row['Time (minutes)'].iloc[0]
            modified_links.append({
                'from_node ID': from_node,
                'to_node ID': to_node,
                'start_lat': node_coords[from_node][0],
                'start_lon': node_coords[from_node][1],
                'end_lat': node_coords[to_node][0],
                'end_lon': node_coords[to_node][1]
            })
            modified_travel_data.append({
                'from_node ID': from_node,
                'to_node ID': to_node,
                'Distance (miles)': round(distance, 2),
                'Time (minutes)': round(time, 2)
            })

    # 4. Add X → C+ and C- → X if charging node is in allowed_charging_node_ids
    for from_node, to_node in travel_data_df[['from_node ID', 'to_node ID']].values:
        from_label = nodes_df[nodes_df['Node ID'] == from_node]['Label'].values[0]
        to_label = nodes_df[nodes_df['Node ID'] == to_node]['Label'].values[0]

        if (from_label not in ['C+', 'C-'] and to_label == 'C+' and to_node in allowed_charging_node_ids) or \
           (from_label == 'C-' and from_node in allowed_charging_node_ids and to_label not in ['C+', 'C-']):

            row = travel_data_df[(travel_data_df['from_node ID'] == from_node) &
                                 (travel_data_df['to_node ID'] == to_node)]
            if not row.empty:
                distance = row['Distance (miles)'].iloc[0]
                time = row['Time (minutes)'].iloc[0]
                modified_links.append({
                    'from_node ID': from_node,
                    'to_node ID': to_node,
                    'start_lat': node_coords[from_node][0],
                    'start_lon': node_coords[from_node][1],
                    'end_lat': node_coords[to_node][0],
                    'end_lon': node_coords[to_node][1]
                })
                modified_travel_data.append({
                    'from_node ID': from_node,
                    'to_node ID': to_node,
                    'Distance (miles)': round(distance, 2),
                    'Time (minutes)': round(time, 2)
                })

    # Final node set includes depot, customers, and allowed charging nodes
    final_nodes = set(V_nodes + [D_node]).union(allowed_charging_node_ids)
    modified_nodes_df = nodes_df[nodes_df['Node ID'].isin(final_nodes)].drop_duplicates()

    # Save outputs
    pd.DataFrame(modified_links).to_csv(save_modified_links, index=False)
    pd.DataFrame(modified_travel_data).to_csv(save_modified_travel_data, index=False)
    modified_nodes_df.to_csv(save_modified_nodes, index=False)


def plot_network_nodes_simple(
    nodes_csv_path: str,
    output_path: str = None,
    charging_links_csv_path: str = None
) -> None:
    """
    Visualize network nodes from a CSV file and save the figure.
    - V nodes: white fill, enlarged so IDs are visible
    - C+/C- nodes: very small red fill so charging arcs remain visible
    - D nodes: orange fill
    - Axis labels and tick marks are shown

    Args:
        nodes_csv_path: Path to CSV with columns ['Node ID', 'X Coordinate', 'Y Coordinate', 'Label']
        output_path: File path to save the SVG visualization
            (default: Figures/network_visualization.svg in the nodes file's folder)
        charging_links_csv_path: charging_links.csv used to label the charging arcs
            (default: charging_links.csv in the nodes file's folder)
    """
    nodes_folder = os.path.dirname(os.path.abspath(nodes_csv_path))
    if output_path is None:
        output_path = os.path.join(nodes_folder, "Figures", "network_visualization.svg")
    if charging_links_csv_path is None:
        charging_links_csv_path = os.path.join(nodes_folder, "charging_links.csv")

    # ── Load & validate ───────────────────────────────────────────────────────
    df = pd.read_csv(nodes_csv_path)
    required_cols = {'Node ID', 'X Coordinate', 'Y Coordinate', 'Label'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # ── Charging arc map: (C+ node ID, C- node ID) → Link ID ─────────────────
    charging_links_df = pd.read_csv(charging_links_csv_path)
    charging_arcs = {
        (int(row['from_node ID']), int(row['to_node ID'])): int(row['Link ID'])
        for _, row in charging_links_df.iterrows()
    }

    charging_node_ids = {nid for pair in charging_arcs for nid in pair}

    # ── Build NetworkX graph ──────────────────────────────────────────────────
    G = nx.DiGraph()
    pos = {}

    for _, row in df.iterrows():
        nid   = int(row['Node ID'])
        label = row['Label']
        G.add_node(nid, label=label)
        pos[nid] = (row['X Coordinate'], row['Y Coordinate'])

    # Add directed charging edges
    for (src, dst) in charging_arcs:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)

    # ── Separate node groups ──────────────────────────────────────────────────
    v_nodes = [n for n, d in G.nodes(data=True) if d['label'] == 'V']
    c_plus_nodes = [n for n, d in G.nodes(data=True) if d['label'] == 'C+' and n in charging_node_ids]
    c_minus_nodes = [n for n, d in G.nodes(data=True) if d['label'] == 'C-' and n in charging_node_ids]
    d_nodes = [n for n, d in G.nodes(data=True) if d['label'] == 'D']

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))

    v_node_size = 600
    d_node_size = 600
    charging_node_size = 70

    # V nodes — large white circles so IDs are clearly readable
    nx.draw_networkx_nodes(G, pos, nodelist=v_nodes, ax=ax,
                           node_color='white', edgecolors='black',
                           node_size=v_node_size, linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in v_nodes},
                            ax=ax, font_size=10)

    # C+ / C- nodes — very small red circles to keep charging arcs visible
    nx.draw_networkx_nodes(G, pos, nodelist=c_plus_nodes, ax=ax,
                           node_color='red', edgecolors='black',
                           node_size=charging_node_size, linewidths=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=c_minus_nodes, ax=ax,
                           node_color='red', edgecolors='black',
                           node_size=charging_node_size, linewidths=0.8)

    # D nodes — orange circles; original is start depot and offset copy is return depot
    nx.draw_networkx_nodes(G, pos, nodelist=d_nodes, ax=ax,
                           node_color='orange', edgecolors='black',
                           node_size=d_node_size, linewidths=1.0)

    # Add a small visual offset for the return depot copy so both depots are visible.
    depot_copy_pos = {}
    if d_nodes:
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        dx = max((max(x_vals) - min(x_vals)) * 0.03, 0.005)
        dy = max((max(y_vals) - min(y_vals)) * 0.03, 0.005)
        for n in d_nodes:
            x, y = pos[n]
            depot_copy_pos[n] = (x + dx, y + dy)

        nx.draw_networkx_nodes(
            G,
            depot_copy_pos,
            nodelist=d_nodes,
            ax=ax,
            node_color='orange',
            edgecolors='black',
            node_size=d_node_size,
            linewidths=1.0,
        )

    # Label original depot as 121+ and copied depot as 121-
    for n in d_nodes:
        x, y = pos[n]
        ax.text(x, y, f"{n}+", fontsize=11, ha='center', va='center', zorder=6)
        if n in depot_copy_pos:
            x2, y2 = depot_copy_pos[n]
            ax.text(x2, y2, f"{n}-", fontsize=11, ha='center', va='center', zorder=6)

    # ── Draw curved directed arcs for charging pairs ──────────────────────────
    for (src, dst), arc_num in charging_arcs.items():
        if src not in pos or dst not in pos:
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        # Draw curved arrow
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=1.5,
                connectionstyle="arc3,rad=0.35",
            ),
            zorder=4
        )

        # Label at arc midpoint (offset perpendicular to the line)
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy) or 1
        # Perpendicular offset scaled to map units
        offset_scale = 0.003
        ox = -dy / length * offset_scale
        oy =  dx / length * offset_scale

        ax.text(mx + ox, my + oy, str(arc_num),
                fontsize=8, color='black', fontweight='bold',
                ha='center', va='center', zorder=5)

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    ax.set_title("Spatial location of residential communities, charging stations and ET depot",
                 fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.tick_params(axis='both', labelsize=9)

    # Legend (node-only)
    legend_handles = [
        mlines.Line2D([], [], color='black', marker='o', markerfacecolor='white',
                      linestyle='None', markersize=12, label='V'),
        mlines.Line2D([], [], color='red', marker='o', markerfacecolor='red',
                      linestyle='None', markersize=10, label='C+'),
        mlines.Line2D([], [], color='red', marker='o', markerfacecolor='red',
                      linestyle='None', markersize=10, label='C-'),
        mlines.Line2D([], [], color='orange', marker='o', markerfacecolor='orange',
                      linestyle='None', markersize=12, label='D'),
    ]
    ax.legend(handles=legend_handles, title='Node Label', loc='upper right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {output_path}")





########################################################################################
"""Generate Data for the Tallahassee Case Study — full-scale instances"""
# This script lives in <project root>/DP_FullScale_scripts/. Every file it reads
# or writes is under <project root>/DP_FullScale_data/, whatever the working
# directory the IDE / terminal launches it from:
#   DP_FullScale_data/{period}_period/   {period}_period_nodes/_links/_travel_data/
#                                        _waste_demand/_service_time.csv, charging_links.csv
#   DP_FullScale_data/tallahassee_data/  full Tallahassee network (network_*.csv, complete_*.csv,
#                                        charging_links.csv) -- used by RUN_NETWORK and period sampling
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "DP_FullScale_data")


def period_folder(period):
    """<project root>/DP_FullScale_data/{period}_period"""
    return os.path.join(DATA_ROOT, f"{period}_period")


# open_charging_link_ids: None -> every Link ID in the folder's charging_links.csv
DATASETS = {
    "tallahassee": {"folder": os.path.join(DATA_ROOT, "tallahassee_data"), "open_charging_link_ids": None},
}
DATASET = "tallahassee"

PLOT_NODES = False

# Table D.8: communities, service duration T_i ~ U(min, max) in minutes,
# waste demand q_i ~ N(mean, std) in lbs.
PERIOD_CONFIG = {
    "first":   {"communities": 60,  "service": (15, 25),   "demand": (1220, 262)},
    "second":  {"communities": 70,  "service": (20, 35),   "demand": (1280, 282)},
    "third":   {"communities": 80,  "service": (30, 45),   "demand": (1310, 305)},
    "fourth":  {"communities": 90,  "service": (40, 55),   "demand": (1490, 455)},
    "fifth":   {"communities": 100, "service": (50, 65),   "demand": (1620, 510)},
    "sixth":   {"communities": 110, "service": (60, 75),   "demand": (1650, 535)},
    "seventh": {"communities": 120, "service": (70, 85),   "demand": (1830, 675)},
    "eighth":  {"communities": 130, "service": (80, 95),   "demand": (1920, 732)},
    "ninth":   {"communities": 140, "service": (90, 105),  "demand": (1998, 795)},
    "tenth":   {"communities": 150, "service": (100, 115), "demand": (2005, 805)},
}
DECIMAL_PLACES = 2

# RUN_NETWORK = True
# PERIODS_TO_RUN = []
# OVERWRITE_PERIOD_FILES = False

RUN_NETWORK = False
PERIODS_TO_RUN = ["first", "second", "third", "fourth", "fifth",
                  "sixth", "seventh", "eighth", "ninth", "tenth"]

OVERWRITE_PERIOD_FILES = True


def generate_network_data(folder, open_charging_link_ids):
    def path(name):
        return os.path.join(folder, name)

    nodes_file = path("network_nodes.csv")
    links_file = path("network_links.csv")
    travel_file = path("travel_data.csv")
    charging_links_file = path("charging_links.csv")

    if open_charging_link_ids is None:
        open_charging_link_ids = pd.read_csv(charging_links_file)["Link ID"].astype(int).tolist()

    if PLOT_NODES:
        plot_network_nodes_simple(nodes_file, charging_links_csv_path=charging_links_file)

    generate_network_links(nodes_file, links_file)

    travel_parameter = {}
    generate_travel_parameters(nodes_file, links_file, travel_parameter,
                               charging_links_file, open_charging_link_ids)
    save_travel_data(travel_parameter, travel_file)

    generate_complete_network(nodes_file, links_file, travel_file,
                              charging_links_file, open_charging_link_ids,
                              path("complete_nodes.csv"),
                              path("complete_links.csv"),
                              path("complete_travel_data.csv"))


def generate_period_demand_and_service(period):
    config = PERIOD_CONFIG[period]
    prefix = os.path.join(period_folder(period), f"{period}_period")
    nodes_file = f"{prefix}_nodes.csv"

    nodes_df = pd.read_csv(nodes_file)
    n_communities = (nodes_df["Label"] == "V").sum()
    if n_communities != config["communities"]:
        raise ValueError(f"{period}: {nodes_file} has {n_communities} communities, "
                         f"expected {config['communities']}.")

    demand_path = f"{prefix}_waste_demand.csv"
    service_path = f"{prefix}_service_time.csv"

    if os.path.exists(demand_path) and not OVERWRITE_PERIOD_FILES:
        print(f"{period}: keeping existing {demand_path}")
    else:
        mean, std = config["demand"]
        generate_and_save_waste_demands(network_nodes_path=nodes_file, output_path=demand_path,
                                        mean=mean, std=std, decimal_places=DECIMAL_PLACES)
        print(f"{period}: saved {demand_path} ~ N({mean}, {std})")

    if os.path.exists(service_path) and not OVERWRITE_PERIOD_FILES:
        print(f"{period}: keeping existing {service_path}")
    else:
        min_val, max_val = config["service"]
        generate_and_save_service_times(network_nodes_path=nodes_file, output_path=service_path,
                                        min_val=min_val, max_val=max_val,
                                        decimal_places=DECIMAL_PLACES)
        print(f"{period}: saved {service_path} ~ U({min_val}, {max_val})")


"Sample Period Data"
# Periods are sampled from the complete network in DP_FullScale_data/tallahassee_data/.
DATA_FOLDER = DATASETS["tallahassee"]["folder"]

NODE_FILE = "complete_nodes.csv"
TRAVEL_FILE = "complete_travel_data.csv"
LINKS_FILE = "complete_links.csv"

PERIODS = ["first", "second", "third", "fourth", "fifth",
           "sixth", "seventh", "eighth", "ninth", "tenth"]

# Number of sampled communities per Period (needed only for Periods not yet sampled).
SAMPLE_SIZES = {"first": 60, "second": 70, "third": 80, "fourth": 90, "fifth": 100,
                "sixth": 110, "seventh": 120, "eighth": 130, "ninth": 140, "tenth": 150}

# Periods to generate in this run.
#PERIODS_TO_RUN = ["first"]
PERIODS_TO_RUN = ["first", "second", "third", "fourth", "fifth",
                  "sixth", "seventh", "eighth", "ninth", "tenth"]

RANDOM_SEED = 42


def get_period_folder(period_name):
    folder = period_folder(period_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Period folder not found: {folder}")
    return folder


def find_existing_nodes_file(period_name):
    path = os.path.join(period_folder(period_name), f"{period_name}_period_nodes.csv")
    return path if os.path.exists(path) else None


def read_sampled_ids(nodes_file):
    df = pd.read_csv(nodes_file)
    return df.loc[df["Label"] == "V", "Node ID"].tolist()


def select_v_nodes(v_nodes, ids, source):
    selected = v_nodes[v_nodes["Node ID"].isin(ids)]
    if len(selected) != len(ids):
        raise ValueError(f"{len(ids) - len(selected)} community ID(s) in {source} "
                         f"not found in {NODE_FILE}.")
    return selected


def sample_v_nodes(v_nodes, period_name):
    period_index = PERIODS.index(period_name)
    if period_index == 0:
        fixed = v_nodes.iloc[0:0]
    else:
        prev_period = PERIODS[period_index - 1]
        prev_file = find_existing_nodes_file(prev_period)
        if prev_file is None:
            raise FileNotFoundError(f"Generate '{prev_period}' before '{period_name}'.")
        fixed = select_v_nodes(v_nodes, read_sampled_ids(prev_file), prev_file)

    existing_file = find_existing_nodes_file(period_name)
    if existing_file:
        ids = read_sampled_ids(existing_file)
        missing = set(fixed["Node ID"]) - set(ids)
        if missing:
            raise ValueError(f"{existing_file} is missing {len(missing)} communities "
                             f"from the previous Period.")
        print(f"{period_name}: reusing {len(ids)} sampled communities from {existing_file}")
        return select_v_nodes(v_nodes, ids, existing_file)

    if period_name not in SAMPLE_SIZES:
        raise ValueError(f"No sample size set for '{period_name}' in SAMPLE_SIZES.")
    sample_size = SAMPLE_SIZES[period_name]

    additional_needed = sample_size - len(fixed)
    if additional_needed < 0:
        raise ValueError(f"{period_name}: sample size {sample_size} is smaller than "
                         f"the {len(fixed)} communities of the previous Period.")

    remaining = v_nodes[~v_nodes["Node ID"].isin(fixed["Node ID"])]
    additional = remaining.sample(n=additional_needed, random_state=RANDOM_SEED)
    print(f"{period_name}: kept {len(fixed)} communities, sampled {additional_needed} new")
    return pd.concat([fixed, additional])


def filter_by_nodes(df, node_ids):
    return df[df["from_node ID"].isin(node_ids) &
              df["to_node ID"].isin(node_ids)].reset_index(drop=True)


def generate_period_data(period_name, nodes_df, travel_df, links_df):
    folder = get_period_folder(period_name)
    prefix = f"{period_name}_period"

    v_nodes = nodes_df[nodes_df["Label"] == "V"].sort_values("Node ID").reset_index(drop=True)
    sampled_v_nodes = sample_v_nodes(v_nodes, period_name)

    keep_nodes = pd.concat([sampled_v_nodes, nodes_df[nodes_df["Label"] != "V"]]).reset_index(drop=True)
    keep_ids = keep_nodes["Node ID"]

    outputs = {
        "nodes": keep_nodes,
        "travel_data": filter_by_nodes(travel_df, keep_ids),
        "links": filter_by_nodes(links_df, keep_ids),
    }
    for name, df in outputs.items():
        path = os.path.join(folder, f"{prefix}_{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {path} ({len(df)} rows)")


# def main():
#     nodes_df = pd.read_csv(os.path.join(DATA_FOLDER, NODE_FILE))
#     travel_df = pd.read_csv(os.path.join(DATA_FOLDER, TRAVEL_FILE))
#     links_df = pd.read_csv(os.path.join(DATA_FOLDER, LINKS_FILE))
#
#     for period_name in PERIODS_TO_RUN:
#         print("=" * 60)
#         print(f"Generating data for {period_name} Period")
#         print("=" * 60)
#         generate_period_data(period_name, nodes_df, travel_df, links_df)
#
#
# if __name__ == "__main__":
#     main()


def main():
    if RUN_NETWORK:
        dataset = DATASETS[DATASET]
        generate_network_data(dataset["folder"], dataset["open_charging_link_ids"])

    for period in PERIODS_TO_RUN:
        generate_period_demand_and_service(period)


if __name__ == "__main__":
    main()





