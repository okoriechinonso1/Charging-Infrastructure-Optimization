
#######################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import random
import csv
from typing import List
import networkx as nx


def generate_node(existing_nodes, region, min_distance=1.0):
    """Generate a random node location ensuring minimum distance from existing nodes."""
    while True:
        candidate = np.random.uniform(0, region, size=2)
        if all(np.linalg.norm(candidate - node) >= min_distance for node in existing_nodes):
            return candidate


def generate_charging_station(existing_nodes, region, min_distance=1.0, offset=1.0):
    """
    Generate a candidate location for a charging station (charging-on node) using generate_node.
    Then produce a paired charging-off node by shifting the candidate by a random offset.
    """
    candidate = generate_node(existing_nodes, region, min_distance)
    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        disp = np.array([offset * np.cos(angle), offset * np.sin(angle)])
        candidate_shifted = candidate + disp
        if np.all(candidate_shifted >= 0) and np.all(candidate_shifted < region):
            break
    return candidate, candidate_shifted


def build_network(depot, communities, charging_on, charging_off):
    """
    Given arrays for depot, communities, charging_on and charging_off nodes,
    build the alternative network with node IDs and labels, and generate the directed links.
    Returns a tuple (nodes, links) as lists of dictionaries.
    """
    nodes = []
    node_id = 1

    # Depot (Label 'D')
    nodes.append({'Node ID': node_id, 'X Coordinate': round(depot[0, 0], 4), 'Y Coordinate': round(depot[0, 1], 4), 'Label': 'D'})
    node_id += 1

    # Community nodes (Label 'V')
    for i in range(len(communities)):
        nodes.append(
            {'Node ID': node_id, 'X Coordinate': round(communities[i, 0], 4), 'Y Coordinate': round(communities[i, 1], 4), 'Label': 'V'})
        node_id += 1

    # Charging station pairs: each C+ immediately followed by its paired C-
    charging_on_ids = []
    charging_off_ids = []
    charging_links = []
    for i in range(len(charging_on)):
        on_x  = round(charging_on[i, 0], 4)
        on_y  = round(charging_on[i, 1], 4)
        off_x = round(charging_off[i, 0], 4)
        off_y = round(charging_off[i, 1], 4)

        nodes.append({'Node ID': node_id, 'X Coordinate': on_x, 'Y Coordinate': on_y, 'Label': 'C+'})
        charging_on_ids.append(node_id)
        on_node_id = node_id
        node_id += 1

        nodes.append({'Node ID': node_id, 'X Coordinate': off_x, 'Y Coordinate': off_y, 'Label': 'C-'})
        charging_off_ids.append(node_id)
        off_node_id = node_id
        node_id += 1

        charging_links.append({
            'Link ID':      i + 1,
            'from_node ID': on_node_id,
            'to_node ID':   off_node_id,
            'start_lat':    on_x,
            'start_lon':    on_y,
            'end_lat':      off_x,
            'end_lon':      off_y,
        })

    # Create mapping for charging pairs (from charging-on to charging-off)
    charging_pairs = {charging_on_ids[i]: charging_off_ids[i] for i in range(len(charging_on_ids))}
    total_nodes = len(nodes)

    # Build links: every ordered pair of distinct nodes is connected, except skip the reverse of a charging arc.
    links = []
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i == j:
                continue
            source = nodes[i]
            target = nodes[j]
            if target['Node ID'] in charging_pairs and charging_pairs[target['Node ID']] == source['Node ID']:
                continue
            links.append({
                'from_node ID': source['Node ID'],
                'to_node ID': target['Node ID'],
                'start_lat': source['X Coordinate'],
                'start_lon': source['Y Coordinate'],
                'end_lat': target['X Coordinate'],
                'end_lon': target['Y Coordinate']
            })
    return nodes, links, charging_links


def save_network(nodes, links, charging_links, node_filename, link_filename, charging_link_filename):
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)
    charging_links_df = pd.DataFrame(charging_links)
    nodes_df.to_csv(node_filename, index=False)
    links_df.to_csv(link_filename, index=False)
    charging_links_df.to_csv(charging_link_filename, index=False)


def plot_network(depot, communities, charging_on, charging_off, region, filename,
                  charging_rates=None, charging_costs=None,
                  label_perp_dist=1.3, label_line_gap=0.35, edge_margin=0.3):
    """
    Plot the network layout.

    - Depot: orange square
    - Communities: black circle
    - Charging-on/off nodes: small red circles, joined by a thin black arrow (on -> off)
    - Optional charging rate (alpha_ij, red) / cost per unit charge (pi_ij, blue) labels,
      offset perpendicular to the on-off line so they never overlap a node marker
    - No axis ticks, no title
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Depot
    ax.scatter(depot[:, 0], depot[:, 1], marker="s", color="orange", s=100,
               label="Depot", zorder=3, edgecolors='black', linewidths=0.5)

    # Communities
    ax.scatter(communities[:, 0], communities[:, 1], marker="o", color="black", s=100,
               label="Community", zorder=3)

    # Charging-on and charging-off nodes: smaller red circles, single merged legend entry
    if len(charging_on) > 0:
        combined_charging = np.vstack([charging_on, charging_off])
        ax.scatter(combined_charging[:, 0], combined_charging[:, 1], marker="o", color="red",
                   s=45, label="Charging Station Node", zorder=3, edgecolors='black', linewidths=0.5)

    # Thin black arrow from charging-on to charging-off, for each station
    for i in range(len(charging_on)):
        ax.annotate(
            '', xy=(charging_off[i, 0], charging_off[i, 1]),
            xytext=(charging_on[i, 0], charging_on[i, 1]),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.3, mutation_scale=12),
            zorder=2
        )

    # Thin blue/green links from the charging station closest to the depot to every community node
    if len(charging_on) > 0:
        depot_dists = np.linalg.norm(charging_on - depot[0], axis=1)
        nearest_idx = int(np.argmin(depot_dists))

        for j in range(len(communities)):
            ax.plot([charging_on[nearest_idx, 0], communities[j, 0]],
                    [charging_on[nearest_idx, 1], communities[j, 1]],
                    color='blue', linewidth=0.4, zorder=1,
                    label='Charging Access' if j == 0 else None)

        for j in range(len(communities)):
            ax.plot([charging_off[nearest_idx, 0], communities[j, 0]],
                    [charging_off[nearest_idx, 1], communities[j, 1]],
                    color='green', linewidth=0.4, zorder=1,
                    label='Charging Egress' if j == 0 else None)

    # Charging rate (alpha_ij) / cost per unit charge (pi_ij) labels, offset to the
    # side of the on-off line so they never sit on top of a node marker
    label_texts = []
    if charging_rates is not None and charging_costs is not None:
        for i in range(len(charging_on)):
            on, off = charging_on[i], charging_off[i]
            mid = (on + off) / 2.0

            direction = off - on
            norm = np.linalg.norm(direction)
            unit = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
            perp = np.array([-unit[1], unit[0]])  # rotate 90 degrees

            anchor = mid + perp * label_perp_dist

            # Anchor text so it grows away from the on/off markers (along the same
            # side as the perpendicular offset) rather than centering on the anchor,
            # which is what let wide labels swing back over the station markers.
            ha = 'left' if perp[0] >= 0 else 'right'

            label_texts.append(ax.text(anchor[0], anchor[1] + label_line_gap,
                    fr"$\alpha_{{ij}} = {charging_rates[i]}$",
                    color="red", fontsize=14, ha=ha, va='bottom', zorder=4))
            label_texts.append(ax.text(anchor[0], anchor[1] - label_line_gap,
                    fr"$\pi_{{ij}} = {charging_costs[i]}$",
                    color="purple", fontsize=14, ha=ha, va='top', zorder=4))

    # Pull any label back inside the axes by its actual measured overflow, rather
    # than guessing from the anchor position alone (LaTeX label width varies with
    # the string, so a fixed margin around the anchor can't be trusted to catch it).
    if label_texts:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        pad = edge_margin
        for t in label_texts:
            bbox = t.get_window_extent(renderer=renderer)
            (x0, y0), (x1, y1) = inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
            dx = max(pad - x0, 0) or min(region - pad - x1, 0)
            dy = max(pad - y0, 0) or min(region - pad - y1, 0)
            if dx or dy:
                cx, cy = t.get_position()
                t.set_position((cx + dx, cy + dy))

    ax.set_xlim(0, region)
    ax.set_ylim(0, region)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def generate_network_instance(depot_location, num_communities=None, num_charging_stations=None, region=20,
                              min_distance=1.0, offset=1.0, random_seed=None,
                              charging_locations=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    depot = np.array([depot_location])
    existing_nodes = [depot[0]]

    # Generate community nodes
    communities = []
    if num_communities is not None and num_communities > 0:
        for _ in range(num_communities):
            new_node = generate_node(existing_nodes, region, min_distance)
            communities.append(new_node)
            existing_nodes.append(new_node)
    communities = np.array(communities) if communities else np.array([]).reshape(0, 2)

    # Generate charging station nodes
    charging_on = []
    charging_off = []

    if charging_locations is not None:
        # Use provided exact coordinates
        for coords in charging_locations:
            if len(coords) != 4:
                raise ValueError(f"Each charging location must have 4 values (x_on, y_on, x_off, y_off). Got: {coords}")
            x_on, y_on, x_off, y_off = coords
            charging_on.append([x_on, y_on])
            charging_off.append([x_off, y_off])
            existing_nodes.append([x_on, y_on])
    elif num_charging_stations is not None and num_charging_stations > 0:
        # Generate random charging stations
        for _ in range(num_charging_stations):
            candidate_on, candidate_off = generate_charging_station(existing_nodes, region, min_distance, offset)
            charging_on.append(candidate_on)
            charging_off.append(candidate_off)
            existing_nodes.append(candidate_on)

    charging_on = np.array(charging_on) if charging_on else np.array([]).reshape(0, 2)
    charging_off = np.array(charging_off) if charging_off else np.array([]).reshape(0, 2)

    return depot, communities, charging_on, charging_off


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)



def generate_travel_parameters_euclidean(nodes_file_path: str, links_file_path: str, travel_parameter,
                                         charging_links_file_path: str, open_charging_link_ids):
    nodes_df = pd.read_csv(nodes_file_path)
    links_df = pd.read_csv(links_file_path)
    charging_links_df = pd.read_csv(charging_links_file_path)

    selected_charging_links = charging_links_df[
        charging_links_df['Link ID'].isin(open_charging_link_ids)
    ]
    selected_charging_pairs = set(zip(
        selected_charging_links['from_node ID'].astype(str),
        selected_charging_links['to_node ID'].astype(str)
    ))

    node_coordinates = {
        str(int(row['Node ID'])): (row['X Coordinate'], row['Y Coordinate'])
        for _, row in nodes_df.iterrows()
    }
    node_labels = {
        str(int(row['Node ID'])): row['Label']
        for _, row in nodes_df.iterrows()
    }

    speed = 0.5

    for _, row in links_df.iterrows():
        from_node = str(int(row['from_node ID']))
        to_node = str(int(row['to_node ID']))

        from_label = node_labels.get(from_node, '')
        to_label = node_labels.get(to_node, '')

        # ---------- ENFORCED RULES ----------
        if from_label == 'C+' and to_label == 'C-':
            key = (from_node, to_node)
            if key in selected_charging_pairs:
                travel_parameter[key] = [0.0, 0.0]
            continue

        elif to_label == 'C+' and from_label not in ['C+', 'C-']:
            pass

        elif from_label == 'C-' and to_label not in ['C+', 'C-']:
            pass

        elif 'C+' in [from_label, to_label] or 'C-' in [from_label, to_label]:
            continue

        # ---------- DISTANCE CALCULATION ----------
        lon1, lat1 = node_coordinates[from_node]
        lon2, lat2 = node_coordinates[to_node]
        distance = euclidean_distance(lat1, lon1, lat2, lon2)
        travel_time = distance / speed  # equivalent to 2 × distance

        key = (from_node, to_node)
        travel_parameter[key] = [round(distance, 2), round(travel_time, 2)]

    return travel_parameter


# Function to save the travel parameter dictionary to a file
def save_travel_data(travel_parameter, file_name):
    # Save in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['from_node ID', 'to_node ID', 'Distance (miles)', 'Time (minutes)'])

        for (from_node, to_node), (distance, time) in travel_parameter.items():
            csvwriter.writerow([from_node, to_node, f"{distance:.2f}", f"{time:.2f}"])

    print(f"✅ Travel data saved to: {file_path}")



def load_customer_nodes(network_nodes_path: str) -> set:
    """
    Load and return the set of customer node IDs (Label == 'V') from the network nodes CSV.

    Args:
        network_nodes_path: Path to CSV file containing node information with 'Node ID' and 'Label' columns.

    Returns:
        Set of customer node IDs prefixed with 'i' (e.g., {'i1', 'i2', ...})
    """
    customers = set()
    # Use pandas for robustness against varied CSV formatting
    df = pd.read_csv(network_nodes_path)
    # Expect columns: 'Node ID' and 'Label'
    if 'Node ID' not in df.columns or 'Label' not in df.columns:
        raise ValueError("network_nodes_path must contain 'Node ID' and 'Label' columns")
    for _, row in df.iterrows():
        if str(row['Label']).strip() == 'V':
            node_id = str(row['Node ID']).strip()
            customers.add(node_id)
    return customers


def generate_service_times(num_customers: int,
                           min_val: float, max_val: float,
                           decimal_places: int = 2) -> List[float]:
    """
    Generate service times using a uniform distribution between min_val and max_val.

    Note: mean and std parameters are kept for signature compatibility but are not used
    in uniform generation.

    Args:
        num_customers: Number of service time values to generate
        mean: Unused (kept for compatibility)
        std: Unused (kept for compatibility)
        min_val: Minimum allowed value (lower bound of uniform)
        max_val: Maximum allowed value (upper bound of uniform)
        decimal_places: Number of decimal places to round to

    Returns:
        List of generated service time values
    """
    service_times = [round(random.uniform(min_val, max_val), decimal_places)
                     for _ in range(num_customers)]
    return service_times



def generate_waste_demands(num_customers: int, mean: float, std: float,
                           decimal_places: int = 2) -> List[float]:
    """
    Generate waste pickup demands using a Gaussian (normal) distribution with clipping.

    Each value is drawn from N(mean, std^2) and then automatically clipped to
    ±2 standard deviations from the mean, i.e., [mean - 2*std, mean + 2*std].
    This retains ~95% of the distribution while blocking extreme outliers.

    Args:
        num_customers: Number of demand values to generate
        mean:          Mean of the normal distribution
        std:           Standard deviation of the normal distribution
        decimal_places: Number of decimal places to round to

    Returns:
        List of generated waste demand values
    """
    # Automatically compute clip bounds as ±2 standard deviations
    min_val = mean - 2 * std
    max_val = mean + 2 * std

    waste_demands = []
    for _ in range(num_customers):
        value = random.gauss(mean, std)
        # Clip to [mean - 2*std, mean + 2*std]
        value = max(min_val, min(max_val, value))
        waste_demands.append(round(value, decimal_places))
    return waste_demands


def save_parameters_to_csv(customer_nodes: List[str], values: List[float],
                           output_path: str, column_name: str = "Value"):
    """
    Save parameter values to CSV file with customer node IDs.

    Args:
        customer_nodes: List of customer node IDs (with 'i' prefix)
        values: List of parameter values corresponding to each node
        output_path: Path to output CSV file
        column_name: Name for the value column in the CSV
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Node", column_name])
        for node, value in zip(customer_nodes, values):
            writer.writerow([node, value])



def generate_and_save_service_times(network_nodes_path: str, output_path: str,
                                    min_val: float, max_val: float,
                                    decimal_places: int):
    """
    Read customer nodes from a network nodes CSV file, generate service times,
    and save results to a CSV file.

    Args:
        network_nodes_path: Path to CSV file containing 'Node ID' and 'Label' columns.
                            Only rows where Label == 'V' are treated as customers.
        output_path:        Path to output CSV file for service times.
        min_val:            Minimum allowed service time (lower bound of uniform).
        max_val:            Maximum allowed service time (upper bound of uniform).
        decimal_places:     Number of decimal places to round generated values to.
    """
    # Step 1: Read CSV and filter for customer nodes (Label == 'V')
    df = pd.read_csv(network_nodes_path)
    if 'Node ID' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Input file must contain 'Node ID' and 'Label' columns.")

    # Step 2: Extract node IDs as plain strings without 'i' prefix
    customer_nodes = [
        str(row['Node ID']).strip()
        for _, row in df.iterrows()
        if str(row['Label']).strip() == 'V'
    ]

    if not customer_nodes:
        raise ValueError("No customer nodes (Label == 'V') found in the input file.")

    # Step 3: Generate service times and save to CSV
    service_times = generate_service_times(
        len(customer_nodes), min_val, max_val, decimal_places
    )
    save_parameters_to_csv(customer_nodes, service_times, output_path, "Service Time")
    print(f"Service times for {len(customer_nodes)} customers saved to {output_path}")



def generate_and_save_waste_demands(network_nodes_path: str, output_path: str,
                                    mean: float, std: float,
                                    decimal_places: int):
    """
    Read customer nodes from a network nodes CSV file, generate waste demands
    clipped to ±2 standard deviations, and save results to a CSV file.

    Args:
        network_nodes_path: Path to CSV file with 'Node ID' and 'Label' columns
        output_path:        Path to output CSV file for waste demands
        mean:               Mean of the normal distribution
        std:                Standard deviation of the normal distribution
        decimal_places:     Number of decimal places to round generated values to
    """
    # Step 1: Read CSV and filter for customer nodes (Label == 'V')
    df = pd.read_csv(network_nodes_path)
    if 'Node ID' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Input file must contain 'Node ID' and 'Label' columns.")

    # Step 2: Extract node IDs as plain strings without 'i' prefix
    customer_nodes = [
        str(row['Node ID']).strip()
        for _, row in df.iterrows()
        if str(row['Label']).strip() == 'V'
    ]

    if not customer_nodes:
        raise ValueError("No customer nodes (Label == 'V') found in the input file.")

    # Step 3: Generate waste demands and save to CSV
    waste_demands = generate_waste_demands(
        len(customer_nodes), mean, std, decimal_places
    )
    save_parameters_to_csv(customer_nodes, waste_demands, output_path, "Demand")
    print(f"Waste demands for {len(customer_nodes)} customers saved to {output_path}")


def load_charging_arc_params(params_file, selected_charging_links):
    """
    Reads beta1 and alpha values from the saved CSV file,
    but only loads entries whose link_id is in selected_charging_links.

    Parameters
    ----------
    params_file             : str  - path to the saved charging_arc_params.csv
    selected_charging_links : list - list of charging link IDs to include

    Returns
    -------
    beta1_dict : dict - {arc_key: beta1_value} for selected charging arcs only
    alpha_dict : dict - {arc_key: alpha_value} for selected charging arcs only
    """

    # Convert to a set for fast lookup
    selected_ids = set(selected_charging_links)

    beta1_dict = {}
    alpha_dict = {}

    with open(params_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_id = int(row['link_id'])

            # Only load entries that are in selected_charging_links
            if link_id not in selected_ids:
                continue

            from_node = row['from_node']
            to_node = row['to_node']
            arc_key = ('iC' + from_node + '+', 'iC' + to_node + '-')

            beta1_dict[arc_key] = float(row['beta1'])
            alpha_dict[arc_key] = float(row['alpha'])

    # Verify all selected links were found in the file
    if len(beta1_dict) != len(selected_ids):
        missing = selected_ids - {
            int(row['link_id'])
            for row in csv.DictReader(open(params_file))
        }
        raise ValueError(f"The following link IDs were not found in {params_file}: {missing}")

    # Print summary
    print("=== Loaded Charging Arc Parameters ===")
    print(f"{'Arc Key':<40} {'beta1':>8} {'alpha':>8}")
    print("-" * 58)
    for arc_key in beta1_dict:
        print(f"{str(arc_key):<40} {beta1_dict[arc_key]:>8.3f} {alpha_dict[arc_key]:>8.3f}")

    return beta1_dict, alpha_dict


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
        for v2 in V_nodes[i + 1:]:
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


    "Version I will use in my Districting Problem"
    # import networkx as nx  # add to the imports at the top of the file
    #
    # def generate_complete_network(
    #         nodes_data_file,
    #         links_data_file,
    #         travel_data_file,
    #         charging_links_file,
    #         open_charging_link_ids,
    #         save_modified_nodes,
    #         save_modified_links,
    #         save_modified_travel_data
    # ):
    #     """
    #     Build the complete-network version of an instance.
    #
    #       1. V <-> V  shortest paths over depot/community arcs only (depot excluded as
    #                   an intermediate node); the v1->v2 value is copied to v2->v1.
    #       2. D <-> V  shortest paths over depot/community arcs only; the D->v value
    #                   is copied to v->D.
    #       3. C+ -> C- charging arcs of the open charging links.
    #       4. X -> C+ and C- -> X (X = D or V) access/egress arcs of the open stations.
    #
    #     Charging nodes (and therefore charging arcs) are never part of a V-V or D-V
    #     shortest path. Coordinates follow build_network: lat = X, lon = Y.
    #     """
    #     # Load node, travel and charging-link data
    #     nodes_df = pd.read_csv(nodes_data_file)
    #     travel_data_df = pd.read_csv(travel_data_file)
    #     charging_links_df = pd.read_csv(charging_links_file)
    #
    #     node_labels = dict(zip(nodes_df['Node ID'], nodes_df['Label']))
    #
    #     # Identify customer (V) and depot (D) nodes
    #     V_nodes = nodes_df[nodes_df['Label'] == 'V']['Node ID'].tolist()
    #     D_node = nodes_df[nodes_df['Label'] == 'D']['Node ID'].iloc[0]
    #
    #     # Open charging links (C+ -> C-) and their nodes
    #     allowed_charging_links_df = charging_links_df[
    #         charging_links_df['Link ID'].isin(open_charging_link_ids)
    #     ]
    #     allowed_charging_pairs = list(zip(
    #         allowed_charging_links_df['from_node ID'].astype(int),
    #         allowed_charging_links_df['to_node ID'].astype(int)
    #     ))
    #     allowed_charging_node_ids = set(allowed_charging_links_df['from_node ID']).union(
    #         set(allowed_charging_links_df['to_node ID'])
    #     )
    #
    #     # Same convention as build_network / network_links.csv: lat = X, lon = Y
    #     node_coords = {
    #         row['Node ID']: (row['X Coordinate'], row['Y Coordinate']) for _, row in nodes_df.iterrows()
    #     }
    #
    #     # Direct-arc lookup: (from, to) -> (distance, time)
    #     arc_data = {
    #         (int(r['from_node ID']), int(r['to_node ID'])): (r['Distance (miles)'], r['Time (minutes)'])
    #         for _, r in travel_data_df.iterrows()
    #     }
    #
    #     # Road graph for shortest paths: depot and community nodes ONLY.
    #     # Charging nodes (and therefore charging arcs) are never part of a V-V or D-V path.
    #     G = nx.DiGraph()
    #     G.add_nodes_from([D_node] + V_nodes)
    #     for (i, j), (dist, time) in arc_data.items():
    #         if node_labels[i] in ('D', 'V') and node_labels[j] in ('D', 'V'):
    #             G.add_edge(i, j, weight=dist, time=time)
    #
    #     modified_links = []
    #     modified_travel_data = []
    #
    #     def add_arc(a, b, dist, time):
    #         modified_links.append({
    #             'from_node ID': a,
    #             'to_node ID': b,
    #             'start_lat': node_coords[a][0],
    #             'start_lon': node_coords[a][1],
    #             'end_lat': node_coords[b][0],
    #             'end_lon': node_coords[b][1]
    #         })
    #         modified_travel_data.append({
    #             'from_node ID': a,
    #             'to_node ID': b,
    #             'Distance (miles)': round(dist, 2),
    #             'Time (minutes)': round(time, 2)
    #         })
    #
    #     # 1. V <-> V shortest paths (depot excluded); v1->v2 value copied to v2->v1
    #     G_without_depot = G.subgraph(V_nodes)
    #     for i, v1 in enumerate(V_nodes):
    #         for v2 in V_nodes[i + 1:]:
    #             if nx.has_path(G_without_depot, v1, v2):
    #                 dist = nx.shortest_path_length(G_without_depot, v1, v2, weight='weight')
    #                 time = nx.shortest_path_length(G_without_depot, v1, v2, weight='time')
    #                 for a, b in [(v1, v2), (v2, v1)]:
    #                     add_arc(a, b, dist, time)
    #
    #     # 2. D <-> V shortest paths; D->v value copied to v->D
    #     for v in V_nodes:
    #         if nx.has_path(G, D_node, v):
    #             dist = nx.shortest_path_length(G, D_node, v, weight='weight')
    #             time = nx.shortest_path_length(G, D_node, v, weight='time')
    #             for a, b in [(D_node, v), (v, D_node)]:
    #                 add_arc(a, b, dist, time)
    #
    #     # 3. C+ -> C- only for open charging links
    #     for from_node, to_node in allowed_charging_pairs:
    #         if (from_node, to_node) in arc_data:
    #             add_arc(from_node, to_node, *arc_data[(from_node, to_node)])
    #
    #     # 4. X -> C+ and C- -> X (X = D or V) for open charging nodes
    #     for (from_node, to_node), (dist, time) in arc_data.items():
    #         from_label, to_label = node_labels[from_node], node_labels[to_node]
    #         into_station = (from_label not in ('C+', 'C-') and to_label == 'C+'
    #                         and to_node in allowed_charging_node_ids)
    #         out_of_station = (from_label == 'C-' and from_node in allowed_charging_node_ids
    #                           and to_label not in ('C+', 'C-'))
    #         if into_station or out_of_station:
    #             add_arc(from_node, to_node, dist, time)
    #
    #     # Final node set: depot, customers, and open charging nodes
    #     final_nodes = set(V_nodes + [D_node]).union(allowed_charging_node_ids)
    #     modified_nodes_df = nodes_df[nodes_df['Node ID'].isin(final_nodes)].drop_duplicates()
    #
    #     # Save outputs
    #     pd.DataFrame(modified_links).to_csv(save_modified_links, index=False)
    #     pd.DataFrame(modified_travel_data).to_csv(save_modified_travel_data, index=False)
    #     modified_nodes_df.to_csv(save_modified_nodes, index=False)
    #
    #     print(f"Complete network saved: {len(modified_nodes_df)} nodes, {len(modified_travel_data)} arcs")




if __name__ == "__main__":
    # -------------------- USER CONFIGURATION --------------------
    # Modify these parameters to generate your desired network

    REGION_SIZE = 25                    # Dimension of square region (0 to REGION_SIZE for x and y)
    DEPOT_LOCATION = (12.5, 12.5)      # (x, y) coordinates of the depot
    NUM_COMMUNITIES = 20               # Number of community/customer nodes (randomly placed)
    MIN_DISTANCE = 1.0                 # Minimum distance between nodes (for random generation)
    RANDOM_SEED = 52                # Random seed for reproducibility (None for random)

    # -------- OPTION 1: Random Charging Stations --------
    # Uncomment these lines to randomly generate charging stations:
    # NUM_CHARGING_STATIONS = 4
    # OFFSET = 1.0
    # CHARGING_LOCATIONS = None

    # -------- OPTION 2: Exact Charging Station Coordinates --------
    # Specify exact coordinates for charging stations as (x_on, y_on, x_off, y_off)
    CHARGING_LOCATIONS = [
        (14.5, 14.5, 15, 15),      # Charging station 1: on at (14.5,14.5), off at (15,15)
        (24.0, 5.0, 24.5, 5.5),    # Charging station 2: on at (24,5),      off at (24.5,5.5)
        (5.0, 5.0, 5.5, 5.5),      # Charging station 3: on at (5,5),       off at (5.5,5.5)
        (4.0, 20.0, 4.5, 20.5),    # Charging station 4: on at (4,20),      off at (4.5,20.5)
    ]

    # Charging rate (miles/min) and cost per unit charge ($/min), in the same order as CHARGING_LOCATIONS
    CHARGING_RATES = [1.2, 2.5, 4.8, 2.5]
    CHARGING_COSTS = [0.8, 1.8, 2.4, 1.8]

    NUM_CHARGING_STATIONS = None  # Ignored when CHARGING_LOCATIONS is provided
    OFFSET = None                 # Ignored when CHARGING_LOCATIONS is provided

    # Charging links (Link IDs from charging_links.csv) that are open in this instance
    OPEN_CHARGING_LINK_IDS = [1, 2, 3, 4]

    # Waste demand (normal, clipped to +/- 2 std) and service time (uniform) parameters
    DEMAND_MEAN, DEMAND_STD = 25.0, 4.0
    SERVICE_MIN, SERVICE_MAX = 5.0, 10.0

    # -------------------- OUTPUT PATHS --------------------
    # All files for this instance go to <project root>/synthetic_data/<SUBFOLDER>/.
    # The project root is the parent of synthetic_scripts/, so paths are correct
    # regardless of which folder the script is run from.
    SUBFOLDER    = "synthetic_data"
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "synthetic_data", SUBFOLDER)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The network plot goes to the project's Results/ folder (data files stay in
    # OUTPUT_DIR). The same filename is used on every run, so it is overwritten.
    FIGURES_DIR = os.path.join(PROJECT_ROOT, "Results")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Base network and model parameters
    NODE_FILE          = os.path.join(OUTPUT_DIR, "network_nodes.csv")
    LINK_FILE          = os.path.join(OUTPUT_DIR, "network_links.csv")
    CHARGING_LINK_FILE = os.path.join(OUTPUT_DIR, "charging_links.csv")
    PLOT_FILE          = os.path.join(FIGURES_DIR, "synthetic_network.pdf")
    TRAVEL_DATA_FILE   = os.path.join(OUTPUT_DIR, "travel_data.csv")
    WASTE_DEMAND_FILE  = os.path.join(OUTPUT_DIR, "waste_demand.csv")
    SERVICE_TIME_FILE  = os.path.join(OUTPUT_DIR, "service_time.csv")

    # Complete-network versions
    COMPLETE_NODE_FILE        = os.path.join(OUTPUT_DIR, "complete_nodes.csv")
    COMPLETE_LINK_FILE        = os.path.join(OUTPUT_DIR, "complete_links.csv")
    COMPLETE_TRAVEL_DATA_FILE = os.path.join(OUTPUT_DIR, "complete_travel_data.csv")

    # Seed Python's `random` too (used for demands / service times), so the whole
    # instance is reproducible; generate_network_instance seeds numpy itself.
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # -------------------- GENERATE NETWORK --------------------
    print("Generating network instance...")
    print(f"  Region: {REGION_SIZE}x{REGION_SIZE}")
    print(f"  Depot: {DEPOT_LOCATION}")
    print(f"  Communities: {NUM_COMMUNITIES}")
    if CHARGING_LOCATIONS is not None:
        print(f"  Charging Stations: {len(CHARGING_LOCATIONS)} (custom locations)")
    else:
        print(f"  Charging Stations: {NUM_CHARGING_STATIONS} (random)")

    depot, communities, charging_on, charging_off = generate_network_instance(
        depot_location=DEPOT_LOCATION,
        num_communities=NUM_COMMUNITIES,
        num_charging_stations=NUM_CHARGING_STATIONS,
        region=REGION_SIZE,
        min_distance=MIN_DISTANCE,
        offset=OFFSET,
        random_seed=RANDOM_SEED,
        charging_locations=CHARGING_LOCATIONS
    )

    # -------------------- BUILD NETWORK --------------------
    print("Building network links...")
    nodes, links, charging_links = build_network(depot, communities, charging_on, charging_off)

    # -------------------- SAVE NETWORK --------------------
    print(f"Saving network to {OUTPUT_DIR}...")
    save_network(nodes, links, charging_links, NODE_FILE, LINK_FILE, CHARGING_LINK_FILE)

    # -------------------- PLOT NETWORK --------------------
    print("Generating network plot...")
    plot_network(depot, communities, charging_on, charging_off, REGION_SIZE, PLOT_FILE,
                 charging_rates=CHARGING_RATES, charging_costs=CHARGING_COSTS, label_perp_dist=1.5)

    # ---------- MODEL PARAMETERS (travel, waste demand, service time) ----------
    # Inputs are the network files generated above in this same run.

    # Travel distances and times
    print("Generating travel parameters...")
    travel_parameter = {}
    generate_travel_parameters_euclidean(NODE_FILE, LINK_FILE, travel_parameter,
                                         CHARGING_LINK_FILE, OPEN_CHARGING_LINK_IDS)
    save_travel_data(travel_parameter, TRAVEL_DATA_FILE)   # absolute path -> saved in OUTPUT_DIR

    # Waste demands (normal distribution, clipped to +/- 2 std)
    generate_and_save_waste_demands(network_nodes_path=NODE_FILE,
                                    output_path=WASTE_DEMAND_FILE,
                                    mean=DEMAND_MEAN, std=DEMAND_STD, decimal_places=2)

    # Service times (uniform distribution)
    generate_and_save_service_times(network_nodes_path=NODE_FILE,
                                    output_path=SERVICE_TIME_FILE,
                                    min_val=SERVICE_MIN, max_val=SERVICE_MAX, decimal_places=2)

    # -------------------- COMPLETE NETWORK VERSION --------------------
    # Built from the network, charging-link and travel files generated above.
    print("Generating complete network version...")
    generate_complete_network(NODE_FILE, LINK_FILE, TRAVEL_DATA_FILE,
                              CHARGING_LINK_FILE, OPEN_CHARGING_LINK_IDS,
                              COMPLETE_NODE_FILE, COMPLETE_LINK_FILE, COMPLETE_TRAVEL_DATA_FILE)

    # -------------------- SUMMARY --------------------
    num_cs = len(CHARGING_LOCATIONS) if CHARGING_LOCATIONS is not None else NUM_CHARGING_STATIONS
    print("\n" + "=" * 60)
    print("NETWORK AND PARAMETER GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total nodes: {len(nodes)}")
    print(f"  - Depot: 1")
    print(f"  - Communities: {NUM_COMMUNITIES}")
    print(f"  - Charging stations: {num_cs} pairs ({num_cs * 2} nodes)")
    print(f"Total links: {len(links)}")
    print(f"Travel arcs saved: {len(travel_parameter)}")
    print(f"\nFiles saved in {OUTPUT_DIR}:")
    for f in (NODE_FILE, LINK_FILE, CHARGING_LINK_FILE,
              TRAVEL_DATA_FILE, WASTE_DEMAND_FILE, SERVICE_TIME_FILE,
              COMPLETE_NODE_FILE, COMPLETE_LINK_FILE, COMPLETE_TRAVEL_DATA_FILE):
        print(f"  - {os.path.basename(f)}")
    print(f"Network plot saved to {PLOT_FILE}")
    print("=" * 60)