import math
import pandas as pd
import numpy as np
import os
import random
import csv
from typing import List


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)


def generate_travel_parameters(nodes_file_path: str, links_file_path: str, travel_parameter, charging_links_file_path: str,
                               open_charging_link_ids):
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

    for _, row in links_df.iterrows():
        from_node = str(int(row['from_node ID']))
        to_node = str(int(row['to_node ID']))

        #if from_node not in node_coordinates or to_node not in node_coordinates:
            #continue

        from_label = node_labels.get(from_node, '')
        to_label = node_labels.get(to_node, '')

        # ---------- ENFORCED RULES ----------
        # (1) Allow only C+ → C- and cost must be 0.0
        if from_label == 'C+' and to_label == 'C-':
            key = (from_node, to_node)
            if key in selected_charging_pairs:
                travel_parameter[key] = [0.0, 0.0]
            continue  # Skip further processing of this arc

        # (2) Allow arcs TO C+ from any node (but not FROM C+)
        elif to_label == 'C+' and from_label not in ['C+', 'C-']:
            pass  # allowed

        # (3) Allow arcs FROM C- to any node (but not TO C-)
        elif from_label == 'C-' and to_label not in ['C+', 'C-']:
            pass  # allowed

        # (4) All other arcs involving C+ or C- are not allowed
        elif 'C+' in [from_label, to_label] or 'C-' in [from_label, to_label]:
            continue  # skip this arc

        # ---------- DISTANCE CALCULATION ----------
        lon1, lat1 = node_coordinates[from_node]
        lon2, lat2 = node_coordinates[to_node]
        base_distance = haversine_distance(lat1, lon1, lat2, lon2)
        scaled_distance = base_distance  * 7

        # rand_speed_mph = np.random.uniform(20, 25)
        # speed_mpm = rand_speed_mph / 60
        speed_mph = 25
        speed_mpm = speed_mph / 60
        travel_time_minutes = scaled_distance / speed_mpm

        key = (from_node, to_node)
        travel_parameter[key] = [round(scaled_distance, 2), round(travel_time_minutes, 2)]

    return travel_parameter


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



def generate_charging_arc_params(charging_links_file, selected_charging_links,
                                  params_output_file="scripts/data_instance/charging_arc_params.csv"):
    """
    Generates heterogeneous beta1 and alpha parameters for selected charging links,
    prints a summary table, and saves the results to a CSV file.

    Parameters
    ----------
    charging_links_file    : str  - path to charging_links.csv
    selected_charging_links: list - ordered list of charging link IDs to include
    params_output_file     : str  - path to save the output CSV (default provided)

    Returns
    -------
    beta1_dict : dict - {arc_key: beta1_value} for each selected charging arc
    alpha_dict : dict - {arc_key: alpha_value} for each selected charging arc
    """

    # Step 1: Read charging_links.csv and build a lookup by Link ID
    charging_link_lookup = {}
    with open(charging_links_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            link_id   = int(row[0])
            from_node = str(int(float(row[1])))
            to_node   = str(int(float(row[2])))
            charging_link_lookup[link_id] = (from_node, to_node)

    # Step 2: Populate beta1_dict and alpha_dict for selected links
    # beta1 range : 1.5 → 4.0  (charging cost per hour per arc)
    # alpha range : 0.1 → 1.2  (battery recharge rate per arc)
    beta1_dict = {}
    alpha_dict = {}
    n = len(selected_charging_links)

    for idx, link_id in enumerate(selected_charging_links):

        if link_id not in charging_link_lookup:
            raise ValueError(f"Link ID {link_id} not found in charging_links.csv")

        from_node, to_node = charging_link_lookup[link_id]
        arc_key = ('iC' + from_node + '+', 'iC' + to_node + '-')

        # t is strictly increasing: 0/(n-1), 1/(n-1), ... (n-1)/(n-1)
        t = idx / max(n - 1, 1)

        # alpha: LINEAR in t → range [0.1, 1.2]
        alpha_val = 0.1 + 1.1 * t

        # beta1: SQUARE ROOT of t → range [1.5, 4.0]
        beta1_val = 1.5 + 2.5 * math.sqrt(t)

        beta1_dict[arc_key] = round(beta1_val, 2)
        alpha_dict[arc_key] = round(alpha_val, 2)

    # Step 3: Print summary table
    print("=== Charging Arc Parameters ===")
    print(f"{'Arc Key':<40} {'beta1':>8} {'alpha':>8}")
    print("-" * 58)
    for arc_key in beta1_dict:
        print(f"{str(arc_key):<40} {beta1_dict[arc_key]:>8.3f} {alpha_dict[arc_key]:>8.3f}")

    # Step 4: Save to CSV
    with open(params_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['link_id', 'from_node', 'to_node', 'arc_key', 'beta1', 'alpha'])
        for idx, link_id in enumerate(selected_charging_links):
            from_node, to_node = charging_link_lookup[link_id]
            arc_key = ('iC' + from_node + '+', 'iC' + to_node + '-')
            writer.writerow([
                link_id,
                from_node,
                to_node,
                str(arc_key),
                beta1_dict[arc_key],
                alpha_dict[arc_key]
            ])

    print(f"\nCharging arc parameters saved to: {params_output_file}")

    return beta1_dict, alpha_dict



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
            to_node   = row['to_node']
            arc_key   = ('iC' + from_node + '+', 'iC' + to_node + '-')

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