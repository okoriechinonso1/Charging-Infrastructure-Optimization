import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import alpha
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os


def plot_networkx_graph(evrp_main_data, arc_travel, nodes_data):
    # Load CSVs
    nodes_data = pd.read_csv(nodes_data)
    evrp_data = pd.read_csv(evrp_main_data)

    # === Helper to normalize node IDs for internal graph use ===
    def clean_node_id_for_graph(row):
        label = row["Label"]
        if label in ['C+', 'C-']:
            return f'iC{int(row["Node ID"])}'  # Drop +/-
        else:
            return f'i{int(row["Node ID"])}'

    # Build node positions and colors with cleaned IDs
    node_positions = {}
    node_colors = {}
    depot_id = None

    for _, row in nodes_data.iterrows():
        node_id = clean_node_id_for_graph(row)
        position = (row["X Coordinate"], row["Y Coordinate"])
        label = row['Label']

        node_positions[node_id] = position

        if label == 'V' or label == 'H':
            node_colors[node_id] = 'white'
        elif label in ['C+', 'C-']:
            node_colors[node_id] = 'green'
        elif label == 'D':
            node_colors[node_id] = 'orange'
            depot_id = node_id  # Save for return later
        else:
            node_colors[node_id] = 'white'

    # === Clean arc_travel to match cleaned node IDs ===
    def clean_node_id(node):
        return node.replace('+', '').replace('-', '').replace('h', '')

    updated_arc_travel = []
    for u, v, vehicle in arc_travel:
        u_clean = clean_node_id(u)
        v_clean = clean_node_id(v)
        updated_arc_travel.append((u_clean, v_clean, vehicle))

    # === Create graph ===
    G = nx.DiGraph()
    G.add_nodes_from(node_positions.keys())

    for u, v, vehicle in updated_arc_travel:
        try:
            from_node = int(''.join(filter(str.isdigit, u.lstrip('i'))))
            to_node = int(''.join(filter(str.isdigit, v.lstrip('i'))))
            travel_row = evrp_data[
                (evrp_data['from_node ID'] == from_node) &
                (evrp_data['to_node ID'] == to_node)
            ]
            if not travel_row.empty:
                travel_cost = travel_row['Distance (miles)'].values[0]
                G.add_edge(u, v, weight=travel_cost, vehicle=vehicle)
        except ValueError:
            print(f"Skipping invalid nodes: {u}, {v}")

    # === Vehicle color mapping ===
    vehicle_colors = ['red', 'blue', 'purple', 'orange', 'brown']
    unique_vehicles = sorted({path[2] for path in updated_arc_travel})
    vehicle_color_map = {v: vehicle_colors[i % len(vehicle_colors)] for i, v in enumerate(unique_vehicles)}

    # === Filter node colors to only used nodes ===
    visited_nodes = {u for u, v, _ in updated_arc_travel} | {v for u, v, _ in updated_arc_travel}
    node_colors = {node: color for node, color in node_colors.items() if node in visited_nodes}

    # === Plotting ===
    plt.figure(figsize=(10, 8))

    # Draw edges per vehicle
    for vehicle, color in vehicle_color_map.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('vehicle') == vehicle]
        nx.draw_networkx_edges(
            G, node_positions, edgelist=edges, edge_color=color, width=2,
            arrows=True, arrowstyle='-|>', arrowsize=12
        )

    # Draw nodes
    for node, color in node_colors.items():
        nx.draw_networkx_nodes(
            G, node_positions, nodelist=[node], node_size=500,
            node_color=color, edgecolors='black', linewidths=2
        )

    # Draw node labels (remove 'i', and '+'/'-' for charging nodes)
    labels = {}
    for node in G.nodes():
        if node.startswith("iC"):  # Charging nodes, show as C24
            labels[node] = node[1:]  # Remove 'i'
        elif node.startswith("i"):
            labels[node] = node[1:]  # Remove 'i' for others
        else:
            labels[node] = node

    nx.draw_networkx_labels(G, node_positions, labels, font_size=10)

    # Add legend for routes
    legend_elements = [
        Line2D([0], [0], color=vehicle_color_map[v], lw=2, label=f'Route {i+1}')
        for i, v in enumerate(unique_vehicles)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.axis('on')
    plt.tight_layout()

    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/VehicleRoutes.svg", format='svg')

    # === Return depot node as ['i5+', 'i5-'] format ===
    if depot_id is None:
        raise ValueError("No depot node found!")

    return [f"{depot_id}+", f"{depot_id}-"]


def plot_single_route_network(evrp_main_data, arc_travel, nodes_data):

    # === Load CSVs ===
    nodes_data = pd.read_csv(nodes_data)
    evrp_data = pd.read_csv(evrp_main_data)

    # === Normalize node ID ===
    def clean_node_id_for_graph(row):
        label = row["Label"]
        if label in ['C+', 'C-']:
            return f'iC{int(row["Node ID"])}'
        return f'i{int(row["Node ID"])}'

    def normalize_node(node):
        return node.replace('+', '').replace('-', '').replace('h', '')

    # === Build node positions ===
    node_positions = {}
    node_colors = {}
    depot_id = None

    for _, row in nodes_data.iterrows():
        node_id = clean_node_id_for_graph(row)
        node_positions[node_id] = (row["X Coordinate"], row["Y Coordinate"])

        label = row["Label"]

        if label in ['V', 'H']:
            node_colors[node_id] = 'white'
        elif label in ['C+', 'C-']:
            node_colors[node_id] = 'green'
        elif label == 'D':
            node_colors[node_id] = 'orange'
            depot_id = node_id
        else:
            node_colors[node_id] = 'white'

    # === Clean arc list ===
    cleaned_arcs = []
    visited_nodes = set()

    for u, v in arc_travel:
        u_clean = normalize_node(u)
        v_clean = normalize_node(v)

        cleaned_arcs.append((u_clean, v_clean))
        visited_nodes.update([u_clean, v_clean])

    # === Filter nodes ===
    node_positions = {n: p for n, p in node_positions.items() if n in visited_nodes}
    node_colors = {n: c for n, c in node_colors.items() if n in visited_nodes}

    # === Build graph ===
    G = nx.DiGraph()
    G.add_nodes_from(visited_nodes)

    for u, v in cleaned_arcs:
        try:
            from_node = int(''.join(filter(str.isdigit, u)))
            to_node = int(''.join(filter(str.isdigit, v)))

            travel_row = evrp_data[
                (evrp_data['from_node ID'] == from_node) &
                (evrp_data['to_node ID'] == to_node)
            ]

            if not travel_row.empty:
                dist = travel_row['Distance (miles)'].values[0]
                G.add_edge(u, v, weight=dist)
            else:
                # still add edge (important for visualization)
                G.add_edge(u, v)

        except:
            print(f"Skipping edge: {u} → {v}")

    # === Plot ===
    plt.figure(figsize=(10, 8))

    # Draw edges (single route)
    nx.draw_networkx_edges(
        G, node_positions,
        edgelist=cleaned_arcs,
        edge_color='black',
        width=2,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=12
    )

    # Draw nodes
    for node, color in node_colors.items():
        nx.draw_networkx_nodes(
            G, node_positions,
            nodelist=[node],
            node_size=500,
            node_color=color,
            edgecolors='black',
            linewidths=1.5
        )

    # === Labels (preserve charging +/-) ===
    labels = {}

    for node in G.nodes():
        num = ''.join(filter(str.isdigit, node))

        has_plus = any(f"{node}+" in arc for arc in arc_travel for node in arc)
        has_minus = any(f"{node}-" in arc for arc in arc_travel for node in arc)

        if node.startswith('iC'):
            if has_plus:
                labels[node] = f"{num}+"
            elif has_minus:
                labels[node] = f"{num}-"
            else:
                labels[node] = num
        else:
            labels[node] = num

    nx.draw_networkx_labels(G, node_positions, labels, font_size=10)

    plt.title("Single Route Visualization")
    plt.axis('on')
    plt.tight_layout()

    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/SingleRoute.svg", format='svg', dpi=300)

    # === Return depot ===
    if depot_id is None:
        raise ValueError("No depot node found!")

    return [f"{depot_id}+", f"{depot_id}-"]




def generate_plots(arc_travel, time_level, SoC_level, depot_nodes, max_soc, max_time):
    def extract_tour_path(arcs, start, end, vehicle_id):
        current_node = start
        tour_path = [current_node]

        while current_node != end:
            for arc in arcs:
                if arc[0] == current_node and arc[2] == vehicle_id:
                    tour_path.append(arc[1])
                    current_node = arc[1]
                    break

        return tour_path

    # Extracting unique vehicle IDs
    vehicle_ids = set(arc[2] for arc in arc_travel)

    start_depot = depot_nodes[0]
    end_depot = depot_nodes[1]

    # Dictionary to store tour paths for each vehicle
    vehicle_tour_paths = {}

    for vehicle_id in vehicle_ids:
        tour_path = extract_tour_path(arc_travel, start_depot, end_depot, vehicle_id)
        vehicle_tour_paths[vehicle_id] = tour_path
        print(f"Tour path for vehicle {vehicle_id}: {tour_path}")

    for vehicle_id, tour_path in vehicle_tour_paths.items():
        # Strip 'i' prefix from nodes in the tour path
        stripped_tour_path = [node.lstrip('i') for node in tour_path]

        time_values = [time_level.get((node, vehicle_id), 0) for node in tour_path]
        SoC_values = [SoC_level.get((node, vehicle_id), 0) for node in tour_path]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Vehicle Battery Levels SoC
        color = 'tab:green'
        ax1.set_xlabel('Node number', fontsize=14)
        ax1.set_ylabel('SoC Level', fontsize=14, color=color)
        ln1 = ax1.plot(stripped_tour_path, SoC_values, color=color, marker='o', linestyle='-', label='SoC Level')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, max_soc)  # Set y-axis limits for SoC level

        # Plot Node Visit Time
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Node Visit Time', fontsize=14, color=color)
        ln2 = ax2.plot(stripped_tour_path, time_values, color=color, marker='o', linestyle='-', label='Node Visit Time')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, max_time)  # Set y-axis limits for node visit time

        # Title and layout adjustments
        plt.title(f'Performance Metrics for Vehicle {vehicle_id}')

        # Combine all legends into one
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc='upper left')

        fig.tight_layout()

        # Rotate x-axis labels
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Save and show plot
        plt.savefig(f'Figures/Performance Metrics_Vehicle {vehicle_id}.svg')

    return vehicle_tour_paths


def generate_plots_single_vehicle(
    arc_travel,
    time_level,
    SoC_level,
    depot_nodes,
    max_soc,
    max_time
):

    # === Extract tour path ===
    def extract_tour_path(arcs, start, end):
        current_node = start
        tour_path = [current_node]

        visited = set()

        while current_node != end:
            found = False

            for u, v in arcs:
                if u == current_node and (u, v) not in visited:
                    tour_path.append(v)
                    visited.add((u, v))
                    current_node = v
                    found = True
                    break

            if not found:
                print("⚠️ Path extraction stopped early — broken route")
                break

        return tour_path

    start_depot = depot_nodes[0]
    end_depot = depot_nodes[1]

    # === Extract path ===
    tour_path = extract_tour_path(arc_travel, start_depot, end_depot)
    print(f"Tour path: {tour_path}")

    # === Clean labels ===
    stripped_tour_path = [node.lstrip('i') for node in tour_path]

    # === Extract values ===
    time_values = [time_level.get(node, 0) for node in tour_path]
    SoC_values = [SoC_level.get(node, 0) for node in tour_path]

    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # SoC plot
    color = 'tab:green'
    ax1.set_xlabel('Node number', fontsize=14)
    ax1.set_ylabel('SoC Level', fontsize=14, color=color)

    ln1 = ax1.plot(
        stripped_tour_path,
        SoC_values,
        color=color,
        marker='o',
        linestyle='-',
        label='SoC Level'
    )

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max_soc)

    # Time plot (secondary axis)
    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Node Visit Time', fontsize=14, color=color)

    ln2 = ax2.plot(
        stripped_tour_path,
        time_values,
        color=color,
        marker='o',
        linestyle='-',
        label='Node Visit Time'
    )

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max_time)

    # Title
    plt.title('Performance Metrics for Single Vehicle')

    # Legend
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper left')

    # Layout
    fig.tight_layout()
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Save
    os.makedirs("Figures", exist_ok=True)
    plt.savefig('Figures/Performance_Metrics_Single_Vehicle.svg', dpi=300)

    return tour_path










