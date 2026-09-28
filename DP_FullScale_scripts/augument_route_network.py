import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import numpy as np
import math

# Route figures go inside each period folder: <period folder>/Figures/
# (not fleet-size specific, since CG generates the same routes for every
# fleet size). FIGURES_DIR (<project root>/Figures) is only used by the
# standalone plot_network_with_* functions, which are not tied to a period folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures")


def _figures_folder(folder, id):
    """Figure output folder for one period (folder). id (fleet size) is
    accepted for call compatibility but no longer used in the folder name."""
    out_dir = os.path.join(folder, "Figures")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_depot_and_charging_arcs(nodes_file, charging_links_file):
    """
    Read depot node IDs from a nodes file (Label 'D') and the charging-arc
    numbering from the period's charging_links.csv:
    (from_node ID, to_node ID) -> Link ID. Link IDs do not follow node-file
    order in the real-world data, so charging_links.csv is the only source.
    Returns (depot_ids, charging_arc_id_map) with node IDs as strings.
    """
    nodes_df = pd.read_csv(nodes_file)
    depot_ids = [str(int(n)) for n in nodes_df.loc[nodes_df["Label"] == "D", "Node ID"]]

    charging_links_df = pd.read_csv(charging_links_file)
    charging_arc_id_map = {
        (str(int(row["from_node ID"])), str(int(row["to_node ID"]))): int(row["Link ID"])
        for _, row in charging_links_df.iterrows()
    }

    return depot_ids, charging_arc_id_map


def _offset_overlapping_depot_nodes(node_positions, node_types, depot_id, depot_base_id):
    """Spread depot variants slightly so D, D+, and D- do not overlap visually."""
    if not depot_id or depot_base_id is None:
        return node_positions

    if depot_id not in node_positions:
        return node_positions

    x_vals = [p[0] for p in node_positions.values()]
    y_vals = [p[1] for p in node_positions.values()]
    dx = max((max(x_vals) - min(x_vals)) * 0.03, 0.005)
    dy = max((max(y_vals) - min(y_vals)) * 0.03, 0.005)

    x0, y0 = node_positions[depot_id]

    offsets = {
        'D': (0.0, 0.0),
        'D+': (-dx, -dy),
        'D-': (dx, dy),
    }

    for node, node_type in node_types.items():
        if not str(node).startswith(f"i{depot_base_id}"):
            continue
        if node_type in offsets:
            off_x, off_y = offsets[node_type]
            node_positions[node] = (x0 + off_x, y0 + off_y)

    return node_positions


def plot_routes(evrp_main_data, nodes_data, routes, folder, id):

    nodes_data = pd.read_csv(nodes_data)
    evrp_data = pd.read_csv(evrp_main_data)

    # === Normalize node (remove + / -) ===
    def normalize_node(node):
        return str(node).replace('+', '').replace('-', '')

    # === Clean node ID from dataset ===
    def clean_node_id_for_graph(row):
        if row["Label"] in ['C+', 'C-']:
            return f'iC{int(row["Node ID"])}'
        return f'i{int(row["Node ID"])}'

    # === Build node info ===
    node_positions = {}
    node_types = {}
    depot_id = None

    for _, row in nodes_data.iterrows():
        node_id = clean_node_id_for_graph(row)
        node_positions[node_id] = (row["X Coordinate"], row["Y Coordinate"])
        node_types[node_id] = row["Label"]

        if row["Label"] == 'D':
            depot_id = node_id

    # === Convert routes → edges ===
    edge_list = []
    visited_nodes = set()

    for vehicle_id, route in enumerate(routes):

        # 🔹 Ensure route is list of strings
        route = [str(n) for n in route]

        for i in range(len(route) - 1):
            u_raw = route[i]
            v_raw = route[i + 1]

            u = normalize_node(u_raw)
            v = normalize_node(v_raw)

            edge_list.append((u, v, vehicle_id))
            visited_nodes.update([u, v])

    # === Filter nodes strictly ===
    node_positions = {n: p for n, p in node_positions.items() if n in visited_nodes}
    node_types = {n: t for n, t in node_types.items() if n in visited_nodes}

    # === Build graph ===
    G = nx.DiGraph()
    G.add_nodes_from(visited_nodes)

    for u, v, vehicle in edge_list:
        try:
            from_node = int(''.join(filter(str.isdigit, u)))
            to_node = int(''.join(filter(str.isdigit, v)))

            travel_row = evrp_data[
                (evrp_data['from_node ID'] == from_node) &
                (evrp_data['to_node ID'] == to_node)
            ]

            if not travel_row.empty:
                dist = travel_row['Distance (miles)'].values[0]
                G.add_edge(u, v, weight=dist, vehicle=vehicle)

        except:
            continue

    # === Node colors ===
    node_colors = {}
    for n, t in node_types.items():
        if t in ['C+', 'C-']:
            node_colors[n] = 'green'
        elif t == 'D':
            node_colors[n] = 'orange'
        else:
            node_colors[n] = 'white'

    # === Scalable colormap ===
    unique_vehicles = sorted(set(v for _, _, v in edge_list))
    cmap = plt.get_cmap('tab20', len(unique_vehicles))

    vehicle_color_map = {
        v: cmap(i) for i, v in enumerate(unique_vehicles)
    }

    # === Plot ===
    plt.figure(figsize=(12, 10))

    # Draw edges
    for vehicle, color in vehicle_color_map.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['vehicle'] == vehicle]

        nx.draw_networkx_edges(
            G, node_positions,
            edgelist=edges,
            edge_color=[color],
            width=2,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10
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

    # === Labels ===
    labels = {}
    for node in G.nodes():
        num = ''.join(filter(str.isdigit, node))
        labels[node] = num

    nx.draw_networkx_labels(G, node_positions, labels, font_size=9)

    # === Legend ===
    legend_elements = [
        Line2D([0], [0], color=vehicle_color_map[v], lw=2, label=f'Route {i+1}')
        for i, v in enumerate(unique_vehicles)
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.axis('on')
    plt.tight_layout()

    plt.savefig(os.path.join(_figures_folder(folder, id), "VehicleRoutes.svg"), format='svg', dpi=300)
    plt.close()

    if depot_id is None:
        raise ValueError("No depot node found!")

    return [f"{depot_id}+", f"{depot_id}-"]


def export_route_data(
    selected_routes,
    node_file,
    service_time_file,
    waste_demand_file,
    travel_data_file,
    travel_links_file,
    output_folder="CG_route_exports"
):
    # Safety check
    if selected_routes is None:
        raise ValueError("selected_routes is None. Ensure solve() returns selected routes.")

    if len(selected_routes) == 0:
        print("No routes to export.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 🔹 Load datasets
    nodes_df = pd.read_csv(node_file)
    service_df = pd.read_csv(service_time_file)
    demand_df = pd.read_csv(waste_demand_file)
    travel_df = pd.read_csv(travel_data_file)
    links_df = pd.read_csv(travel_links_file)

    for idx, route in enumerate(selected_routes, start=1):

        try:
            node_ids = [int(str(n).replace('i', '')) for n in route]
        except Exception as e:
            print(f"Skipping route {idx} due to parsing error: {e}")
            continue

        # 🔹 Remove duplicates (preserve order)
        node_ids_unique = list(dict.fromkeys(node_ids))

        # =========================
        # 🔹 NODE-LEVEL EXPORT (KEEP DEPOT)
        # =========================
        nodes_sub = nodes_df[nodes_df["Node ID"].isin(node_ids_unique)]
        service_sub = service_df[service_df["Node"].isin(node_ids_unique)]
        demand_sub = demand_df[demand_df["Node"].isin(node_ids_unique)]

        nodes_sub.to_csv(os.path.join(output_folder, f"route_{idx}_nodes.csv"), index=False)
        service_sub.to_csv(os.path.join(output_folder, f"route_{idx}_service_time.csv"), index=False)
        demand_sub.to_csv(os.path.join(output_folder, f"route_{idx}_waste_demand.csv"), index=False)

        # =========================
        # 🔹 ARC EXTRACTION
        # =========================
        arc_list = []
        for i in range(len(node_ids) - 1):
            from_node = node_ids[i]
            to_node = node_ids[i + 1]
            arc_list.append((from_node, to_node))

        arcs_df = pd.DataFrame(arc_list, columns=["from_node ID", "to_node ID"])

        # =========================
        # 🔹 TRAVEL DATA FILTER
        # =========================
        travel_sub = travel_df.merge(
            arcs_df,
            on=["from_node ID", "to_node ID"],
            how="inner"
        )

        travel_sub.to_csv(os.path.join(output_folder, f"route_{idx}_travel_data.csv"), index=False)

        # =========================
        # 🔹 TRAVEL LINKS FILTER
        # =========================
        links_sub = links_df.merge(
            arcs_df,
            on=["from_node ID", "to_node ID"],
            how="inner"
        )

        links_sub.to_csv(os.path.join(output_folder, f"route_{idx}_travel_links.csv"), index=False)

        # =========================
        print(f"Route {idx}:")
        print(f"  Nodes exported: {len(node_ids_unique)}")
        print(f"  Arcs exported: {len(arc_list)}")



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


def augment_route_with_charging(
        route_nodes_file,
        route_links_file,
        route_travel_file,
        route_service_time_file,
        route_waste_demand_file,
        charging_links_file,
        selected_charging_link_ids,
        output_folder="augmented_routes_for_charging",
        route_id=1
):
    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # Load data
    # =========================
    nodes_df = pd.read_csv(route_nodes_file)
    links_df = pd.read_csv(route_links_file)
    travel_df = pd.read_csv(route_travel_file)
    charging_df = pd.read_csv(charging_links_file)
    service_df = pd.read_csv(route_service_time_file)
    demand_df = pd.read_csv(route_waste_demand_file)

    charging_df = charging_df[charging_df["Link ID"].isin(selected_charging_link_ids)]

    # =========================
    # Node coordinates lookup
    # =========================
    node_coordinates = {}

    for _, row in nodes_df.iterrows():
        node_coordinates[row["Node ID"]] = (
            row["X Coordinate"],  # lon
            row["Y Coordinate"]   # lat
        )

    # =========================
    # Identify node sets
    # =========================
    depot_nodes = nodes_df[nodes_df["Label"] == "D"]["Node ID"].tolist()
    customer_nodes = nodes_df[nodes_df["Label"] == "V"]["Node ID"].tolist()

    # =========================
    # Add charging nodes
    # =========================
    charging_nodes = []

    for _, row in charging_df.iterrows():
        c_plus = row["from_node ID"]
        c_minus = row["to_node ID"]

        # store coordinates
        node_coordinates[c_plus] = (row["start_lon"], row["start_lat"])
        node_coordinates[c_minus] = (row["end_lon"], row["end_lat"])

        charging_nodes.append([c_plus, row["start_lon"], row["start_lat"], "C+"])
        charging_nodes.append([c_minus, row["end_lon"], row["end_lat"], "C-"])

    charging_nodes_df = pd.DataFrame(
        charging_nodes,
        columns=["Node ID", "X Coordinate", "Y Coordinate", "Label"]
    )

    augmented_nodes = pd.concat([nodes_df, charging_nodes_df], ignore_index=True)

    # =========================
    # Build new links & travel
    # =========================
    new_links = []
    new_travel = []

    def compute_travel(from_node, to_node, is_charging_arc=False):

        if is_charging_arc:
            return 0.0, 0.0

        lon1, lat1 = node_coordinates[from_node]
        lon2, lat2 = node_coordinates[to_node]

        base_distance = haversine_distance(lat1, lon1, lat2, lon2)
        scaled_distance = base_distance * 6.5

        # rand_speed_mph = np.random.uniform(20, 25)
        speed_mph = 25
        speed_mpm = speed_mph / 60

        travel_time = scaled_distance / speed_mpm

        return round(scaled_distance, 2), round(travel_time, 2)

    def add_arc(from_id, to_id, is_charging_arc=False):

        lon1, lat1 = node_coordinates[from_id]
        lon2, lat2 = node_coordinates[to_id]

        new_links.append([
            from_id, to_id,
            lat1, lon1,
            lat2, lon2
        ])

        dist, time = compute_travel(from_id, to_id, is_charging_arc)

        new_travel.append([
            from_id, to_id,
            dist,
            time
        ])

    # =========================
    # Generate arcs
    # =========================
    for _, row in charging_df.iterrows():

        c_plus = row["from_node ID"]
        c_minus = row["to_node ID"]

        # 🔹 Charging arc (ZERO COST)
        add_arc(c_plus, c_minus, is_charging_arc=True)

        # 🔹 Depot → C+
        for d in depot_nodes:
            add_arc(d, c_plus)

        # 🔹 Customer → C+
        for v in customer_nodes:
            add_arc(v, c_plus)

        # 🔹 C- → Customer
        for v in customer_nodes:
            add_arc(c_minus, v)

        # 🔹 C- → Depot
        for d in depot_nodes:
            add_arc(c_minus, d)

    # =========================
    # Convert to DataFrames
    # =========================
    new_links_df = pd.DataFrame(
        new_links,
        columns=["from_node ID", "to_node ID", "start_lat", "start_lon", "end_lat", "end_lon"]
    )

    new_travel_df = pd.DataFrame(
        new_travel,
        columns=["from_node ID", "to_node ID", "Distance (miles)", "Time (minutes)"]
    )

    # =========================
    # Merge with original
    # =========================
    augmented_links = pd.concat([links_df, new_links_df], ignore_index=True)
    augmented_travel = pd.concat([travel_df, new_travel_df], ignore_index=True)

    # =========================
    # Save
    # =========================
    augmented_nodes.to_csv(
        os.path.join(output_folder, f"route_{route_id}_aug_nodes.csv"),
        index=False
    )

    augmented_links.to_csv(
        os.path.join(output_folder, f"route_{route_id}_aug_links.csv"),
        index=False
    )

    augmented_travel.to_csv(
        os.path.join(output_folder, f"route_{route_id}_aug_travel.csv"),
        index=False
    )

    service_df.to_csv(
        os.path.join(output_folder, f"route_{route_id}_aug_service_time.csv"),
        index=False
    )

    demand_df.to_csv(
        os.path.join(output_folder, f"route_{route_id}_aug_waste_demand.csv"),
        index=False
    )

    print(f"Route {route_id}: Route charging network successfully constructed.")


def visualize_augmented_network(
    route_nodes_file,
    route_travel_file,
    aug_nodes_file,
    aug_links_file,
    aug_travel_file,
    folder,
    id,
    route_id=1,
    charging_links_file=None,
):
    """
    charging_links_file: the period's charging_links.csv. The charging arcs
    (C+ -> C-) are labelled with their Link ID from this file.
    """

    # =========================
    # LOAD DATA
    # =========================
    route_nodes = pd.read_csv(route_nodes_file)
    route_travel = pd.read_csv(route_travel_file)

    aug_nodes = pd.read_csv(aug_nodes_file)
    aug_links = pd.read_csv(aug_links_file)
    aug_travel = pd.read_csv(aug_travel_file)

    # =========================
    # NODE POSITIONS
    # =========================
    def get_positions(df):
        return {
            str(int(float(row["Node ID"]))): (row["X Coordinate"], row["Y Coordinate"])
            for _, row in df.iterrows()
        }

    # route_pos = get_positions(route_nodes)
    # aug_pos = get_positions(aug_nodes)
    # Use ONE unified position mapping
    global_pos = get_positions(aug_nodes)
    route_pos = global_pos
    aug_pos = global_pos

    # =========================
    # BUILD ORIGINAL GRAPH
    # =========================
    G_route = nx.DiGraph()

    for _, row in route_travel.iterrows():
        u = str(int(float(row["from_node ID"])))
        v = str(int(float(row["to_node ID"])))
        dist = row["Distance (miles)"]

        G_route.add_edge(u, v, weight=dist)

    # =========================
    # BUILD AUGMENTED GRAPH
    # =========================
    G_aug = nx.DiGraph()

    aug_travel_dict = {
        (str(int(float(row["from_node ID"]))), str(int(float(row["to_node ID"])))): row["Distance (miles)"]
        for _, row in aug_travel.iterrows()
    }

    for _, row in aug_links.iterrows():
        u = str(int(float(row["from_node ID"])))
        v = str(int(float(row["to_node ID"])))

        dist = aug_travel_dict.get((u, v), 0)
        G_aug.add_edge(u, v, weight=dist)

    # =========================
    # NODE TYPE CLASSIFICATION
    # =========================
    node_types = {
        str(int(float(row["Node ID"]))): row["Label"]
        for _, row in aug_nodes.iterrows()
    }

    charging_on_nodes = [n for n, t in node_types.items() if t == "C+"]
    charging_off_nodes = [n for n, t in node_types.items() if t == "C-"]
    charging_nodes = charging_on_nodes + charging_off_nodes
    depot_nodes = [n for n, t in node_types.items() if t == "D"]
    regular_nodes = [n for n in node_types if n not in charging_nodes + depot_nodes]

    # Charging-arc numbering (C+, C-) -> Link ID, read from charging_links.csv
    _, charging_arc_id_map = load_depot_and_charging_arcs(aug_nodes_file, charging_links_file)

    # Depot labels for plot text and legends, e.g. "151+" / "151-"
    depot_start_label = ", ".join(f"{d}+" for d in depot_nodes)
    depot_return_label = ", ".join(f"{d}-" for d in depot_nodes)
    depot_legend_label = f"Depot ({depot_start_label}, {depot_return_label})"

    # =========================
    # CLASSIFY ARCS
    # =========================
    route_edges = set(G_route.edges())

    existing_edges = []
    charging_edges = []
    access_edges = []
    egress_edges = []
    depot_access_edges = []
    depot_egress_edges = []
    depot_egress_distance = {}

    for u, v, d in G_aug.edges(data=True):

        if (u, v) in route_edges:
            existing_edges.append((u, v))

        elif u in charging_on_nodes and v in charging_off_nodes:
            charging_edges.append((u, v))

        elif u in regular_nodes and v in charging_on_nodes:
            access_edges.append((u, v))  # customer (V) to C+

        elif u in depot_nodes and v in charging_on_nodes:
            depot_access_edges.append((u, v))  # start depot (121+) to C+

        elif u in charging_off_nodes and v in regular_nodes:
            egress_edges.append((u, v))  # C- to customer (V)

        elif u in depot_nodes and v in charging_off_nodes:
            depot_egress_edges.append((u, v))  # return depot (121-) to C-
            depot_egress_distance[(u, v)] = d["weight"]

        elif u in charging_off_nodes and v in depot_nodes:
            # If only C- -> depot exists in data, mirror it to plot requested 121- -> C- arc.
            mirrored = (v, u)
            depot_egress_edges.append(mirrored)
            depot_egress_distance[mirrored] = d["weight"]

    depot_access_edges = list(dict.fromkeys(depot_access_edges))
    depot_egress_edges = list(dict.fromkeys(depot_egress_edges))

    # Keep original depot fixed and create return-depot copy with requested offset rule.
    depot_copy_pos = {}
    if depot_nodes:
        x_vals = [p[0] for p in global_pos.values()]
        y_vals = [p[1] for p in global_pos.values()]
        dx = max((max(x_vals) - min(x_vals)) * 0.03, 0.005)
        dy = max((max(y_vals) - min(y_vals)) * 0.03, 0.005)
        for d in depot_nodes:
            x0, y0 = global_pos[d]
            depot_copy_pos[d] = (x0 + dx, y0 + dy)

    # For egress arcs, keep all node positions the same except depot target at return-depot copy.
    aug_pos_return = dict(aug_pos)
    for d in depot_nodes:
        if d in depot_copy_pos:
            aug_pos_return[d] = depot_copy_pos[d]

    route_pos_return = dict(route_pos)
    for d in depot_nodes:
        if d in depot_copy_pos:
            route_pos_return[d] = depot_copy_pos[d]

    # Draw depot-origin egress arcs from the return-depot copy location (121-).
    aug_pos_depot_start = dict(aug_pos)
    for d in depot_nodes:
        if d in depot_copy_pos:
            aug_pos_depot_start[d] = depot_copy_pos[d]

    # =========================
    # PLOTTING (SIDE BY SIDE)
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # =========================
    # 🔷 LEFT: ORIGINAL ROUTE
    # =========================
    ax = axes[0]

    route_regular_nodes = [n for n in G_route.nodes() if n in regular_nodes]
    route_charging_nodes = [n for n in G_route.nodes() if n in charging_nodes]
    route_depot_nodes = [n for n in G_route.nodes() if n in depot_nodes]
    route_return_edges = [(u, v) for (u, v) in G_route.edges() if u in regular_nodes and v in depot_nodes]
    route_non_return_edges = [(u, v) for (u, v) in G_route.edges() if (u, v) not in route_return_edges]

    nx.draw_networkx_nodes(
        G_route, route_pos,
        nodelist=route_regular_nodes,
        node_color='white',
        edgecolors='black',
        node_size=500,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G_route, route_pos,
        nodelist=route_charging_nodes,
        node_color='red',
        edgecolors='black',
        node_size=120,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G_route, route_pos,
        nodelist=route_depot_nodes,
        node_color='orange',
        edgecolors='black',
        node_size=500,
        ax=ax
    )

    if route_depot_nodes:
        nx.draw_networkx_nodes(
            G_route,
            depot_copy_pos,
            nodelist=route_depot_nodes,
            node_color='orange',
            edgecolors='black',
            node_size=500,
            ax=ax
        )

    nx.draw_networkx_edges(
        G_route, route_pos,
        edgelist=route_non_return_edges,
        edge_color='black',
        width=3,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        ax=ax
    )

    nx.draw_networkx_edges(
        G_route, route_pos_return,
        edgelist=route_return_edges,
        edge_color='black',
        width=3,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        ax=ax
    )

    for d in route_depot_nodes:
        x0, y0 = route_pos[d]
        ax.text(x0, y0, f"{d}+", fontsize=11, ha='center', va='center', zorder=6)
        if d in depot_copy_pos:
            x1, y1 = depot_copy_pos[d]
            ax.text(x1, y1, f"{d}-", fontsize=11, ha='center', va='center', zorder=6)

    # Label non-charging nodes (customers) only.
    route_customer_labels = {n: int(n) for n in route_regular_nodes}
    if route_customer_labels:
        nx.draw_networkx_labels(
            G_route, route_pos,
            labels=route_customer_labels,
            font_size=10,
            ax=ax
        )

    route_charging_labels = {
        (u, v): str(charging_arc_id_map[(u, v)])
        for (u, v) in G_route.edges()
        if (u, v) in charging_arc_id_map
    }
    if route_charging_labels:
        nx.draw_networkx_edge_labels(
            G_route, route_pos,
            edge_labels=route_charging_labels,
            font_size=10,
            font_color='black',
            ax=ax
        )

    ax.set_title(f"Route {route_id} - Original Route", fontsize=14)

    legend_elements_left = [
        Line2D([0], [0], color='black', lw=3, label='Route Edge'),
        Line2D([0], [0], marker='o', color='w', label='Customer Node',
               markerfacecolor='white', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Charging Node (C+, C-)',
               markerfacecolor='red', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='w', label=depot_legend_label,
               markerfacecolor='orange', markeredgecolor='black', markersize=10)
    ]

    ax.legend(handles=legend_elements_left, loc='best')

    # =========================
    # 🔷 RIGHT: AUGMENTED NETWORK
    # =========================
    ax = axes[1]

    # Regular nodes
    nx.draw_networkx_nodes(
        G_aug, aug_pos,
        nodelist=regular_nodes,
        node_color='white',
        edgecolors='black',
        node_size=500,
        ax=ax
    )

    # Charging nodes (red)
    nx.draw_networkx_nodes(
        G_aug, aug_pos,
        nodelist=charging_nodes,
        node_color='red',
        edgecolors='black',
        node_size=120,
        ax=ax
    )

    # Start and return depots in orange (same style, different labels)
    nx.draw_networkx_nodes(
        G_aug, aug_pos,
        nodelist=depot_nodes,
        node_color='orange',
        edgecolors='black',
        node_size=500,
        ax=ax
    )

    if depot_nodes:
        nx.draw_networkx_nodes(
            G_aug,
            depot_copy_pos,
            nodelist=depot_nodes,
            node_color='orange',
            edgecolors='black',
            node_size=500,
            ax=ax
        )

    for d in depot_nodes:
        x0, y0 = aug_pos[d]
        ax.text(x0, y0, f"{d}+", fontsize=11, ha='center', va='center', zorder=6)
        if d in depot_copy_pos:
            x1, y1 = depot_copy_pos[d]
            ax.text(x1, y1, f"{d}-", fontsize=11, ha='center', va='center', zorder=6)

    # Label non-charging nodes (customers) only.
    aug_customer_labels = {n: int(n) for n in regular_nodes}
    if aug_customer_labels:
        nx.draw_networkx_labels(
            G_aug, aug_pos,
            labels=aug_customer_labels,
            font_size=10,
            ax=ax
        )

    # Charging arc labels only
    charging_edge_labels = {
        (u, v): str(charging_arc_id_map[(u, v)])
        for (u, v) in charging_edges
        if (u, v) in charging_arc_id_map
    }
    if charging_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos,
            edge_labels=charging_edge_labels,
            font_size=10,
            font_color='black',
            ax=ax
        )

    # No general node labels; only depot and charging-arc labels are shown.
    # Route edges
    existing_return_edges = [(u, v) for (u, v) in existing_edges if u in regular_nodes and v in depot_nodes]
    existing_non_return_edges = [(u, v) for (u, v) in existing_edges if (u, v) not in existing_return_edges]

    nx.draw_networkx_edges(
        G_aug, aug_pos,
        edgelist=existing_non_return_edges,
        edge_color='black',
        width=3,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        ax=ax
    )

    nx.draw_networkx_edges(
        G_aug, aug_pos_return,
        edgelist=existing_return_edges,
        edge_color='black',
        width=3,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        ax=ax
    )

    # 🔵 Access arcs (V to C+)
    nx.draw_networkx_edges(
        G_aug, aug_pos,
        edgelist=access_edges,
        edge_color='blue',
        style='dashed',
        width=1.5,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        ax=ax
    )

    # Additional access arcs from 121+ to C+
    nx.draw_networkx_edges(
        G_aug, aug_pos,
        edgelist=depot_access_edges,
        edge_color='blue',
        style='dashed',
        width=1.5,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        ax=ax
    )

    # 🟢 Egress arcs (C- to V)
    nx.draw_networkx_edges(
        G_aug, aug_pos,
        edgelist=egress_edges,
        edge_color='green',
        style='dashed',
        width=1.5,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        ax=ax
    )

    # Additional egress arcs from 121- to C-
    nx.draw_networkx_edges(
        G_aug, aug_pos_depot_start,
        edgelist=depot_egress_edges,
        edge_color='green',
        style='dashed',
        width=1.5,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        ax=ax
    )

    # Charging arcs (directed C+ -> C-)
    nx.draw_networkx_edges(
        G_aug, aug_pos,
        edgelist=charging_edges,
        edge_color='black',
        width=2,
        arrows=True,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.35',
        arrowsize=12,
        ax=ax
    )

    # Travel-distance labels for route, access, and egress arcs.
    existing_edge_labels = {
        (u, v): f"{G_aug[u][v]['weight']:.1f}"
        for (u, v) in existing_non_return_edges
        if G_aug.has_edge(u, v)
    }
    existing_return_edge_labels = {
        (u, v): f"{G_aug[u][v]['weight']:.1f}"
        for (u, v) in existing_return_edges
        if G_aug.has_edge(u, v)
    }
    access_edge_labels = {
        (u, v): f"{G_aug[u][v]['weight']:.1f}"
        for (u, v) in access_edges
        if G_aug.has_edge(u, v)
    }
    depot_access_edge_labels = {
        (u, v): f"{G_aug[u][v]['weight']:.1f}"
        for (u, v) in depot_access_edges
        if G_aug.has_edge(u, v)
    }
    egress_edge_labels = {
        (u, v): f"{G_aug[u][v]['weight']:.1f}"
        for (u, v) in egress_edges
        if G_aug.has_edge(u, v)
    }
    depot_egress_edge_labels = {
        (u, v): f"{depot_egress_distance[(u, v)]:.1f}"
        for (u, v) in depot_egress_edges
        if (u, v) in depot_egress_distance
    }

    if existing_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos,
            edge_labels=existing_edge_labels,
            font_size=8,
            font_color='black',
            label_pos=0.5,
            ax=ax
        )
    if existing_return_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos_return,
            edge_labels=existing_return_edge_labels,
            font_size=8,
            font_color='black',
            label_pos=0.5,
            ax=ax
        )
    if access_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos,
            edge_labels=access_edge_labels,
            font_size=8,
            font_color='blue',
            label_pos=0.55,
            ax=ax
        )
    if depot_access_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos,
            edge_labels=depot_access_edge_labels,
            font_size=8,
            font_color='blue',
            label_pos=0.6,
            ax=ax
        )
    if egress_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos,
            edge_labels=egress_edge_labels,
            font_size=8,
            font_color='green',
            label_pos=0.45,
            ax=ax
        )
    if depot_egress_edge_labels:
        nx.draw_networkx_edge_labels(
            G_aug, aug_pos_depot_start,
            edge_labels=depot_egress_edge_labels,
            font_size=8,
            font_color='green',
            label_pos=0.4,
            ax=ax
        )

    ax.set_title(f"Route {route_id} - Charging Network", fontsize=14)

    legend_elements_right = [
        Line2D([0], [0], color='black', lw=3, label='Original Route'),
        Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label=f'Access Arc (V/{depot_start_label} → C+)'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label=f'Egress Arc (C- → V/{depot_return_label})'),
        Line2D([0], [0], color='black', lw=2, label='Charging Arc (C+ → C-)'),

        Line2D([0], [0], marker='o', color='w', label='Customer Node',
               markerfacecolor='white', markeredgecolor='black', markersize=10),

        Line2D([0], [0], marker='o', color='w', label='Charging Node (C+, C-)',
               markerfacecolor='red', markeredgecolor='black', markersize=8),

        Line2D([0], [0], marker='o', color='w', label=depot_legend_label,
               markerfacecolor='orange', markeredgecolor='black', markersize=10),
    ]

    ax.legend(handles=legend_elements_right, loc='best')

    # =========================
    # SAVE + SHOW
    # =========================
    plt.tight_layout()

    plt.savefig(
        os.path.join(_figures_folder(folder, id), f"route_{route_id}_augmented_network.svg"),
        format='svg'
    )
    plt.close(fig)







