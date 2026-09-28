import os
import CG_Algorithm as mf
from gams import *
import augument_route_network as augment
import pandas as pd


class FleetSizeInfeasibleError(RuntimeError):
    """Raised when the route-selection MIP is infeasible for a given fleet
    size / period, so the caller can warn and move on to the next fleet
    size instead of the whole run crashing."""

    def __init__(self, fleet_size, folder, original_exc):
        self.fleet_size = fleet_size
        self.folder = folder
        super().__init__(
            f"Fleet size {fleet_size} is infeasible for the route-selection "
            f"MIP in {folder!r} ({original_exc})."
        )


def run_fleet_size(
    all_routes, ws, db, fleet_size,
    folder, node_path, service_path, demand_path, travel_path, links_path,
    charging_links_path, selected_charging_links,
):
    """
    Phase 2 (route selection) + charging-network augmentation for ONE fleet
    size, using the fixed route pool / GAMS db built once in Phase 1.

    num_routes for the augmentation loop is derived from
    len(selected_routes) -- the number of routes actually selected by the
    MIP for this fleet size -- rather than from fleet_size itself, since the
    two are not always equal.
    """
    print("=" * 60)
    print(f"PHASE 2: Selecting optimal routes for fleet size = {fleet_size}...")
    print("=" * 60)

    try:
        selected_routes = mf.select_routes(all_routes, ws, db, fleet_size=fleet_size)
    except ValueError as exc:
        # mf.select_routes raises ValueError('MIP is infeasible') when the
        # route-selection MIP can't be solved for this fleet size -- re-raise
        # as a dedicated, identifiable exception carrying the fleet size and
        # period so the caller can warn and continue instead of crashing.
        raise FleetSizeInfeasibleError(fleet_size, folder, exc) from exc

    num_routes = len(selected_routes)
    print(f"Selected {num_routes} route(s) for fleet size {fleet_size}")
    print(selected_routes)

    export_folder = os.path.join(folder, f"CG_route_exports_{fleet_size}")
    augment.export_route_data(
        selected_routes,
        node_path, service_path, demand_path,
        travel_data_file=travel_path,
        travel_links_file=links_path,
        output_folder=export_folder,
    )

    if num_routes == 0:
        print(f"Fleet size {fleet_size}: no routes selected -- skipping plot and charging augmentation.")
        return selected_routes

    augment.plot_routes(
        evrp_main_data=travel_path,
        nodes_data=node_path,
        routes=selected_routes,
        folder=folder,
        id=fleet_size,
    )

    # ------------------------------------------------------------------ #
    # Charging network augmentation -- one call per SELECTED route, not
    # per fleet_size (num_routes can differ from fleet_size).
    # ------------------------------------------------------------------ #
    aug_output_folder = os.path.join(folder, f"augmented_routes_for_charging_{fleet_size}")
    for k in range(1, num_routes + 1):
        augment.augment_route_with_charging(
            route_nodes_file=os.path.join(export_folder, f"route_{k}_nodes.csv"),
            route_links_file=os.path.join(export_folder, f"route_{k}_travel_links.csv"),
            route_travel_file=os.path.join(export_folder, f"route_{k}_travel_data.csv"),
            route_service_time_file=os.path.join(export_folder, f"route_{k}_service_time.csv"),
            route_waste_demand_file=os.path.join(export_folder, f"route_{k}_waste_demand.csv"),
            charging_links_file=charging_links_path,
            selected_charging_link_ids=selected_charging_links,
            output_folder=aug_output_folder,
            route_id=k,
        )
        augment.visualize_augmented_network(
            route_nodes_file=os.path.join(export_folder, f"route_{k}_nodes.csv"),
            route_travel_file=os.path.join(export_folder, f"route_{k}_travel_data.csv"),
            aug_nodes_file=os.path.join(aug_output_folder, f"route_{k}_aug_nodes.csv"),
            aug_links_file=os.path.join(aug_output_folder, f"route_{k}_aug_links.csv"),
            aug_travel_file=os.path.join(aug_output_folder, f"route_{k}_aug_travel.csv"),
            route_id=k,
            folder=folder,
            id=fleet_size,
            charging_links_file=charging_links_path,
        )

    return selected_routes


def main():
    # ------------------------------------------------------------------ #
    # File paths -- one benchmark planning period per run
    #   <project>/DP_benchmark_data/{subfolder_name}_period/
    # All exports (CG_route_exports_{fleet_size},
    # augmented_routes_for_charging_{fleet_size}) are written inside that
    # period folder, and so are the figures (Figures_{fleet_size}).
    # ------------------------------------------------------------------ #
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "DP_benchmark_data")

    subfolder_name = 'first'   # 'first', 'second', 'third', 'fourth' or 'fifth'
    folder = os.path.join(DATA_ROOT, f"{subfolder_name}_period")

    node_path    = os.path.join(folder, f"{subfolder_name}_period_nodes.csv")
    travel_path  = os.path.join(folder, f"{subfolder_name}_period_travel_data.csv")
    service_path = os.path.join(folder, f"{subfolder_name}_period_service_time.csv")
    demand_path  = os.path.join(folder, f"{subfolder_name}_period_waste_demand.csv")
    links_path   = os.path.join(folder, f"{subfolder_name}_period_links.csv")
    charging_links_path = os.path.join(folder, "charging_links.csv")

    missing = [p for p in (node_path, travel_path, service_path, demand_path,
                           links_path, charging_links_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing input file(s) for the {subfolder_name} period:\n  " + "\n  ".join(missing)
        )

    # Charging links open in this run (Link IDs). Their C+/C- node IDs are
    # read from this period's charging_links.csv.
    selected_charging_links = [1, 2, 3, 4, 5]

    available_links = set(pd.read_csv(charging_links_path)["Link ID"].astype(int))
    unknown_links = sorted(set(selected_charging_links) - available_links)
    if unknown_links:
        raise ValueError(
            f"Charging link(s) {unknown_links} not in {charging_links_path} "
            f"(available: {sorted(available_links)})"
        )

    # Fleet sizes to solve for this period -- edit this list per run.
    fleet_sizes = [5, 6, 7, 8, 9, 10] #, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20


    # ------------------------------------------------------------------ #
    # PHASE 1 — Route Generation (run ONCE per planning period)
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("PHASE 1: Generating route pool Omega...")
    print("=" * 60)

    all_routes, ws, db = mf.generate_routes(
        node_path, travel_path, service_path, demand_path
    )

    # ------------------------------------------------------------------ #
    # PHASE 2 — Route Selection + charging augmentation, per fleet size
    # ------------------------------------------------------------------ #
    infeasible_fleet_sizes = []
    for fleet_size in fleet_sizes:
        try:
            run_fleet_size(
                all_routes, ws, db, fleet_size,
                folder, node_path, service_path, demand_path, travel_path, links_path,
                charging_links_path, selected_charging_links,
            )
        except FleetSizeInfeasibleError as exc:
            print(f"WARNING: {exc}")
            infeasible_fleet_sizes.append(fleet_size)
            continue

    if infeasible_fleet_sizes:
        print("=" * 60)
        print(f"Infeasible fleet size(s) for {folder}: {infeasible_fleet_sizes}")
        print("=" * 60)


if __name__ == "__main__":
    main()



