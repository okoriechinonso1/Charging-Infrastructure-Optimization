###############################################################################
' Exact algorithm Model solved with GUROBI for the EVRP'

###############################################################################
'Import the necessary functions from the main_v3.py script'

from DP_benchmark_scripts import charging_problem
import csv
from DP_benchmark_scripts import load_charging_arc_params
import os
import time


"Charging-second optimization problem"

def main():

    # ================================================================
    # CONFIGURATION — change only these values per run
    # ================================================================

    PERIOD        = 3          # planning period (1 to 5)
    FLEET_SIZE    = 7          # fleet size for this run
    NUM_ROUTES    = None        # None -> number of route_*_aug_travel.csv files in the folder
    DAYS_PER_PERIOD = 1000


    # Physical parameters
    DRIVING_RANGE    = 150
    TOUR_DURATION    = 480
    VEHICLE_CAPACITY = 20000
    DEPOT            = '151'
    LOWEST_RANGE     = 10
    DELTA_MIN        = 10
    DELTA_MAX        = 240
    SoC_REDUCTION_RATE = 1.0
    BIG_M            = 1000
    COST_PER_MILE    = 6.50

    # File paths -- everything lives under <project root>/DP_benchmark_data/,
    # whatever the working directory the IDE / terminal launches from:
    #   {period_name}/augmented_routes_for_charging_{fleet}/  route_k_aug_*.csv (from CG_Algorithm_Main)
    #   {period_name}/charging_links.csv                      charging links of the period
    #   {period_name}/charging_arcs_params.csv                beta1 / alpha per charging link
    #   brute_force_solution/                                 Period_{P}_FleetSize_{F}.csv results
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT    = os.path.join(PROJECT_ROOT, "DP_benchmark_data")

    PERIOD_FOLDER_TEMPLATE = os.path.join(DATA_ROOT, "{period_name}", "augmented_routes_for_charging_{fleet}")
    CHARGING_LINKS_TEMPLATE = os.path.join(DATA_ROOT, "{period_name}", "charging_links.csv")
    CHARGING_ARC_PARAMS_TEMPLATE = os.path.join(DATA_ROOT, "{period_name}", "charging_arcs_params.csv")

    # Where the final summary table for this (PERIOD, FLEET_SIZE) run is saved,
    # as Period_{PERIOD}_FleetSize_{FLEET_SIZE}.csv. DP_Algorithm_Benchmark.py
    # reads these files from this folder.
    RESULTS_DIR = os.path.join(DATA_ROOT, "brute_force_solution")
    # False -> stop before solving if the results CSV for this run already exists
    OVERWRITE_RESULTS = False

    # Period name mapping
    PERIOD_NAMES = {
        1: "first_period",
        2: "second_period",
        3: "third_period",
        4: "fourth_period",
        5: "fifth_period",
    }

    # ================================================================
    # ALL DP STATES for the given fleet size
    # These are all station configurations (subsets of E containing
    # E_bar={1}) paired with the given fleet size.
    # Edit this list to match your exact DP states for E={1,2,3,4,5}
    # ================================================================

    ALL_STATION_CONFIGS = [
        [1, 2, 3, 4, 5],   # ({1,2,3,4,5}, eta)
        [1, 3, 4, 5],       # ({1,3,4,5}, eta)
        [1, 2, 4, 5],       # ({1,2,4,5}, eta)
        [1, 2, 3, 5],       # ({1,2,3,5}, eta)
        [1, 2, 3, 4],       # ({1,2,3,4}, eta)
        [1, 4, 5],          # ({1,4,5}, eta)
        [1, 3, 5],          # ({1,3,5}, eta)
        [1, 3, 4],          # ({1,3,4}, eta)
        [1, 2, 5],          # ({1,2,5}, eta)
        [1, 2, 4],          # ({1,2,4}, eta)
        [1, 2, 3],          # ({1,2,3}, eta)
        [1, 5],             # ({1,5}, eta)
        [1, 4],             # ({1,4}, eta)
        [1, 3],             # ({1,3}, eta)
        [1, 2],             # ({1,2}, eta)
        [1],                # ({1}, eta)   ← base config E_bar
    ]

    # ================================================================
    # HELPER: format state label for reporting
    # ================================================================
    def state_label(config, fleet):
        stations = "{" + ", ".join(str(c) for c in sorted(config)) + "}"
        return f"({stations}, {fleet})"


    # ================================================================
    # CORE FUNCTION: solve all routes for one DP state
    # ================================================================
    def solve_state(selected_charging_links, num_routes, period_folder,
                    beta1_dict, alpha_dict):
        """
        Runs the charging-second optimization for all routes of one
        DP state. Returns a results dictionary.
        """
        total_travel_distance      = 0.0
        total_travel_distance_cost = 0.0
        total_charging_cost        = 0.0
        total_charging_time        = 0.0
        total_waste                = 0.0
        failed_routes              = []

        for route_num in range(1, num_routes + 1):

            evrp_main_data    = os.path.join(period_folder, f"route_{route_num}_aug_travel.csv")
            nodes_data_file   = os.path.join(period_folder, f"route_{route_num}_aug_nodes.csv")
            service_time_file = os.path.join(period_folder, f"route_{route_num}_aug_service_time.csv")
            waste_demand_file = os.path.join(period_folder, f"route_{route_num}_aug_waste_demand.csv")

            try:
                (job, db, ws, feasible,
                 arc_travel, arc_cost_dict,
                 time_level, SoC_level, waste_level,
                 visited_customer_nodes,
                 SolutionTime, objective_function_value,
                 Total_travel_distance, Total_charging_time,
                 Total_waste_amount, dict_computation_time) = charging_problem(
                    evrp_main_data,
                    service_time_file,
                    waste_demand_file,
                    nodes_data_file,
                    DRIVING_RANGE, TOUR_DURATION,
                    VEHICLE_CAPACITY, SoC_REDUCTION_RATE, DEPOT,
                    LOWEST_RANGE, DELTA_MIN,
                    DELTA_MAX, COST_PER_MILE,
                    beta1_dict, alpha_dict, BIG_M
                )

                if not feasible:
                    print(f"    WARNING: Route {route_num} infeasible.")
                    failed_routes.append(route_num)
                    continue

                route_travel_cost   = Total_travel_distance * COST_PER_MILE
                route_charging_cost = objective_function_value

                total_travel_distance      += Total_travel_distance
                total_travel_distance_cost += route_travel_cost
                total_charging_cost        += route_charging_cost
                total_charging_time        += Total_charging_time
                total_waste                += Total_waste_amount

                print(f"    Route {route_num}: "
                      f"travel=${route_travel_cost:.2f}, "
                      f"charging=${route_charging_cost:.2f}, "
                      f"time={SolutionTime}s")

            except Exception as e:
                print(f"    ERROR on route {route_num}: {e}")
                failed_routes.append(route_num)
                continue

        total_daily_cost  = total_travel_distance_cost + total_charging_cost
        total_period_cost = total_daily_cost * DAYS_PER_PERIOD

        return {
            "total_travel_distance"     : total_travel_distance,
            "total_travel_distance_cost": total_travel_distance_cost,
            "total_charging_cost"       : total_charging_cost,
            "total_charging_time"       : total_charging_time,
            "total_waste"               : total_waste,
            "total_daily_cost"          : total_daily_cost,
            "total_period_cost"         : total_period_cost,
            "solved_routes"             : num_routes - len(failed_routes),
            "failed_routes"             : failed_routes,
        }

    # ================================================================
    # MAIN: iterate over all DP states for the given period/fleet
    # ================================================================
    def main():

        period_name   = PERIOD_NAMES[PERIOD]
        period_folder = PERIOD_FOLDER_TEMPLATE.format(
            period_name=period_name,
            fleet=FLEET_SIZE
        )
        charging_links_file = CHARGING_LINKS_TEMPLATE.format(period_name=period_name)
        charging_arc_params = CHARGING_ARC_PARAMS_TEMPLATE.format(period_name=period_name)
        csv_path = os.path.join(RESULTS_DIR, f"Period_{PERIOD}_FleetSize_{FLEET_SIZE}.csv")

        # ---- Check all inputs before any (long) solve starts ----
        if not os.path.isdir(period_folder):
            raise FileNotFoundError(
                f"Augmented routes folder not found: {period_folder}\n"
                f"Run CG_Algorithm_Main.py for {period_name}, fleet size {FLEET_SIZE} first."
            )
        n_route_files = len([f for f in os.listdir(period_folder)
                             if f.startswith("route_") and f.endswith("_aug_travel.csv")])
        num_routes = n_route_files if NUM_ROUTES is None else NUM_ROUTES
        if num_routes != n_route_files:
            print(f"WARNING: NUM_ROUTES = {NUM_ROUTES} but {period_folder} "
                  f"has {n_route_files} routes.")

        for path in (charging_links_file, charging_arc_params):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing input file: {path}")

        available_links = set()
        with open(charging_links_file, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                available_links.add(int(row["Link ID"]))
        unknown = sorted({c for config in ALL_STATION_CONFIGS for c in config} - available_links)
        if unknown:
            raise ValueError(f"Station configs use link ID(s) {unknown} that are not in "
                             f"{charging_links_file} (available: {sorted(available_links)})")

        if os.path.exists(csv_path) and not OVERWRITE_RESULTS:
            raise FileExistsError(
                f"{csv_path} already exists. Set OVERWRITE_RESULTS = True to replace it."
            )

        print("=" * 70)
        print(f"PERIOD: {PERIOD} ({period_name})  |  "
              f"FLEET SIZE: {FLEET_SIZE}  |  "
              f"ROUTES PER STATE: {num_routes}")
        print(f"DATA FOLDER: {period_folder}")
        print(f"ARC PARAMS : {charging_arc_params}")
        print("=" * 70)

        # Collect results for all states
        all_results     = {}
        infeasible_log  = {}   # state_label → list of failed route numbers

        for config in ALL_STATION_CONFIGS:

            label = state_label(config, FLEET_SIZE)
            print(f"\n--- Solving state {label} ---")

            # Load arc parameters for this station configuration
            try:
                beta1_dict, alpha_dict = load_charging_arc_params(
                    charging_arc_params,
                    config
                )
            except Exception as e:
                print(f"  ERROR loading arc params for {label}: {e}")
                infeasible_log[label] = ["param_load_error"]
                continue

            # Solve all routes for this state
            state_start_time = time.time()
            # Solve all routes for this state
            results = solve_state(
                config, num_routes, period_folder,
                beta1_dict, alpha_dict
            )

            state_elapsed_time = time.time() - state_start_time

            all_results[label] = results
            all_results[label]["comp_time"] = state_elapsed_time

            # Log infeasible routes if any
            if results["failed_routes"]:
                infeasible_log[label] = results["failed_routes"]

            # Per-state summary
            print(f"  Solved routes    : "
                  f"{results['solved_routes']} / {num_routes}")
            print(f"  Travel cost      : "
                  f"${results['total_travel_distance_cost']:.2f}")
            print(f"  Charging cost    : "
                  f"${results['total_charging_cost']:.2f}")
            print(f"  Daily op. cost   : "
                  f"${results['total_daily_cost']:.2f}")
            print(f"  Computation time : {state_elapsed_time:.2f} seconds")
            print(f"  Period cost(×{DAYS_PER_PERIOD}): "
                  f"${results['total_period_cost']:.2f}")
            if results["failed_routes"]:
                print(f"  INFEASIBLE ROUTES: {results['failed_routes']}")

        # ================================================================
        # FINAL SUMMARY TABLE
        # ================================================================
        print("\n" + "=" * 70)
        print(f"FULL SUMMARY — Period {PERIOD}, Fleet {FLEET_SIZE}")
        print("=" * 70)
        print(f"{'DP State':<30} {'Daily Cost':>14} "
              f"{'Period Cost':>14} {'Travel Dist (mi)':>18} "
              f"{'Charging Time (min)':>21} {'Comp Time (s)':>15} "
              f"{'Solved':>8} {'Failed Routes'}")
        print("-" * 125)

        for label, res in all_results.items():
            failed_str = (str(res["failed_routes"])
                          if res["failed_routes"] else "None")

            print(f"  {label:<28} "
                  f"${res['total_daily_cost']:>12.2f}  "
                  f"${res['total_period_cost']:>12.2f}  "
                  f"{res['total_travel_distance']:>16.2f}  "
                  f"{res['total_charging_time']:>19.2f}  "
                  f"{res['comp_time']:>13.2f}s  "
                  f"{res['solved_routes']:>5}/{num_routes}  "
                  f"{failed_str}")

        # Infeasibility report
        if infeasible_log:
            print("\n--- INFEASIBILITY REPORT ---")
            for label, failed in infeasible_log.items():
                print(f"  State {label}: infeasible/error on routes {failed}")
        else:
            print("\nAll routes feasible across all states.")

        # ================================================================
        # SAVE FINAL SUMMARY TABLE TO CSV
        # Written to DP_benchmark_data/brute_force_solution/
        # Period_{PERIOD}_FleetSize_{FLEET_SIZE}.csv -- one row per station
        # configuration solved this run, same columns as the printed table.
        # ================================================================
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "DP State", "Daily Cost", "Period Cost",
                "Travel Dist (mi)", "Charging Time (min)",
                "Comp Time (s)", "Solved Routes", "Total Routes",
                "Failed Routes",
            ])
            for label, res in all_results.items():
                failed_str = (str(res["failed_routes"])
                              if res["failed_routes"] else "None")
                writer.writerow([
                    label,
                    f"{res['total_daily_cost']:.2f}",
                    f"{res['total_period_cost']:.2f}",
                    f"{res['total_travel_distance']:.2f}",
                    f"{res['total_charging_time']:.2f}",
                    f"{res['comp_time']:.2f}",
                    res["solved_routes"],
                    num_routes,
                    failed_str,
                ])

        print(f"\nResults table saved to: {csv_path}")

        print("=" * 70)
        print("Instance run of the DP algorithm has been completed.")

        return all_results

    # ================================================================
    if __name__ == "__main__":
        main()


if __name__ == "__main__":
    main()


