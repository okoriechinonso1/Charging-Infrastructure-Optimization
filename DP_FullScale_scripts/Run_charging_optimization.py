from charging_optimization_model import charging_problem
import csv
from model_parameter_analysis import load_charging_arc_params
import os
import re
import time
from typing import Dict, List, Optional, Tuple


"Charging-second optimization problem"

# ================================================================
# Reusable pieces (module level, importable) -- these used to live only as
# closures inside main(), which made this script un-callable from anywhere
# else (e.g. the labeling DP driver's automated mu-solving step).
# ================================================================

STATE_LABEL_RE = re.compile(r"\(\{([^}]*)\},\s*([+-]?\d+)\)")

PERIOD_NAMES = {
    1: "first_period",
    2: "second_period",
    3: "third_period",
    4: "fourth_period",
    5: "fifth_period",
    6: "sixth_period",
    7: "seventh_period",
    8: "eighth_period",
    9: "ninth_period",
    10: "tenth_period",
}


def parse_state_label(label: str) -> Optional[Tuple[List[int], int]]:
    """Parse a state label like '({1, 2, 3, 4, 5, 6, 7}, 25)' into
    (config_list, fleet_size), e.g. ([1, 2, 3, 4, 5, 6, 7], 25). Station IDs
    come back as ints, matching the format the DP script's state_to_key()
    prints. Returns None for blank/unmatched lines (skipped silently, not
    treated as errors)."""
    label = label.strip()
    if not label:
        return None
    m = STATE_LABEL_RE.search(label)
    if not m:
        return None
    stations_part, fleet_part = m.groups()
    config = [int(s.strip()) for s in stations_part.split(",") if s.strip()]
    fleet = int(fleet_part.strip())
    return config, fleet


def load_dp_states(path: str) -> List[Tuple[List[int], int]]:
    """Read a plain text file with ONE state label per line and return a
    list of (config_list, fleet_size) tuples. Lines that don't match the
    expected '({...}, fleet)' pattern (blank lines, stray whitespace, etc.)
    are skipped rather than raising."""
    states = []
    with open(path, "r") as f:
        for line in f:
            parsed = parse_state_label(line)
            if parsed is not None:
                states.append(parsed)
    if not states:
        raise ValueError(f"No DP states parsed from {path}. Check the file "
                          f"contains lines like '({{1, 2, 3}}, 10)'.")
    return states


def state_label(config: List[int], fleet: int) -> str:
    stations = "{" + ", ".join(str(c) for c in sorted(config)) + "}"
    return f"({stations}, {fleet})"


_ROUTE_TRAVEL_FILE_RE = re.compile(r"^route_(\d+)_aug_travel\.csv$")


def discover_route_numbers(period_folder: str) -> List[int]:
    """
    Scan period_folder for 'route_<N>_aug_travel.csv' files and return the
    sorted list of route numbers N actually present there -- used to
    determine how many routes to solve instead of trusting a hardcoded
    NUM_ROUTES that may not match what's really on disk for this period.
    """
    if not os.path.isdir(period_folder):
        raise FileNotFoundError(f"Route data folder not found: {period_folder}")
    route_numbers = sorted(
        int(m.group(1))
        for fname in os.listdir(period_folder)
        if (m := _ROUTE_TRAVEL_FILE_RE.match(fname))
    )
    if not route_numbers:
        raise FileNotFoundError(
            f"No 'route_<N>_aug_travel.csv' files found in {period_folder}."
        )
    return route_numbers


def solve_state(
        selected_charging_links: List[int],
        route_numbers: List[int],
        period_folder: str,
        beta1_dict, alpha_dict,
        driving_range: float, tour_duration: float, vehicle_capacity: float,
        depot: str, lowest_range: float, delta_min: float, delta_max: float,
        soc_reduction_rate: float, big_m: float, cost_per_mile: float,
        days_per_period: float,
) -> Dict:
    """
    Runs the charging-second optimization for every route number in
    route_numbers, for one DP state. Returns a results dictionary.
    """
    total_travel_distance      = 0.0
    total_travel_distance_cost = 0.0
    total_charging_cost        = 0.0
    total_charging_time        = 0.0
    total_waste                = 0.0
    failed_routes: List[int]   = []
    num_routes = len(route_numbers)

    for route_num in route_numbers:

        evrp_main_data    = f"{period_folder}/route_{route_num}_aug_travel.csv"
        nodes_data_file   = f"{period_folder}/route_{route_num}_aug_nodes.csv"
        service_time_file = f"{period_folder}/route_{route_num}_aug_service_time.csv"
        waste_demand_file = f"{period_folder}/route_{route_num}_aug_waste_demand.csv"

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
                driving_range, tour_duration,
                vehicle_capacity, soc_reduction_rate, depot,
                lowest_range, delta_min,
                delta_max, cost_per_mile,
                beta1_dict, alpha_dict, big_m
            )

            if not feasible:
                print(f"    WARNING: Route {route_num} infeasible.")
                failed_routes.append(route_num)
                continue

            route_travel_cost   = Total_travel_distance * cost_per_mile
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
    total_period_cost = total_daily_cost * days_per_period

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

def solve_dp_states(
        period: int,
        dp_states: List[Tuple[List[int], int]],
        period_folder: str,
        charging_arc_params_path: str,
        days_per_period: float,
        num_routes: Optional[int] = None,
        driving_range: float = 150,
        tour_duration: float = 480,
        vehicle_capacity: float = 20000,
        depot: str = '151',
        lowest_range: float = 10,
        delta_min: float = 10,
        delta_max: float = 240,
        soc_reduction_rate: float = 1.0,
        big_m: float = 1000,
        cost_per_mile: float = 6.50,
        strict: bool = True,
) -> Dict[str, Dict]:
    """
    Solve the lower-level charging-second optimization for every
    (station_config, fleet_size) DP state in dp_states. Returns
    {state_label: results_dict} where results_dict is solve_state()'s
    return value plus 'fleet_size' and 'num_routes'. results_dict[
    'total_period_cost'] is D^p * mu^p -- already scaled by
    days_per_period, i.e. exactly the value the DP script's
    'mu_Dp_scaled' column expects.

    num_routes: leave as None (the default) to auto-detect how many routes
    to solve by scanning period_folder for 'route_<N>_aug_travel.csv' files
    (via discover_route_numbers) -- this is preferred, since it can never
    drift out of sync with what's actually on disk for this period. Pass an
    explicit int only to deliberately override/cap it (e.g. for a quick
    partial test run).

    strict=True (the default, used by the DP labeling automation): a
    charging-arc-param load error, or a state that solves ZERO routes,
    raises immediately instead of silently producing a 0-cost mu -- 0 is
    reserved for the DP script's pre-filled fleet-size-infeasibility
    convention, so a genuinely-attempted state must never collide with it.

    strict=False (used by the standalone CLI run below, to preserve this
    script's original resilient behavior): those two cases are skipped
    and logged into the returned dict's "_errors" key
    ({state_label: error message}) instead of raising.
    """
    if num_routes is None:
        route_numbers = discover_route_numbers(period_folder)
        print(f"Auto-detected {len(route_numbers)} route(s) in {period_folder}: {route_numbers}")
    else:
        route_numbers = list(range(1, num_routes + 1))

    all_results: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}

    for config, fleet_size in dp_states:
        label = state_label(config, fleet_size)
        print(f"\n--- Solving state {label} ---")

        try:
            beta1_dict, alpha_dict = load_charging_arc_params(
                charging_arc_params_path, config
            )
        except Exception as e:
            msg = f"param_load_error: {e}"
            print(f"  ERROR loading arc params for {label}: {e}")
            if strict:
                raise RuntimeError(f"State {label}: {msg}") from e
            errors[label] = msg
            continue

        state_start_time = time.time()
        results = solve_state(
            config, route_numbers, period_folder, beta1_dict, alpha_dict,
            driving_range, tour_duration, vehicle_capacity, depot,
            lowest_range, delta_min, delta_max, soc_reduction_rate, big_m,
            cost_per_mile, days_per_period,
        )
        state_elapsed_time = time.time() - state_start_time

        if results["solved_routes"] == 0:
            msg = (f"all {len(route_numbers)} route(s) failed/infeasible -- "
                   f"refusing to return a 0-cost mu for a state that was "
                   f"actually attempted (0 is reserved for the DP script's "
                   f"fleet-size-infeasibility convention)")
            print(f"  ERROR: {msg}")
            if strict:
                raise RuntimeError(f"State {label}: {msg}")
            errors[label] = msg
            continue

        results["comp_time"]  = state_elapsed_time
        results["fleet_size"] = fleet_size
        results["num_routes"] = len(route_numbers)
        all_results[label] = results

        print(f"  Fleet size       : {fleet_size}")
        print(f"  Solved routes    : {results['solved_routes']} / {len(route_numbers)}")
        print(f"  Travel cost      : ${results['total_travel_distance_cost']:.2f}")
        print(f"  Charging cost    : ${results['total_charging_cost']:.2f}")
        print(f"  Daily op. cost   : ${results['total_daily_cost']:.2f}")
        print(f"  Computation time : {state_elapsed_time:.2f} seconds")
        print(f"  Period cost(x{days_per_period}): ${results['total_period_cost']:.2f}")
        if results["failed_routes"]:
            print(f"  INFEASIBLE ROUTES: {results['failed_routes']}")

    if errors:
        all_results["_errors"] = errors
    return all_results

