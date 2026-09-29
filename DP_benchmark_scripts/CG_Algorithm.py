import datetime
import math
from itertools import combinations
from sklearn.cluster import KMeans
import itertools
import pickle
import networkx as nx
import numpy as np
from gams import *
import pandas as pd
import CG_Algorithm as mf
import mip_solve as exact
import rmip_solve as rmip
import operator
import os
import re
import logging
import sys
import datetime
import itertools
from gams import GamsWorkspace, GamsExceptionExecution
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# SHARED HELPER FUNCTIONS
# (Used by both Phase 1 and Phase 2)
# =============================================================================

def compute_shortest_path(df):
    """
    Build all-pairs shortest path dictionaries for distance and time
    from the travel data CSV using Dijkstra's algorithm.
    Returns travel_dist_dict and travel_time_dict.
    """
    G_distance = nx.DiGraph()
    G_time = nx.DiGraph()

    for _, row in df.iterrows():
        from_node = 'i' + str(int(row['from_node ID']))
        to_node   = 'i' + str(int(row['to_node ID']))
        distance  = float(row['Distance (miles)'])
        time      = float(row['Time (minutes)'])
        G_distance.add_edge(from_node, to_node, weight=distance)
        G_time.add_edge(from_node, to_node, weight=time)

    shortest_paths_distance = dict(nx.all_pairs_dijkstra_path_length(G_distance, weight='weight'))
    shortest_paths_time     = dict(nx.all_pairs_dijkstra_path_length(G_time,     weight='weight'))

    travel_dist_dict = {source: targets for source, targets in shortest_paths_distance.items()}
    travel_time_dict = {source: targets for source, targets in shortest_paths_time.items()}

    return travel_dist_dict, travel_time_dict


def determine_charge_time(route):
    """
    Given a route that already contains a charging station node,
    compute how long the truck needs to charge to complete the route
    while respecting min_driving_range safety buffer.
    Returns (True, charge_time) if charging is needed, else (False, 0).
    """
    remaining_range = max_driving_range

    for idx in range(len(route) - 1):

        cur_id = route[idx][0]
        if cur_id in truck_info:
            cur_node = truck_info[cur_id]['loc']
        elif cur_id in community_info:
            cur_node = community_info[cur_id]['loc']
        else:
            cur_node = charge_station_info[cur_id]["off_charge_node"]

        next_id = route[idx + 1][0]

        # ---------------- CHARGING STATION ----------------
        if next_id in charge_station_info:

            on_node  = charge_station_info[next_id]["on_charge_node"]
            off_node = charge_station_info[next_id]["off_charge_node"]

            dist_to_cs = (
                travel_dist_dict[cur_node][on_node] +
                travel_dist_dict[on_node][off_node]
            )
            remaining_range -= dist_to_cs

            # compute required future distance from charging station onward
            future_dist = 0
            temp_node   = off_node
            for j in range(idx + 1, len(route) - 1):
                next_j_id = route[j + 1][0]
                if next_j_id in truck_info:
                    next_j_node = truck_info[next_j_id]['loc']
                elif next_j_id in community_info:
                    next_j_node = community_info[next_j_id]['loc']
                else:
                    next_j_node = charge_station_info[next_j_id]["off_charge_node"]
                future_dist += travel_dist_dict[temp_node][next_j_node]
                temp_node = next_j_node

            # include safety buffer and compute charge needed
            required_energy = future_dist + min_driving_range
            needed_charge   = max(0, required_energy - remaining_range)
            charge_time     = needed_charge / charge_rate

            if charge_time > 0:
                charge_time = max(min_charge_time, charge_time)
                return True, charge_time

        # ---------------- NORMAL ARC ----------------
        else:
            if next_id in truck_info:
                next_node = truck_info[next_id]['loc']
            elif next_id in community_info:
                next_node = community_info[next_id]['loc']
            else:
                next_node = charge_station_info[next_id]["off_charge_node"]

            remaining_range -= travel_dist_dict[cur_node][next_node]

    return False, 0


def insert_charge_station(route):
    """
    Given a route with a range violation, try inserting every available
    charging station at every possible position. Returns a list of all
    routes that become feasible after the insertion.
    """
    feasible_inserts = []

    for cs_id in charge_station_info.keys():
        for i in range(1, len(route)):
            cur_insert = route[:i] + [(cs_id, 0)] + route[i:]

            charge_feasible, charge_time = mf.determine_charge_time(cur_insert)

            if charge_feasible:
                cur_insert  = route[:i] + [(cs_id, charge_time)] + route[i:]
                is_feasible = mf.check_route_feasibility(cur_insert)
                if is_feasible == True:
                    route_pool.append(cur_insert)
                    feasible_inserts.append(cur_insert)

    return feasible_inserts


def check_route_feasibility(route):
    """
    Walk through a route step by step, tracking cumulative waste load
    and remaining battery range.
    Returns:
        True                 — route is fully feasible
        'capacity_violation' — total waste exceeds truck_cap
        'range_violation'    — remaining range drops below min_driving_range
    """
    load            = 0
    remaining_range = max_driving_range

    for idx in range(len(route) - 1):
        cur_id = route[idx][0]

        if cur_id in truck_info.keys():
            cur_node = truck_info[cur_id]['loc']
        elif cur_id in community_info.keys():
            cur_node = community_info[cur_id]['loc']
        else:
            cur_node = charge_station_info[cur_id]["off_charge_node"]

        next_id = route[idx + 1][0]

        if next_id in truck_info.keys():
            next_node = truck_info[next_id]['loc']
        elif next_id in community_info.keys():
            next_node = community_info[next_id]['loc']
        else:
            next_node = charge_station_info[next_id]["off_charge_node"]

        if next_id in community_info.keys():
            load += community_info[next_id]['waste_amount']
            if load > truck_cap:
                return 'capacity_violation'

        if next_id in charge_station_info.keys():
            charge_time      = route[idx + 1][1]
            on_charge_node   = charge_station_info[next_id]["on_charge_node"]
            off_charge_node  = charge_station_info[next_id]["off_charge_node"]
            distance         = (travel_dist_dict[cur_node][on_charge_node]
                                + travel_dist_dict[on_charge_node][off_charge_node])
            remaining_range  = remaining_range - distance
            range_increment  = charge_time * charge_rate
            remaining_range  = min(max_driving_range, remaining_range + range_increment)
        else:
            distance        = travel_dist_dict[cur_node][next_node]
            remaining_range = remaining_range - distance
            if remaining_range < min_driving_range - 0.001:
                return 'range_violation'

    return True


def generate_1v1(truck_id, cm_id):
    """Generate the simplest route: one truck visiting one community and returning."""
    return [(truck_id, 0), (cm_id, community_info[cm_id]["service_time"]), (truck_id, 0)]


def get_route_loc(route):
    """Convert a route from ID-based representation to actual network node location strings."""
    route_loc = []
    for idx in route:
        idx = idx[0]
        if idx in truck_info.keys():
            route_loc.append(truck_info[idx]['loc'])
        elif idx in community_info.keys():
            route_loc.append(community_info[idx]['loc'])
        else:
            on_charge_node  = charge_station_info[idx]["on_charge_node"]
            off_charge_node = charge_station_info[idx]["off_charge_node"]
            route_loc.extend([on_charge_node, off_charge_node])
    return route_loc


def find_route_dist_time(route):
    """
    Compute total travel distance, total travel time, and total charge time
    for a given route. Charging arcs are tracked separately from driving arcs.
    Returns (total_travel_dist, total_travel_time, total_charge_time).
    """
    total_travel_dist = 0
    total_travel_time = 0
    total_charge_time = 0

    for idx in range(len(route) - 1):
        cur_id = route[idx][0]
        nxt_id = route[idx + 1][0]
        t      = route[idx + 1][1]   # service time or charging time at that stop

        if cur_id in truck_info:
            cur_node = truck_info[cur_id]['loc']
        elif cur_id in community_info:
            cur_node = community_info[cur_id]['loc']
        else:
            cur_node = charge_station_info[cur_id]['off_charge_node']

        if nxt_id in truck_info:
            nxt_node = truck_info[nxt_id]['loc']
        elif nxt_id in community_info:
            nxt_node = community_info[nxt_id]['loc']
        else:
            nxt_node = charge_station_info[nxt_id]['off_charge_node']

        if nxt_id in charge_station_info:
            on  = charge_station_info[nxt_id]['on_charge_node']
            off = charge_station_info[nxt_id]['off_charge_node']
            total_travel_dist += (travel_dist_dict[cur_node][on]
                                  + travel_dist_dict[on][off])
            total_travel_time += (travel_time_dict[cur_node][on]
                                  + travel_time_dict[on][off])
            total_charge_time += t
        else:
            total_travel_dist += travel_dist_dict[cur_node][nxt_node]
            total_travel_time += travel_time_dict[cur_node][nxt_node]

    return total_travel_dist, total_travel_time, total_charge_time


def add_to_all_route(route, if_route_feasible, truck_id):
    """
    Assign a new route ID, compute all route metrics, and store everything
    in the global all_routes dictionary.
    Route cost = unit_dist_cost * total_travel_dist  (distance only, no charging cost).
    Returns the new route_id.
    """
    route_id = f"r{len(all_routes) + 1}"

    all_routes[route_id] = {}
    all_routes[route_id]["route_id"]   = route_id
    all_routes[route_id]["route"]      = route
    all_routes[route_id]["route_loc"]  = mf.get_route_loc(route)
    all_routes[route_id]["isFeasible"] = if_route_feasible

    total_travel_dist, total_travel_time, total_charging_time = find_route_dist_time(route)

    all_routes[route_id]["trip_milage"]  = total_travel_dist
    all_routes[route_id]["trip_duration"]= total_travel_time
    all_routes[route_id]["truck_id"]     = truck_id
    all_routes[route_id]["cs_ids"]       = [nid for nid, _ in route if nid in charge_station_info]

    cm_covered = [nid for nid, _ in route if nid in community_info]
    all_routes[route_id]["cm_ids"]    = cm_covered
    all_routes[route_id]["count_cm"]  = len(cm_covered)

    total_load         = sum(community_info[cm]['waste_amount']  for cm in cm_covered)
    total_service_time = sum(community_info[cm]['service_time']  for cm in cm_covered)

    all_routes[route_id]["waste_collection"] = total_load
    all_routes[route_id]["service_time"]     = total_service_time
    all_routes[route_id]["charge_time"]      = total_charging_time

    # Route cost: distance only — charging cost muted
    all_routes[route_id]["route_cost"] = unit_dist_cost * total_travel_dist

    return route_id


def get_basic_solution(gams_job):
    """Extract route IDs with x(r) > 0 from the GAMS RMP output — the current basis."""
    sol = []
    for rec in gams_job.out_db["x"]:
        if rec.level > 0:
            sol.append(rec.keys[0])
    return sol


def generate_nvm_routes(m):
    """
    Generate m-community routes by inserting one new community into every
    (m-1)-community route from the previous level. Handles range violations
    by attempting charging station insertion.
    """
    nvm_route_ids  = {}
    prem_route_ids = all_route_ids[m - 1]

    for truck_id in truck_info.keys():
        cur_truck_prem_route_ids = prem_route_ids[truck_id]
        cur_truck_vm_route_ids   = []

        for cur_truck_rtd in cur_truck_prem_route_ids:
            route      = all_routes[cur_truck_rtd]['route']
            cm_covered = all_routes[cur_truck_rtd]['cm_ids']

            for cm_id in community_info.keys():
                if cm_id not in cm_covered:
                    for i in range(1, len(route)):
                        cur_insert = route[:i] + [(cm_id, community_info[cm_id]['service_time'])] + route[i:]
                        if cur_insert not in route_pool:
                            if_feasible = mf.check_route_feasibility(cur_insert)
                            if if_feasible == True:
                                route_pool.append(cur_insert)
                                route_id = mf.add_to_all_route(cur_insert, True, truck_id)
                                cur_truck_vm_route_ids.append(route_id)
                            else:
                                if if_feasible == 'range_violation':
                                    fesible_routes_with_charge = mf.insert_charge_station(cur_insert)
                                    if len(fesible_routes_with_charge) > 0:
                                        for fesible_route in fesible_routes_with_charge:
                                            route_pool.append(fesible_route)
                                            route_id = mf.add_to_all_route(fesible_route, True, truck_id)
                                            cur_truck_vm_route_ids.append(route_id)

        nvm_route_ids[truck_id] = cur_truck_vm_route_ids

    all_route_ids[m] = nvm_route_ids
    print(f"m: {m}; # number of routes: {len(all_routes)}")


def generate_best_route_by_one_insertion(route, am_id_insert, community_dual):
    """
    PSP: Try inserting community am_id_insert at every position in the given route.
    For each feasible insertion, compute reduced cost:
        phi_w = g_w - sum(tau_v for v in route communities)
    If phi_w < -0.00001, the route is improving and added to list_good_routes.
    Range violations are handled by attempting charging station insertion.
    Returns (True, list_good_routes) if any improving routes found, else (False, None).
    """
    list_good_routes = []
    #dict_reduced_cost = {}

    for i in range(1, len(route)):
        cur_insert = route[:i] + [(am_id_insert, community_info[am_id_insert]['service_time'])] + route[i:]

        if cur_insert not in route_pool:
            if_feasible = mf.check_route_feasibility(cur_insert)

            # ---- FEASIBLE INSERTION ----
            if if_feasible == True:
                cm_ids = [id[0] for id in cur_insert if id[0] in community_info.keys()]
                total_travel_dist, total_travel_time, charging_time = mf.find_route_dist_time(cur_insert)
                total_route_cost = total_travel_dist * unit_dist_cost   # distance cost only
                reduced_cost     = total_route_cost - sum(community_dual[cm] for cm in cm_ids)

                if reduced_cost < -0.00001:
                    list_good_routes.append(cur_insert)
                    #dict_reduced_cost[tuple(cur_insert)] = reduced_cost
                    route_pool.append(cur_insert)

            # ---- RANGE VIOLATION — try inserting charging station ----
            else:
                if if_feasible == 'range_violation':
                    fesible_routes_with_charge = mf.insert_charge_station(cur_insert)
                    if len(fesible_routes_with_charge) > 0:
                        for fesible_route in fesible_routes_with_charge:
                            if fesible_route not in route_pool:
                                cm_ids = [id[0] for id in fesible_route if id[0] in community_info.keys()]
                                total_travel_dist, total_travel_time, charging_time = mf.find_route_dist_time(cur_insert)
                                total_route_cost = total_travel_dist * unit_dist_cost   # distance cost only
                                reduced_cost     = total_route_cost - sum(community_dual[cm] for cm in cm_ids)

                                if reduced_cost < -0.00001:
                                    list_good_routes.append(fesible_route)
                                    #dict_reduced_cost[tuple(fesible_route)] = reduced_cost
                                    route_pool.append(fesible_route)

    # # Sort routes by reduced cost
    # if list_good_routes:
    #     list_good_routes = sorted(list_good_routes, key=lambda x: dict_reduced_cost[tuple(x)])
    #
    #     # Find the index of the first positive reduced cost
    #     positive_index = next((i for i, route in enumerate(list_good_routes) if dict_reduced_cost[tuple(route)] >= 0), len(list_good_routes))
    #
    #     # Calculate the 50% index
    #     percentage_index = int(len(list_good_routes) * 1.0)
    #
    #     # Retain routes with reduced cost below the threshold
    #     list_good_routes = list_good_routes[:max(percentage_index, positive_index)]

    if len(list_good_routes) > 0:
        return True, list_good_routes
    else:
        return False, None


def pause_and_continue():
    while True:
        user_input = input("Enter 'y' to run next instance: ").lower()
        if user_input == 'y':
            break
        else:
            print("Waiting for 'y' to run next instance...")


# =============================================================================
# PHASE 1 — ROUTE GENERATION
# Runs the CG iterative loop (RMP + PSP) until no improving routes exist.
# Returns the finalized all_routes dictionary (fixed set Omega) and the
# GAMS workspace + database ready for Phase 2.
# =============================================================================

def generate_routes(node_file_path, travel_data_file_path,
                    service_time_file_path, waste_demand_file_path):
    """
    Phase 1: Route Generation (offline, once per planning period).
    Timing is computed and printed internally.
    """
    global travel_dist_dict, travel_time_dict
    global max_driving_range, min_driving_range
    global community_info, charge_station_info, truck_info, truck_cap
    global all_route_ids, all_routes, route_pool
    global unit_dist_cost, unit_time_cost, unit_charge_cost, charge_rate, min_charge_time

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    'Real-world parameter specifications'
    # 'Real-world parameter specifications'
    unit_dist_cost   = 6.50    # cost per mile driven
    unit_time_cost   = 0.45    # cost per routing time (currently unused in cost) #Parameter not used
    unit_charge_cost = 2.45    # cost per unit of charge (currently muted) #Parameter not used
    charge_rate      = 0.5     # miles per minute charging rate #Parameter not used
    min_charge_time  = 10       # minimum charging duration (minutes) #Parameter not used
    max_driving_range = 1600  # battery capacity (miles) #Parameter not used
    min_driving_range = 10     # minimum allowable SoC (miles) #Parameter not used
    fleet_size        = 10 #Parameter not used
    truck_cap         = 20000  # lbs


    max_cg_iter       = 30
    insert_strategy   = 'IS#1'
    init_max_cm_count = 2      # initial routes cover up to 2 communities

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    travel_data      = pd.read_csv(travel_data_file_path)
    node_data        = pd.read_csv(node_file_path)
    service_time_data= pd.read_csv(service_time_file_path)
    waste_demand_data= pd.read_csv(waste_demand_file_path)

    travel_dist_dict, travel_time_dict = mf.compute_shortest_path(travel_data)

    community_info      = dict()
    truck_info          = dict()
    charge_station_info = dict()

    for idx in range(len(node_data)):
        node_loc = node_data.loc[idx, 'Node ID']
        if node_data.loc[idx, 'Label'] == 'V':
            service_time = service_time_data.loc[
                service_time_data['Node'] == node_loc, 'Service Time'].iloc[0]
            waste_amount = waste_demand_data.loc[
                waste_demand_data['Node'] == node_loc, 'Demand'].iloc[0]
            community_info[f"cm{len(community_info) + 1}"] = {
                "cm_id":        f"cm{len(community_info) + 1}",
                "loc":          "i" + str(node_loc),
                "waste_amount": float(waste_amount),
                "service_time": float(service_time)
            }
        if node_data.loc[idx, 'Label'] == 'D':
            truck_info[f"truck{len(truck_info) + 1}"] = {
                "truck_id": f"truck{len(truck_info) + 1}",
                "loc":      "i" + str(node_loc),
            }
        if node_data.loc[idx, 'Label'] == 'C+':
            on_charge_node  = node_loc
            off_charge_node = int(
                travel_data.loc[travel_data['from_node ID'] == on_charge_node,
                                'to_node ID'].values[0])
            charge_station_info[f"cs{len(charge_station_info) + 1}"] = {
                "cs_id":          f"cs{len(charge_station_info) + 1}",
                "on_charge_node": "i" + str(on_charge_node),
                "off_charge_node":"i" + str(off_charge_node)
            }

    print(f"community_info: {community_info}")
    print(f"truck_info: {truck_info}")
    print(f"charge_station_info: {charge_station_info}")

    # ------------------------------------------------------------------ #
    # Initialize data structures
    # ------------------------------------------------------------------ #
    all_route_ids = {}
    all_routes    = {}
    route_pool    = []

    Time_1 = datetime.datetime.now()

    # ------------------------------------------------------------------ #
    # Step 1: Generate initial columns (1-community and 2-community routes)
    # ------------------------------------------------------------------ #
    nv1_route_ids = {}
    for truck_id in truck_info.keys():
        cur_truck_route_ids = []
        for cm_id in community_info.keys():
            route            = mf.generate_1v1(truck_id, cm_id)
            if_route_feasible= mf.check_route_feasibility(route)

            if if_route_feasible == True:
                if route not in route_pool:
                    route_pool.append(route)
                    route_id = mf.add_to_all_route(route, if_route_feasible, truck_id)
                    cur_truck_route_ids.append(route_id)
            else:
                if if_route_feasible == 'range_violation':
                    fesible_routes_with_charge = mf.insert_charge_station(route)
                    if len(fesible_routes_with_charge) > 0:
                        for fesible_route in fesible_routes_with_charge:
                            if fesible_route not in route_pool:
                                route_id = mf.add_to_all_route(fesible_route, True, truck_id)
                                cur_truck_route_ids.append(route_id)

        nv1_route_ids[truck_id] = cur_truck_route_ids

    all_route_ids[1] = nv1_route_ids

    for m in range(2, init_max_cm_count + 1):
        generate_nvm_routes(m)

    print("\nInitial columns generated:")
    for route_id, route_data in all_routes.items():
        print(f"Route ID: {route_id}, Route: {route_data['route']}")

    all_keys    = list(all_routes.keys())
    select_keys = all_keys[:len(all_routes)]
    updated_routes = {k: all_routes[k] for k in select_keys}

    # ------------------------------------------------------------------ #
    # Step 2: Set up GAMS workspace and database with initial columns
    # ------------------------------------------------------------------ #
    if len(sys.argv) > 1:
        ws = GamsWorkspace(system_directory=sys.argv[1], debug=DebugLevel.Verbose)
    else:
        ws = GamsWorkspace(debug=DebugLevel.KeepFiles)

    db = ws.add_database()

    r_set = db.add_set("r", 1, "candidate route set")
    [r_set.add_record(route_id) for route_id in all_routes.keys()]

    a_set = db.add_set("a", 1, "community id set")
    [a_set.add_record(f"cm{i}") for i in range(1, len(community_info) + 1)]

    fleet_size_par = db.add_parameter('fleet_size', 0, 'fleet size')
    fleet_size_par.add_record().value = fleet_size

    route_community_par = GamsParameter(db, "route_community", 2, "route community incidence")
    route_cost_par      = GamsParameter(db, "route_cost", 1, "route cost")

    for route_id, route_data in all_routes.items():
        for cm_id in route_data["cm_ids"]:
            route_community_par.add_record((route_data["route_id"], cm_id)).value = 1
        route_cost_par.add_record(route_data["route_id"]).value = route_data["route_cost"]

    Time_2 = datetime.datetime.now()
    print(f"Initial routes: {len(all_routes)}, Route pool size: {len(route_pool)}")

    # ------------------------------------------------------------------ #
    # Step 3: CG iterative loop — RMP + PSP
    # Terminates when no improving column (phi_w < 0) is found
    # after at least 15 iterations.
    # ------------------------------------------------------------------ #
    newly_added_col_dic = {}

    for CG_iter in range(max_cg_iter):

        promising_col_pool = []

        # --- Solve RMP (LP relaxation) ---
        cp  = ws.add_checkpoint()
        t3  = GamsJob(ws, source=rmip.get_model_txt_rmip())
        opt = GamsOptions(ws)
        opt.defines["gdxincname"] = db.name
        opt.all_model_types       = "Gurobi"

        t3.run(opt, databases=db, checkpoint=cp)
        t3 = ws.add_job_from_string(
            "solve MyModel minimizing z using rmip; ms=MyModel.modelstat; ss=MyModel.solvestat;", cp)
        t3.run(opt, databases=db)

        if not (t3.out_db["ms"].find_record().value == 1 and
                t3.out_db["ss"].find_record().value == 1):
            print("\n Modelstatus: " + str(t3.out_db["ms"].find_record().value))
            print(" Solvestatus: " + str(t3.out_db["ss"].find_record().value))
            raise ValueError('RMIP is infeasible. Please enable larger cliques in the initial routes')

        # --- Extract current basis and dual variables ---
        rmp_solution_listBasis = mf.get_basic_solution(t3)

        for slt_rte_id in rmp_solution_listBasis:
            if slt_rte_id not in promising_col_pool:
                promising_col_pool.append(slt_rte_id)
            print(f"Iter: {CG_iter}, obj: {t3.out_db['z'][()].level:.2f}, "
                  f"route: {all_routes[slt_rte_id]['route']}, "
                  f"route cost: {all_routes[slt_rte_id]['route_cost']}")

        # dual variables tau_v for community coverage constraints (eq1)
        community_dual = {community.keys[0]: community.marginal
                          for community in t3.out_db["eq1"]}

        # --- Solve PSP via insertion heuristic ---
        newly_added_col_dic[CG_iter] = []

        if insert_strategy == 'IS#1':
            basis_pool = rmp_solution_listBasis
        elif insert_strategy == 'IS#2':
            basis_pool = updated_routes.keys()
        else:
            basis_pool = promising_col_pool

        for each_col_basis in basis_pool:
            route_info      = all_routes[each_col_basis]
            existing_cm_ids = route_info["cm_ids"]
            route           = route_info["route"]
            truck_id        = route_info["truck_id"]

            for am_id_insert in community_info.keys():
                if am_id_insert not in existing_cm_ids:
                    ExistGoodRoute, list_good_routes = generate_best_route_by_one_insertion(
                        route, am_id_insert, community_dual)

                    if ExistGoodRoute:
                        for new_route in list_good_routes:
                            route_id = mf.add_to_all_route(new_route, True, truck_id)
                            newly_added_col_dic[CG_iter].append(route_id)
                            promising_col_pool.append(route_id)

        # --- Termination check ---
        # Stop after >= 15 iterations with zero new columns added
        if CG_iter >= 15 and len(newly_added_col_dic[CG_iter]) == 0:
            print('No improving routes found. Route generation complete.')
            print(f"Total routes in Omega: {len(all_routes)}")
            break

        # --- Add new columns to GAMS database for next iteration ---
        else:
            for route_id in newly_added_col_dic[CG_iter]:
                r_set.add_record(route_id)
                route_cost_par.add_record(route_id).value = all_routes[route_id]["route_cost"]
                for cm_id in all_routes[route_id]["cm_ids"]:
                    route_community_par.add_record((route_id, cm_id)).value = 1

            updated_routes = {k: all_routes[k] for k in newly_added_col_dic[CG_iter]}

    print(f"Route generation complete. Generated routes set (Omega) contains {len(all_routes)} routes.")
    print(f"Route generation time: {(datetime.datetime.now() - Time_2).seconds} seconds")

    Time_2 = datetime.datetime.now()
    print("=" * 60)
    print("PHASE 1 — ROUTE GENERATION COMPUTATION TIME SUMMARY")
    print("=" * 60)
    print('Total Route Generation time', (Time_2 - Time_1).seconds)
    #print(f"  Final size of generated routes (Omega)         : {len(all_routes)} routes")
    print("=" * 60)

    # Return finalized all_routes, workspace and database for Phase 2
    return all_routes, ws, db


# =============================================================================
# PHASE 2 — ROUTE SELECTION
# Given the fixed pre-generated set Omega (all_routes) from Phase 1,
# solve the integer program to select the optimal set of routes
# for a given fleet size.
# Can be called multiple times with different fleet sizes
# without re-running Phase 1.
# =============================================================================

def select_routes(all_routes, ws, db, fleet_size=None):
    """
    Phase 2: Route Selection.

    Takes the fixed route pool (all_routes) and the pre-populated GAMS
    database (db) from Phase 1 and solves the binary integer program:
    If fleet_size is provided, it overrides the value already in the database.
    Returns the list of selected route location sequences.
    """

    # Optionally override fleet size in the GAMS database
    if fleet_size is not None:
        db["fleet_size"].find_record().value = fleet_size
        print(f"Fleet size overridden to: {fleet_size}")

    Time_1 = datetime.datetime.now()

    # --- Solve MIP (integer program with fleet size constraint) ---
    cp  = ws.add_checkpoint()
    t4  = GamsJob(ws, source=exact.get_model_txt_mip())
    opt = GamsOptions(ws)
    opt.defines["gdxincname"] = db.name
    opt.all_model_types       = "Gurobi"

    t4.run(opt, databases=db, checkpoint=cp)
    t4 = ws.add_job_from_string(
        "solve MyModel minimizing z using mip; ms=MyModel.modelstat; ss=MyModel.solvestat;", cp)
    t4.run(opt, databases=db)

    if not (t4.out_db["ms"].find_record().value == 1 and
            t4.out_db["ss"].find_record().value == 1):
        print("\n Modelstatus: " + str(t4.out_db["ms"].find_record().value))
        print(" Solvestatus: " + str(t4.out_db["ss"].find_record().value))
        raise ValueError('MIP is infeasible')

    # --- Extract selected routes (x(r) > 0.1) ---
    var_x          = t4.out_db.get_variable("x")
    select_rte_list= [rec.keys[0] for rec in var_x if rec.level > 0.1]

    # --- Report results ---
    obj = 0
    for route_id in select_rte_list:
        obj += all_routes[route_id]['route_cost']
        print(f"Selected route: {all_routes[route_id]}")
        print("=" * 60)

    Time_2 = datetime.datetime.now()

    print("=" * 60)
    print(f"PHASE 2 — ROUTE SELECTION COMPUTATION TIME SUMMARY (fleet size = {fleet_size})")
    print("=" * 60)
    print(f"Objective (manual sum)  = {obj}")
    print(f"Objective (GAMS model)  = {t4.out_db['z'][()].level}")
    print(f"Fleet size used         = {fleet_size}")
    print(f"Number of routes selected: {len(select_rte_list)}")
    print(f"Route Selection time: {(Time_2 - Time_1).seconds} seconds")
    print("=" * 60)

    # Return selected route location sequences
    selected_routes = [all_routes[route_id]['route_loc'] for route_id in select_rte_list]
    return selected_routes



