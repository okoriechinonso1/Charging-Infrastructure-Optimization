###############################################################################
' Exact algorithm Model solved with GUROBI for the EVRP'

###############################################################################
'Import the necessary functions from the main_v3.py script'
from synthetic_scripts import evrp_problem
from synthetic_scripts import plot_networkx_graph, generate_plots
import pandas as pd
import networkx as nx
from pathlib import Path
from synthetic_scripts import charging_problem
from  synthetic_scripts import load_charging_arc_params


"Joint Optimization (routing and charging) problem"

def main():

   # ############################################################################################
    "Joint Routing and Charging Optimization Model - Lower_Level Problem Instance Run"

    'Set the file paths for the input data'
    folder_name = 'synthetic_data_1' # Change to instance folder name if running an instance for the real world case study
    data_dir = Path(__file__).resolve().parent / "synthetic_data" / folder_name

    evrp_main_data = str(data_dir / "complete_travel_data.csv")
    nodes_data_file = str(data_dir / "complete_nodes.csv")
    service_time_file = str(data_dir / "service_time.csv")
    waste_demand_file = str(data_dir / "waste_demand.csv")
    charging_links_file = str(data_dir / "charging_links.csv")

    selected_charging_links = [1, 2, 3, 4]

    "Real-world parameter specifications"
    # === Step 3: Run the GAMS model using filtered files ===
    # driving_range = 150
    # tour_duration = 480
    # no_of_vehicles = 2
    # vehicle_capacity = 20000
    # depot = '121'
    # lowest_range = 30
    # delta_min_value = 10
    # delta_max_value = 240
    # beta_0 = 2.20 # routing distance cost factor (or cost coefficient)
    # Big_M = 1000

    'Synthetic parameter specifications'
    driving_range = 50
    tour_duration = 480
    no_of_vehicles = 2
    vehicle_capacity = 380
    minimum_waste = 60
    depot = '1'
    lowest_range = 10
    delta_min_value = 0.5
    delta_max_value = 240
    SoC_reduction_rate = 1.0
    beta_0 = 2.50 # routing distance cost factor (or cost coefficient)
    Big_M = 1000


    # --- All subsequent runs: read from saved file ---
    beta1_dict, alpha_dict = load_charging_arc_params(str(data_dir / "charging_arcs_params.csv"),
        selected_charging_links
    )

    print('\n')

    "Call and run the Gams model for the joint optimization problem"

    'symmetry breaking constraint approach'
    "Call and run the Gams model for the joint optimization problem"
    (job, db, ws, feasible, arc_travel, arc_cost_dict, time_level, SoC_level, waste_level, visited_customer_nodes,
     SolutionTime, objective_function_value, Total_travel_distance, Total_charging_time, Total_service_time, Total_waste_amount, dict_computation_time) = evrp_problem(
           evrp_main_data,
           service_time_file,
           waste_demand_file,
           nodes_data_file,
           driving_range, tour_duration, no_of_vehicles, vehicle_capacity, minimum_waste, SoC_reduction_rate, depot,
           lowest_range, delta_min_value, delta_max_value,
           beta_0, beta1_dict, alpha_dict, Big_M
       )


    print(f"Database: {db}")
    print(f"Workspace: {ws}")
    print(f"Is Feasible: {feasible}")

    # Generate the networkx plot and the performance indicator plots
    depot_nodes = plot_networkx_graph(evrp_main_data, arc_travel, nodes_data_file)
    # depot_nodes = plot_networkx_graph_from_routes(evrp_main_data, nodes_data_file, routes)

    # Specify the arguments for generate_plots function
    max_load = vehicle_capacity  # Set to the maximum expected vehicle load
    max_soc = driving_range  # Set to the maximum expected SoC level
    max_time = tour_duration # Set to the maximum expected node visit time
    vehicle_tour_paths = generate_plots(arc_travel, time_level, SoC_level, depot_nodes, max_soc, max_time)
    print(vehicle_tour_paths)
    print('Visited customers are:', visited_customer_nodes)
    print('arcs travelled by vehicle:', arc_travel)

    # Print summary of results
    print("\n--- Summary of Optimization Model Results ---")
    print(f"Objective Function Value ($): {objective_function_value}")
    print("Total Travel Distance:")
    for vehicle, distance in Total_travel_distance.items():
        print(f"  Vehicle {vehicle}: {distance} miles")
    print("Total Charging Time:")
    for vehicle, charging_time in Total_charging_time.items():
        print(f"  Vehicle {vehicle}: {charging_time} minutes")
    print("Total Service Time:")
    for vehicle, service_time in Total_service_time.items():
        print(f"  Vehicle {vehicle}: {service_time} minutes")
    print("Total Waste Amount:")
    for vehicle, waste_amount in Total_waste_amount.items():
        print(f"  Vehicle {vehicle}: {waste_amount} pounds")


    print('\n')
    print(f"Solution Time: {SolutionTime} seconds")

    'Display the the GAMS model attributes for the solver'
    print('\n')
    print('Dual or Upper bound:', dict_computation_time['objest_dual_bound'])
    print('elapsed time taken by the solver only', dict_computation_time['etSolver'])
    print('Number of equations', dict_computation_time['numEqu'])
    print('Number of variables', dict_computation_time['numVar'])
    print('Number of discrete variables', dict_computation_time['numDVar'])

#=== Run the Main Function ===
if __name__ == "__main__":
    main()

#################################################################################################################


"Charging-second optimization problem"

# def main():
#      # ---------------------------------------------------------------
#      # CONFIGURE HERE: set your charging links and number of routes
#      # ---------------------------------------------------------------
#
#      folder_name = 'synthetic_data_1'  # Change to instance folder name if running an instance for the real world case study
#      selected_charging_links = [1, 2, 3, 4]  # Change these numbers for the specific configuration you want to solve
#      num_routes = 2  # total number of routes to solve
#      fleet_size = 2
#
#      'Set the file paths for the input data'
#      data_dir = Path(__file__).resolve().parent / "synthetic_data" / folder_name
#      period_folder = data_dir / f"augmented_routes_for_charging_{fleet_size}"  # Data folder for this period and fleet size
#      charging_links_file = str(data_dir / "charging_links.csv")
#
#      "Model Instance parameters"
#      driving_range = 50
#      tour_duration = 480
#      vehicle_capacity = 380
#      depot = '1'
#      lowest_range = 10
#      delta_min_value = 0.5
#      delta_max_value = 240
#      SoC_reduction_rate = 1.0
#      cost_per_mile = 2.50 # routing distance cost factor (or cost coefficient)
#      Big_M = 1000
#      beta_0 = cost_per_mile
#
#      # ---------------------------------------------------------------
#      # Load charging arc parameters from saved CSV (consistent values)
#      # ---------------------------------------------------------------
#      beta1_dict, alpha_dict = load_charging_arc_params(
#       str(data_dir / "charging_arcs_params.csv"),
#       selected_charging_links
#      )
#      print(f"\nLoaded charging arc parameters for charging links: {selected_charging_links}")
#
#      # ---------------------------------------------------------------
#      # Accumulators
#      # ---------------------------------------------------------------
#      total_travel_distance_cost = 0.0
#      total_charging_cost = 0.0
#      total_travel_distance = 0.0
#      total_charging_time = 0.0
#      total_waste = 0.0
#      failed_routes = []
#
#      print(f"\nSolving {num_routes} routes...")
#      print("=" * 60)
#
#      # ---------------------------------------------------------------
#      # Main loop: route_num = iteration index (1 to num_routes)
#      # ---------------------------------------------------------------
#      for route_num in range(1, num_routes + 1):
#
#       print(f"\n--- Route {route_num} of {num_routes} ---")
#
#       # Build file paths for this route
#       route_prefix = f"route_{route_num}_aug"
#       evrp_main_data = str(period_folder / f"{route_prefix}_travel.csv")
#       nodes_data_file = str(period_folder / f"{route_prefix}_nodes.csv")
#       service_time_file = str(period_folder / f"{route_prefix}_service_time.csv")
#       waste_demand_file = str(period_folder / f"{route_prefix}_waste_demand.csv")
#
#       try:
#        # Run the charging-second optimization for this route
#        (job, db, ws, feasible, arc_travel, arc_cost_dict, time_level, SoC_level, waste_level,
#         visited_customer_nodes, SolutionTime, objective_function_value,
#         Total_travel_distance, Total_charging_time, Total_waste_amount, dict_computation_time) = charging_problem(
#         evrp_main_data,
#         service_time_file,
#         waste_demand_file,
#         nodes_data_file,
#         driving_range, tour_duration, vehicle_capacity, SoC_reduction_rate, depot,
#         lowest_range, delta_min_value, delta_max_value, cost_per_mile,
#         beta1_dict, alpha_dict, Big_M
#        )
#
#        if not feasible:
#         print(f"  WARNING: Route {route_num} returned infeasible solution.")
#         failed_routes.append(route_num)
#         continue
#
#        # Per-route costs
#        route_travel_cost = Total_travel_distance * cost_per_mile
#        route_charging_cost = objective_function_value
#
#        # Accumulate
#        total_travel_distance += Total_travel_distance
#        total_travel_distance_cost += route_travel_cost
#        total_charging_cost += route_charging_cost
#        total_charging_time += Total_charging_time
#        total_waste += Total_waste_amount
#
#        # Per-route summary
#        print(f"  Feasible         : {feasible}")
#        print(f"  Travel distance  : {Total_travel_distance:.2f} miles")
#        print(f"  Travel distance cost      : ${route_travel_cost:.2f}")
#        print(f"  Charging time cost    : ${route_charging_cost:.2f}")
#        print(f"  Charging time    : {Total_charging_time:.2f} minutes")
#        print(f"  Waste collected  : {Total_waste_amount:.2f} lbs")
#        print(f"  Solution time    : {SolutionTime} seconds")
#
#       except Exception as e:
#        print(f"  ERROR on route {route_num}: {e}")
#        failed_routes.append(route_num)
#        continue
#
#      # ---------------------------------------------------------------
#      # Final summary across all routes
#      # ---------------------------------------------------------------
#      solved_routes = num_routes - len(failed_routes)
#      total_daily_operating_cost = total_travel_distance_cost + total_charging_cost
#
#      print("\n" + "=" * 60)
#      print("=== SUMMARY: All Routes ===")
#      print("=" * 60)
#      print(f"  Charging links used      : {selected_charging_links}")
#      print(f"  Routes attempted         : {num_routes}")
#      print(f"  Routes solved            : {solved_routes}")
#      if failed_routes:
#       print(f"  Failed routes            : {failed_routes}")
#      print(f"  Total travel distance    : {total_travel_distance:.2f} miles")
#      print(f"  Total travel distance cost        : ${total_travel_distance_cost:.2f}")
#      print(f"  Total charging time cost      : ${total_charging_cost:.2f}")
#      print(f"  Total charging time      : {total_charging_time:.2f} minutes")
#      print(f"  Total waste collected    : {total_waste:.2f} lbs")
#      print("-" * 60)
#      print(f"  TOTAL DAILY OPERATING COST: ${total_daily_operating_cost:.2f}")
#      print(f"  TOTAL PERIOD COST : ${total_daily_operating_cost * 1:.2f}")
#      print("=" * 60)
#
#      return total_daily_operating_cost
#
#
# # === Run the Main Function ===
# if __name__ == "__main__":
#  main()

