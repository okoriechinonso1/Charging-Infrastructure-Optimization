from gams import *
import os
import sys
import time
import csv

'''
Electric Vehicle Routing Problem as MIP
'''


def get_model_text():
    return '''
Scalar ms 'model status', ss 'solve status', objest 'dual bound', etSolver 'elapsed time taken by the solver only';
Scalar numEqu 'Number of equations', numVar	'Number of variables', numDVar 'Number of discrete variables';
*$onUELlist
Set
    i 'set of all nodes in the network'

*Let GAMS know that we will be using i and j interchangeably
Alias (i,j);

Set
    v(i)  'set of customer nodes in the network'
    s_plus(i) 'set of on recharge nodes'
    s_minus(i) 'set of off recharge nodes'
    H(i,j) 'Set of predecessors of node i'
    G(i,j) 'Set of successors of node i'
    graph(i,j) 'Set of arcs in the network'
    map(i,j) 'Subset of arcs that are recharge arcs in the network'
;

Parameter

    d(i) Define Waste collection Demand at each customer node i and read the values from a text file
    T(i) Define Service duration for node i and read the values from a text file
    c(i,j) Define travel distances between nodes i to node j and read the values from a text file
    ti(i,j) Define travel time between nodes i and node j and read the values from a text file
    Q Define load capacity of the vehicles and read the values from a tet file
    R Define Battery capacity of the vehicle
    u Minimum Allowable SoC level of an ET before recharge
    delta_min Minimum travel time through the recharge arc
    delta_max Maximum travel time through the recharge arc
    beta0 Routing cost per mile
    sr Soc reduction rate ie the amount of SoC reduced per mile traveled
    beta1(i,j) charging cost per charging arc
    alpha(i,j) Battery recharging rate through the charge arc
    L_max  Maximum route duration of a vehicle
    M A very large constant number
;


$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$load i v s_minus s_plus G H graph map d T c ti Q R u delta_min delta_max beta0 sr beta1 alpha L_max M
$gdxin

*Define our binary decision variables (i.e., 0 <= x_ij <= 1)
Binary Variables
    x(i,j) 'whether an arc (i,j) is traversed by vehicle k or not'
;

*Define our continuous decision variables (i.e rho_i >= 0, gammar_i >= 0, theta_i >= 0)
Positive Variable
        gammar(i) 'continous variable indicating load of vehicle k node i'
        theta(i)  'Continous variable indicating the state of charge of the battery (SoC) of vehicle k at node i'
        rho(i) 'Time at which vehicle k begin service at node i'
        pt 'sum of travel distances between nodes i and nodes j'
        cd 'Total charging time of all used vehicles'
        st 'Total service time of customers'
        wA 'Total amount of waste picked up by an ET'
;

*Define the objective function as a variable as well (tol_travel_distance will stand here for total cost)
Variable
    tol_travel_cost
;

*Specify the equations (i.e., objective and constraints) that we will have in our math program
Equation
def_tol_travel_cost
depot_start
depot_end
exclusive_assignment
self_travel
flow_conservation
time_feasibility_one
time_feasibility_two
charging_duration
time_none_violation
node_battery_resource_one
node_battery_resource_two
depot_full_charge
battery_lower_bound
battery_upper_bound
travel_distance_constraint
charging_time_constraint
total_collected_waste
;


* Objective function
def_tol_travel_cost.. tol_travel_cost =e= sum((i,j) $ map(i,j), beta1(i,j) * (rho(j) - rho(i)));
* beta0 * sum((i,j) $ graph(i,j), c(i,j) * x(i,j)) +

* Vehicle Routing and Flow Conservation Constraints
*depot_start.. sum(i $ (v(i) or s_plus(i)), x('i151+',i)) =e= 1;
*depot_end.. sum(i $ (v(i) or s_minus(i)), x(i,'i151-')) =e= 1;
*exclusive_assignment(i) $ v(i).. sum(j $ ((not sameas(j,'i1+')) and graph(i,j)), x(i,j)) =e= 1;
*self_travel(i,i).. x(i,i) =e= 0;
*flow_conservation(i) $ (v(i) or s_plus(i) or s_minus(i)).. sum(j $ graph(i,j), x(i,j)) =e= sum(j $ graph(j,i), x(j,i));

depot_start.. sum(i $ G('i151+',i), x('i151+',i)) =e= 1;
depot_end.. sum(i $ H('i151-',i), x(i,'i151-')) =e= 1;
exclusive_assignment(i) $ v(i) ..sum(j $ G(i,j), x(i,j)) =e= 1;
self_travel(i,i).. x(i,i) =e= 0;
flow_conservation(i) $ (v(i) or s_plus(i) or s_minus(i)).. sum(j $ G(i,j), x(i,j)) =e= sum(j $ H(i,j), x(j,i));

* Time Feasibility Constraints
time_feasibility_one(i,j) $ (graph(i,j) and not map(i,j)).. rho(j) =g= rho(i) + ti(i,j) + T(i) - M * (1 - x(i,j));
time_feasibility_two(i,j) $ (graph(i,j) and not map(i,j)).. rho(j) =l= rho(i) + ti(i,j) + T(i) + M * (1 - x(i,j));
charging_duration(i,j) $ map(i,j).. rho(j) =g= rho(i) + delta_min * x(i,j);
time_none_violation(i).. rho(i) =g= 0;

* Battery Level and Charging Constraints
node_battery_resource_one(i,j) $ graph(i,j).. theta(j) =g= theta(i) - c(i,j) + alpha(i,j)*(rho(j) - rho(i)) - R * (1 - x(i,j));
node_battery_resource_two(i,j) $ graph(i,j).. theta(j) =l= theta(i) - c(i,j) + alpha(i,j)*(rho(j) - rho(i)) + R * (1 - x(i,j));
depot_full_charge.. theta('i151+') =e= R;
battery_upper_bound(i).. theta(i) =l= R + M * (1 - sum(j $ graph(i,j), x(i,j)));
battery_lower_bound(i).. theta(i) =g= u - M * (1 - sum(j $ graph(i,j), x(i,j)));

* Additional Constraints
travel_distance_constraint.. pt =e= sum((i,j) $ graph(i,j), c(i,j) * x(i,j));
charging_time_constraint.. cd =e= sum((i,j) $ map(i,j), (rho(j) - rho(i)));
total_collected_waste.. wA =e= sum(i $ v(i), d(i) * sum(j, x(i,j)));

*Give a name to the above model
Model EVRP / all /;
* Define the model
*Specify the solver to be used
option MIP = Gurobi;
*Gurobi;
*Xpress;
*Cplex;
* Try Xpress or Gurobi solver

*options. Make sure MIP solver finds global optima.
option optcr = 0.00;
*option Seed = 0;
*option threads = 16;
*option reslim = 100;
*option limrow = 5;
*option limcol = 5;

*Solve the model while indicating this is a MINIMIZATION problem
*Solve EVRP minimizing tol_travel_cost using MIP;
'''


def charging_problem(evrp_main_data, service_time_file, waste_demand_file, nodes_data_file, driving_range, tour_duration,
                 vehicle_capacity, SoC_reduction_rate, depot,lowest_range, delta_min_value, delta_max_value, beta_0, beta1_dict, alpha_dict,  Big_M): # beta_1, alpha_value,

    # prepare necessary input data
    network_nodes_set = []
    customers_node_set = []  # This is defined as set of customer nodes
    customers_id = []
    recharge_arc_start_set = []
    recharge_arc_end_set = []
    insert_start_nodes = []
    insert_end_nodes = []
    depot_node = 'i' + depot
    l_max = tour_duration

    # --- Build allowed charging node numbers from beta1_dict ---
    # beta1_dict keys look like ('iC124+', 'iC125-')
    # Extract the node numbers from these keys
    allowed_start_nodes = set()
    allowed_end_nodes = set()

    for (arc_start, arc_end) in beta1_dict.keys():
        # arc_start = 'iC124+' → extract '124'
        start_num = arc_start.replace('iC', '').replace('+', '')
        end_num = arc_end.replace('iC', '').replace('-', '')
        allowed_start_nodes.add(start_num)
        allowed_end_nodes.add(end_num)

    existing_nodes = set()  # Set to store nodes present in the links
    with open(evrp_main_data, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            from_node = 'i' + str(int(float(row[0])))  # Add 'i' prefix to match the node format
            to_node = 'i' + str(int(float(row[1])))  # Add 'i' prefix to match the node format
            existing_nodes.add(from_node)
            existing_nodes.add(to_node)


    # Process nodes from the processed_nodes.csv file
    with open(nodes_data_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            node_number = row[0]
            node_number = str(int(float(node_number)))
            label = row[3]  # Assuming label is in the fourth column

            # Construct the node identifier with 'i' prefix
            node_id = 'i' + node_number

            # Check if the node exists in the links
            if node_id not in existing_nodes:
                continue  # Skip nodes that do not exist in the links

            # Populate the sets based on the label
            if label == 'V':  # Customer node
                customers_node_set.append(node_id)
                customers_id.append(node_number)
                network_nodes_set.append(node_id)

            elif label == 'D':  # Depot node
                network_nodes_set.append(node_id + '+')
                network_nodes_set.append(node_id + '-')

            # elif label == 'C+':
            #     recharge_arc_start_set.append('iC' + node_number + '+')
            #     insert_start_nodes.append(node_number)
            #     network_nodes_set.append('iC' + node_number + '+')

            elif label == 'C+':
                # Only include if this charging-on node is in the selected links
                if node_number in allowed_start_nodes:
                    recharge_arc_start_set.append('iC' + node_number + '+')
                    insert_start_nodes.append(node_number)
                    network_nodes_set.append('iC' + node_number + '+')

            # elif label == 'C-':
            #     recharge_arc_end_set.append('iC' + node_number + '-')
            #     insert_end_nodes.append(node_number)
            #     network_nodes_set.append('iC' + node_number + '-')

            elif label == 'C-':
                # Only include if this charging-off node is in the selected links
                if node_number in allowed_end_nodes:
                    recharge_arc_end_set.append('iC' + node_number + '-')
                    insert_end_nodes.append(node_number)
                    network_nodes_set.append('iC' + node_number + '-')


    # Reading the data for cost of travel between edges
    # Initialize an empty dictionary to store the data
    arc_cost_dict = {}

    # Open the CSV file for reading
    with open(evrp_main_data, "r") as file:
        reader = csv.reader(file)
        # Skip the header row
        next(reader)
        # Read each row in the file
        for row in reader:
            if len(row) == 4:
                node1, node2, travel_distance, travel_time = row
                key = (str(int(float(node1))), str(int(float(node2))))

                # Use conditional statements to add depot links and recharge arcs to the cost matrix

                # Depot-start to charging-on node connections and reverse
                if key[0] == depot and key[1] in insert_start_nodes:
                    arc_cost_dict[('i' + key[0] + '+', 'iC' + key[1] + '+')] = [float(travel_distance),
                                                                                float(travel_time)]

                # Charging-off nodes to depot-end connections and reverse
                elif key[0] in insert_end_nodes and key[1] == depot:
                    arc_cost_dict[('iC' + key[0] + '-', 'i' + key[1] + '-')] = [float(travel_distance),
                                                                                float(travel_time)]

                # Depot-start to customer node connections and reverse
                elif key[0] == depot and key[1] in customers_id:
                    arc_cost_dict[('i' + key[0] + '+', 'i' + key[1])] = [float(travel_distance), float(travel_time)]

                # Customer node to depot-end node connections and reverse
                elif key[0] in customers_id and key[1] == depot:
                    arc_cost_dict[('i' + key[0], 'i' + key[1] + '-')] = [float(travel_distance), float(travel_time)]

                # Charging-on node to charging--off node connections and reverse
                elif key[0] in insert_start_nodes and key[1] in insert_end_nodes:
                    arc_cost_dict[('iC' + key[0] + '+', 'iC' + key[1] + '-')] = [float(0), float(0)]

                # Customer nodes to charging-on nodes connections and reverse
                elif key[0] in customers_id and key[1] in insert_start_nodes:
                    arc_cost_dict[('i' + key[0], 'iC' + key[1] + '+')] = [float(travel_distance), float(travel_time)]

                # Charging-off nodes to customer nodes connections and reverse
                elif key[0] in insert_end_nodes and key[1] in customers_id:
                    arc_cost_dict[('iC' + key[0] + '-', 'i' + key[1])] = [float(travel_distance), float(travel_time)]

                # Customer to customer nodes connections and reverse
                elif key[0] in customers_id and key[1] in customers_id:
                    arc_cost_dict[('i' + key[0], 'i' + key[1])] = [float(travel_distance), float(travel_time)]

                    arc_cost_dict[('i' + key[1], 'i' + key[0])] = [float(travel_distance), float(travel_time)]


                # else:
                #     arc_cost_dict[('i' + key[0], 'i' + key[1])] = [float(travel_distance), float(travel_time)]

    # Prepare data for predecessors and successors sets
    # Initialize dictionaries for incoming and outgoing nodes
    successors = {}
    predecessors = {}

    # Loop through each edge to populate the dictionaries
    for (origin, destination), weight in arc_cost_dict.items():
        # Update outgoing neighbors (successors)
        if origin not in successors:
            successors[origin] = ()
        successors[origin] += (destination,)

        # Update incoming neighbors (predecessors)
        if destination not in predecessors:
            predecessors[destination] = ()
        predecessors[destination] += (origin,)

    # Set up GAMS workspace and database
    if len(sys.argv) > 1:
        ws = GamsWorkspace(system_directory=sys.argv[1], debug=DebugLevel.Verbose)
    else:
        ws = GamsWorkspace(debug=DebugLevel.KeepFiles)

    # Create Gams database
    db = ws.add_database()  # Creating an instance of the GAMSDatabase

    # Alternatively creating  GAMSDatabase instance to import data from existing GAMS GDX file
    # add a new GamsDatabase and initialize it from the GDX file just created
    # db2 = ws.add_database_from_gdx("out_GamsDatabase.gdx")

    # Define Gams Sets
    i = db.add_set("i", 1, "network_nodes")
    v = db.add_set('v', 1, 'customer_nodes')
    H = db.add_set('H', 2, 'Predecessors of node i')
    G = db.add_set('G', 2, 'Successors of node i')
    gams_graph = db.add_set("graph", 2, "Graph network")
    gams_map = db.add_set("map", 2, "recharge on to off node mapping")
    s_plus = db.add_set("s_plus", 1, "start charge nodes")
    s_minus = db.add_set("s_minus", 1, "end charge nodes")

    # Define Gams Parameters
    c = db.add_parameter("c", 2, "travel cost matrix")
    ti = db.add_parameter('ti', 2, 'travel time matrix')
    d = db.add_parameter('d', 1, 'demand of node i')
    T = db.add_parameter('T', 1, 'service duration of node i')

    # Define Gams Scalars
    R = db.add_parameter('R', 0, 'Battery capacity of an EV')
    u = db.add_parameter('u', 0, 'Remaining Battery range before recharge')
    sr = db.add_parameter('sr', 0, 'SoC reduction rate per mile traveled')
    delta_min = db.add_parameter('delta_min', 0, 'Minimum travel time through the recharge arc')
    delta_max = db.add_parameter('delta_max', 0, 'Maximum travel time through the recharge arc')
    Q = db.add_parameter('Q', 0, 'Vehicle load capacity')
    alpha = db.add_parameter('alpha', 2, 'Miles per minute speed of the vehicle')
    beta0 = db.add_parameter('beta0', 0, 'Routing cost per mile')
    beta1 = db.add_parameter('beta1', 2, 'Charging cost per charging arc')
    L_max = db.add_parameter('L_max', 0, 'Maximum duration of route for a vehicle')
    M = db.add_parameter('M', 0, 'A very sufficient large number')

    # Adding elements to the set of all nodes in the network
    for node_id in network_nodes_set:
        i.add_record(node_id)

    # Adding elements to the set of customer nodes in the network
    for customer_node_id in customers_node_set:
        v.add_record(customer_node_id)

    # Adding elements to the set of start nodes of recharge arcs
    for start_charge_node in recharge_arc_start_set:
        s_plus.add_record(start_charge_node)

    # Adding elements to the set of end nodes of recharge arcs
    for end_charge_node in recharge_arc_end_set:
        s_minus.add_record(end_charge_node)

    for i, j_values in predecessors.items():  # Set B(i,j) which is the set of predecessors of node i
        for j in j_values:
            H.add_record((i, j))

    for i, j_values in successors.items():  # Set G(i,j) which is the set of successors of node i
        for j in j_values:
            G.add_record((i, j))

    # Open and read the CSV file
    with open(waste_demand_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        # Create a dictionary to map node identifiers to their service times
        waste_demand_map = {}
        for row in reader:
            node_id = 'i' + str(int(float(row[0])))  # Add 'i' prefix to node ID
            waste_demand = float(row[1])  # Convert service time to float
            waste_demand_map[node_id] = waste_demand  # Store in the dictionary

    # Ensure that all customer nodes have a corresponding service time
    for node_i in customers_node_set:
        if node_i in network_nodes_set and node_i in waste_demand_map:
            # Assign the service time to the node if it exists in the service time map
            waste_demand = waste_demand_map[node_i]
            d.add_record(node_i).value = waste_demand  # Example of storing the data
        else:
            d.add_record(node_i).value = 0

    # Open and read the CSV file
    with open(service_time_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        # Create a dictionary to map node identifiers to their service times
        service_time_map = {}
        for row in reader:
            node_id = 'i' + str(int(float(row[0])))  # Add 'i' prefix to node ID
            service_time = float(row[1])  # Convert service time to float
            service_time_map[node_id] = service_time  # Store in the dictionary

    # Ensure that all customer nodes have a corresponding service time
    for node_i in customers_node_set:
        if node_i in network_nodes_set and node_i in service_time_map:
            # Assign the service time to the node if it exists in the service time map
            service_time = service_time_map[node_i]
            T.add_record(node_i).value = service_time  # Example of storing the data
        else:
            T.add_record(node_i).value = 0

    # Add values to the travel cost (c_ij) and travel time (t_ij) parameters defined in gams
    for from_node_id in network_nodes_set:
        for to_node_id in network_nodes_set:
            if (from_node_id, to_node_id) in arc_cost_dict:
                if from_node_id in recharge_arc_start_set and to_node_id in recharge_arc_end_set:
                    c.add_record((from_node_id, to_node_id)).value = arc_cost_dict[(from_node_id, to_node_id)][0]
                    ti.add_record((from_node_id, to_node_id)).value = arc_cost_dict[(from_node_id, to_node_id)][1]
                    gams_map.add_record((from_node_id, to_node_id))
                    gams_graph.add_record((from_node_id, to_node_id))


                else:
                    c.add_record((from_node_id, to_node_id)).value = arc_cost_dict[(from_node_id, to_node_id)][0]
                    ti.add_record((from_node_id, to_node_id)).value = arc_cost_dict[(from_node_id, to_node_id)][1]
                    gams_graph.add_record((from_node_id, to_node_id))


    for (from_node_id, to_node_id) in arc_cost_dict.keys():
        if from_node_id in recharge_arc_start_set and to_node_id in recharge_arc_end_set:
            # charging arcs → MUST exist in input dictionaries
            if (from_node_id, to_node_id) not in beta1_dict:
                raise ValueError(f"Missing beta1 value for arc {(from_node_id, to_node_id)}")
            if (from_node_id, to_node_id) not in alpha_dict:
                raise ValueError(f"Missing alpha value for arc {(from_node_id, to_node_id)}")

            beta1.add_record((from_node_id, to_node_id)).value = beta1_dict[(from_node_id, to_node_id)]
            alpha.add_record((from_node_id, to_node_id)).value = alpha_dict[(from_node_id, to_node_id)]

        else:
            # non-charging arcs → alpha MUST be zero
            alpha.add_record((from_node_id, to_node_id)).value = 0.0

    # Add record to other dimensional parameters
    R.add_record().value = driving_range
    u.add_record().value = lowest_range
    sr.add_record().value = SoC_reduction_rate
    delta_min.add_record().value = delta_min_value
    delta_max.add_record().value = delta_max_value
    beta0.add_record().value = beta_0
    Q.add_record().value = vehicle_capacity
    L_max.add_record().value = l_max
    M.add_record().value = Big_M

    """If I need to check model status"""
    # Create and run the GAMS job
    cp = ws.add_checkpoint()  # Checking model status
    GAMS_MODEL = get_model_text()
    job = ws.add_job_from_string(GAMS_MODEL)
    opt = ws.add_options()  # Alternatively GamsOptions(ws)
    opt.defines["gdxincname"] = db.name
    opt.all_model_types = "Gurobi"  # "Gurobi"


    print("Solving EVRP Problem as MIP_______________:")
    """Run Gams job using set gams options, gams database, create checkpoint and create gams output database"""
    job.run(opt, databases=db, checkpoint=cp, create_out_db=True,) #  output=sys.stdout

    # Start the timer
    start_time = time.time()

    job = ws.add_job_from_string(
        "solve EVRP minimizing tol_travel_cost using mip; ms=EVRP.modelstat; ss=EVRP.solvestat;"
        " objest=EVRP.objest; etSolver=EVRP.etSolver; numEqu=EVRP.numEqu;  numDVar=EVRP.numDVar;"
        "  numVar=EVRP.numVar; ", cp)
    job.run(opt, databases=db)
    # Measure the elapsed time
    elapsed_time = time.time()

    # Capture GAMSJob log output in a file named 'EVRPModel_log'
    # with open(os.path.join(ws.working_directory, "gurobi.opt"), "w") as file: # use xpress in case the solver you are using is xpress
    #     file.write("algorithm=barrier")
    # opt.optfile = 1
    # with open("scripts/ChargingProblem_log.log", "w") as log:
    #     job.run(opt, output=log)

    # For printing the log output of the GAMS model and not saving to log
    # job.run(opt, output=sys.stdout)

    # Using an absolute path to export the GamsDatabase to a GDX file with name 'out_GamsDatabase.gdx'
    # # located in the working_directory of the GAMSWorkspace
    # current_directory = os.getcwd()  # Verify my working directory
    # export_path = os.path.join(current_directory, "scripts/Charging_GamsDatabase.gdx")#out_GamsDatabase
    # print(f"Exporting GAMS database to: {export_path}")
    # db.export(export_path)

    # Alternatively store the GamsDatabase in the same directory as the GAMS script
    # export_path = os.path.join(current_directory, "out_GamsDatabase.gdx")
    # db.export(export_path)

    if not (job.out_db["ms"].find_record().value == 1 and job.out_db["ss"].find_record().value == 1):
        pass
    # if not (job.out_db["ms"].find_record().value == 1 or job.out_db["ms"].find_record().value == 8) and (job.out_db["ss"].find_record().value == 1 or job.out_db["ss"].find_record().value == 3):
    #     pass

        # raise ValueError('MIP Model is infeasible')
    feasible = job.out_db["ms"].find_record().value == 1 and job.out_db["ss"].find_record().value == 1
    # feasible = (job.out_db["ms"].find_record().value == 1 or job.out_db["ms"].find_record().value == 8) and (job.out_db["ss"].find_record().value == 1 or job.out_db["ss"].find_record().value == 3)
    dict_computation_time = {}
    dict_obj_beneficial = {}

    dict_computation_time['objest_dual_bound'] = job.out_db['objest'].find_record().value
    dict_computation_time['etSolver'] = job.out_db['etSolver'].find_record().value
    dict_computation_time['numEqu'] = job.out_db['numEqu'].find_record().value
    dict_computation_time['numDVar'] = job.out_db['numDVar'].find_record().value
    dict_computation_time['numVar'] = job.out_db['numVar'].find_record().value

    # Display the decision variables
    arc_travel = []
    visited_customer_nodes = set()
    print('\nDisplaying the selected travel links:')
    for rec in job.out_db["x"]:
        if rec.level >= 0.1:
            print("x(" + rec.key(0) + "," + rec.key(1) + "): level=" + str(rec.level))
            arc_travel.append((rec.key(0), rec.key(1)))

            for node in (rec.key(0), rec.key(1)):
                if node in customers_node_set:
                    visited_customer_nodes.add(node)
            # if rec.key(0) in (depot_node + '-', depot_node + '+') or rec.key(1) in (depot_node + '-', depot_node + '+'):
            #     continue
            # if rec.key(0) in customers_node_set or rec.key(1) in customers_node_set:
            #     visited_customer_nodes.add(rec.key(0))
            #     visited_customer_nodes.add(rec.key(1))
    print('arcs travelled by vehicle:', arc_travel)
    print('Visited customers are:', visited_customer_nodes)


    SoC_level = {}
    # Loop to display values of theta_variable
    print('\nDisplaying the battery level:')
    for rec in job.out_db["theta"]:
        if rec.level != 0.0:  # Display non-null values
            print("theta(" + rec.key(0) + "): level=" + str(rec.level))
            SoC_level[(rec.key(0))] = int(rec.level)

    waste_level = {}
    # Loop to display values of theta_variable
    print('\nDisplaying the waste level:')
    for rec in job.out_db["gammar"]:
        if rec.level != 0.0:  # Display non-null values
            print("gammar(" + rec.key(0) +  "): level=" + str(rec.level))
            waste_level[(rec.key(0))] = int(rec.level)

    time_level = {}
    # Loop to display values of theta_variable
    print('\nDisplaying the time of arrival:')
    for rec in job.out_db["rho"]:
        if rec.level != 0.0:  # Display non-null values
            print("rho(" + rec.key(0) + "): level=" + str(rec.level))
            time_level[(rec.key(0))] = int(rec.level)

    # Get the solution
    objective_function_value = job.out_db["tol_travel_cost"].find_record().level
    print("\nTotal ET operating cost = ", objective_function_value)


    "Get total travel distance"
    Total_travel_distance = job.out_db["pt"].find_record().level
    print("\ntotal travel distance = " , Total_travel_distance)

    Total_charging_time = job.out_db["cd"].find_record().level
    print("\nTotal charging time = " , Total_charging_time)

    Total_waste_amount = job.out_db["WA"].find_record().level
    print("\nTotal waste amount = " , Total_waste_amount)


    SolutionTime = round(elapsed_time - start_time)
    # b Print the solution time, Model status and Solve status
    print(f"Modelstatus: {job.out_db['ms'].find_record().value}")
    print(f"Solvestatus: {job.out_db['ss'].find_record().value}")
    print(f"Solution time: {SolutionTime} seconds")

    return (job, db, ws, feasible, arc_travel, arc_cost_dict, time_level, SoC_level, waste_level,
            visited_customer_nodes, SolutionTime, objective_function_value,
            Total_travel_distance, Total_charging_time, Total_waste_amount, dict_computation_time)