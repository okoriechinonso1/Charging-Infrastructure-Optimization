# Charging Infrastructure Optimization

Bi-level, multi-period optimization of **charging infrastructure expansion** for
electric trucks in **municipal solid waste collection**, with a real-world case study of city of
Tallahassee, Florida.

## Problem overview

**Upper level — strategic planning (dynamic program).**
Over a planning horizon of *P* periods, a municipal planner decides which candidate charging stations are in service
(*S^p*) and how many electric trucks are procured (*η^p*). A DP state is the pair `(S^p, η^p)`,
e.g. `({1, 3, 5}, 13)`. Stations are never closed and the fleet never shrinks
(`S^{p-1} ⊆ S^p`, `η^{p-1} ≤ η^p`). Each stage's cost is:

- capital cost λ of newly built stations,
- fixed service cost γ of every in-service station,
- fleet procurement cost *f^p* · (*η^p* − *η^{p-1}*), and
- operating cost μ(*S^p*, *η^p*) scaled by the days in the period, obtained from the lower level.

**Lower level — joint routing and charging optimization problem (MIP).**
An electric vehicle routing problem (EVRP) with vehicle capacity, driving range (or battery capacity), and minimum charging time limits.
Charging is modelled through **charging arcs** (`C+ → C-`), each with a charging rate α and cost
coefficient β1. Models are written in GAMS and solved with Gurobi.

Two DP solution methods are implemented:

1. **Brute-force DP** — enumerates the full state space (Section 4.2.1 of the manuscript).
2. **Labeling algorithm** — stage-wise dominance pruning that only evaluates non-dominated
   states (Sections 4.2.2–4.2.3).

## Solution pipeline

```
data generation ──► route generation (column generation) ──► charging-arc augmentation
                                                                     │
          DP (brute force / labeling) ◄── μ values ◄── charging-second optimization (GAMS)
```

1. **Data generation** — `realworld_data_generation.py` (Tallahassee) or
   `synthetic_data_generation.py` builds nodes, links, travel data, service times and waste demand.
2. **Route generation** — `CG_Algorithm_Main.py` builds a route pool (Phase 1) and selects routes
   for each fleet size with a MIP (Phase 2); results go to `CG_route_exports/`.
3. **Charging augmentation** — `augment_route_network.py` inserts the open charging arcs into each
   route; results go to `augmented_routes_for_charging/`.
4. **Charging-second optimization** — `charging_optimization_model.py` fixes each route and
   optimizes charging. `benchmark_main.py` / `Run_charging_optimization.py` loop this over DP
   states and write `Period_{P}_FleetSize_{F}.csv` summaries (Daily Cost, Period Cost).
5. **Dynamic program** — `DP_Algorithm_Benchmark.py` (brute force + labeling) or
   `DP_Algorithm_FullScale.py` (labeling) reads these costs and returns the optimal
   expansion path.

## Repository structure

| Path | Contents |
|---|---|
| `synthetic_main.py` | Joint routing-and-charging EVRP on a synthetic instance |
| `benchmark_main.py` | Charging-second optimization over all DP states for one benchmark period / fleet size |
| `manuscript_figures_plotting.py` | Figures for the manuscript |
| `synthetic_scripts/` | Synthetic data generator, joint EVRP model, CG and augmentation scripts |
| `synthetic_data/` | Synthetic instances `synthetic_data_1` … `synthetic_data_5` |
| `DP_benchmark_scripts/` | Benchmark DP (5 periods, 5 candidate stations, fleet sizes 5–10) |
| `DP_benchmark_data/` | Benchmark period data, route exports, brute-force results |
| `DP_FullScale_scripts/` | Full-scale DP (10 periods, 7 candidate stations, fleet sizes 5–20) |
| `DP_FullScale_data/` | Tallahassee network, per-period data, brute-force and labeling results |
| `Results/` | Manuscript figures |

Each period folder contains `{period}_nodes.csv`, `{period}_links.csv`,
`{period}_travel_data.csv`, `{period}_service_time.csv`, `{period}_waste_demand.csv`,
`charging_links.csv` (charging link IDs and C+/C- nodes) and `charging_arcs_params.csv`
(β1 and α per charging link).

## Requirements

- Python 3.10+
- [GAMS](https://www.gams.com/) with its Python API (`gams` package) and a **Gurobi** license
  available to GAMS
- Python packages:

  ```bash
  pip install numpy pandas matplotlib networkx scipy scikit-learn geopandas contextily
  ```

## Usage

The scripts have no command-line interface. Edit the configuration block at the top of each
script (period, fleet sizes, open charging links, parameters) and run the file from your IDE.

1. **Generate routes** for a period — set `subfolder_name` and `fleet_sizes` in
   `DP_FullScale_scripts/CG_Algorithm_Main.py` (or the benchmark / synthetic copy) and run it.
   These scripts use bare imports, so run them with their own folder as the working directory.
2. **Compute operating costs** — for the benchmark, set `PERIOD` and `FLEET_SIZE` in
   `benchmark_main.py` and run it; results are written to
   `DP_benchmark_data/brute_force_solution/`.
3. **Solve the DP** — edit the CONFIG constants (`STATION_COST_PARAMS`, `E_BAR`, `E0`,
   `NUM_PERIODS`, `ALLOWED_FLEET_SIZES`, …) in `DP_Algorithm_Benchmark.py` or
   `DP_Algorithm_FullScale.py`, then run the worked example at the bottom of the file.
   The labeling algorithm checkpoints after every stage, so interrupted runs can be resumed.

Map figures in `manuscript_figures_plotting.py` that use a Stadia Maps basemap read the API key
from the `STADIA_API_KEY` environment variable.

## Synthetic instances (Table 5)

All ten instances use the same 20-community network. The input files (complete_nodes,
complete_travel_data, service_time, waste_demand, charging_links, charging_arcs_params)
are identical in `synthetic_data/synthetic_data_1/` to `synthetic_data_4/`.
An instance differs only in which of the four charging stations are open.

For the proposed approach, the CG algorithm was run again for some instances to generate
and select slightly different routes before the charging optimization. Each instance's
routes are stored in the folder listed below, under `CG_route_exports_2/` and
`augmented_routes_for_charging_2/`. To reproduce an instance with the proposed approach,
set `folder_name` and `selected_charging_links` in the charging-second section of
`synthetic_main.py` as below and use the stored routes. Rerunning
`synthetic_scripts/CG_Algorithm_Main.py` may give different routes.

For the solver, any of the four folders can be used, since the inputs are identical.

| Instance | Open charging stations (`selected_charging_links`) | Route folder |
|---|---|---|
| 20-1  | [ … ] | synthetic_data_[ … ] |
| 20-2  | [ … ] | synthetic_data_[ … ] |
| 20-3  | [ … ] | synthetic_data_[ … ] |
| 20-4  | [ … ] | synthetic_data_[ … ] |
| 20-5  | [ … ] | synthetic_data_[ … ] |
| 20-6  | [ … ] | synthetic_data_[ … ] |
| 20-7  | [ … ] | synthetic_data_[ … ] |
| 20-8  | [ … ] | synthetic_data_[ … ] |
| 20-9  | [ … ] | synthetic_data_[ … ] |
| 20-10 | [ … ] | synthetic_data_[ … ] |

Instance 20-9 is also the instance used for Figure 6.

## Author

Chinonso Okorie — Florida A&M University (chinonso1.okorie@famu.edu)
Graduate Research Assistant & PhD Candidate in Industrial Engineering

## License

Released under the [MIT License](LICENSE).
