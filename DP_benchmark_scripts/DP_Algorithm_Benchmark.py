# -------------------------------------------------------------------------------
# Name:        Dynamic Program for Charging Infrastructure Expansion Problem
# Purpose:     Bi-level Optimization Model for Multi-Period Charging Infrastructure
#              Planning in Municipal Solid Waste Collection.
#
#              NO TERMINAL / NO CLI. Everything below is a plain Python function
#              you call directly -- either by running this file in an IDE
#              (VS Code / PyCharm / Spyder "Run" button) or from a Jupyter
#              notebook. Every function that reads or writes a file takes that
#              file's path as a plain string argument -- you never type a
#              command with flags anywhere.
#
#              1. BRUTE FORCE DP  (Section 4.2.1)
#                 run_bruteforce_export(out_path)
#                 run_bruteforce(mu_path, checkpoint_path, plot_path)
#
#              2. LABELING ALGORITHM  (Sections 4.2.2 - 4.2.3)
#                 run_labeling_export_stage(period, out_path, checkpoint_path)
#                 run_labeling_import_stage(period, mu_path, checkpoint_path)
#                 run_labeling_finalize(checkpoint_path, result_path, plot_path)
#
#              Since solving the lower-level VRP happens externally (in GAMS),
#              you'll call these functions across several separate "Run"
#              clicks, editing the file paths each time -- see the worked
#              example at the bottom of this file.
#
# Author:      Chinonso Okorie (chinonso1.okorie@famu.edu)
# Updated:     Removed argparse/CLI entirely -- direct function calls with
#              explicit file-path arguments only.
# -------------------------------------------------------------------------------

import csv
import math
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, FrozenSet, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# =============================================================================
# SHARED TYPES AND UTILITIES  (used by both algorithms)
# =============================================================================

StationConfig = FrozenSet[str]
FleetSize = int
DP_State = Tuple[StationConfig, FleetSize]

Candidate_station = str
Cost = float

Capital_CostMap = Dict[Tuple[int, Candidate_station], float]
Service_CostMap = Dict[Tuple[int, Candidate_station], float]
FleetCostMap = Dict[int, float]


def powerset(candidate_set: Iterable[Candidate_station]) -> Iterable[StationConfig]:
    """Generate all subsets of candidate_set as frozensets."""
    x = sorted(candidate_set)
    for r in range(len(x) + 1):
        for comb in combinations(x, r):
            yield frozenset(comb)


def feasible_station_configs(
        E_bar: Set[Candidate_station],
        E0: Set[Candidate_station]
) -> List[StationConfig]:
    """All subsets of E_bar (candidate stations) that contain E0 (base in-service
    stations), ordered by |S| ascending then lexicographic. This is the FULL
    feasible state-space definition used only by brute force (and, via the
    Pi[0]={initial} unification, coincides with the labeling algorithm's stage-1
    candidate set -- see stage_generate_candidates)."""
    configs = [S for S in powerset(E_bar) if frozenset(E0).issubset(S)]
    configs.sort(key=lambda S: (len(S), tuple(sorted(S))))
    return configs


def feasible_fleet_sizes(allowed_fleet_sizes: List[FleetSize]) -> List[FleetSize]:
    return sorted(allowed_fleet_sizes)


def feasible_states(
        E_bar: Set[Candidate_station],
        E0: Set[Candidate_station],
        allowed_fleet_sizes: List[FleetSize]
) -> List[DP_State]:
    """Cartesian product of feasible station configs x allowed fleet sizes.
    This is the FULL state space per stage -- used by brute force only."""
    configs = feasible_station_configs(E_bar, E0)
    fleet_vals = feasible_fleet_sizes(allowed_fleet_sizes)
    states = [(S, eta) for S in configs for eta in fleet_vals]
    states.sort(key=lambda st: (len(st[0]), tuple(sorted(st[0])), st[1]))
    return states


def state_to_key(state: DP_State) -> str:
    """Human-readable key: '({1,3,5}, 13)'"""
    S, eta = state
    stations = "{" + ", ".join(sorted(S)) + "}"
    return "(" + stations + ", " + str(eta) + ")"


def transition_feasible(prev_state: DP_State, curr_state: DP_State) -> bool:
    """Feasible iff S^{p-1} subseteq S^p and eta^{p-1} <= eta^p."""
    S_prev, eta_prev = prev_state
    S_curr, eta_curr = curr_state
    return S_prev.issubset(S_curr) and eta_prev <= eta_curr


def stage_transition_cost(
        period: int,
        prev_state: DP_State,
        curr_state: DP_State,
        lam: Capital_CostMap,
        gam: Service_CostMap,
        fleet_cost: FleetCostMap,
        mu_value: Cost,
) -> Cost:
    """
    One-stage cost, Eqs. (45)/(46)/(48):
        Sum_{e in S^p \\ S^{p-1}} lambda^p_e   (capital: newly built stations)
      + Sum_{e in S^p} gamma^p_e               (service: all in-service stations)
      + f^p * (eta^p - eta^{p-1})              (fleet procurement)
      + mu_value                                (D^p * mu^p(S^p, eta^p), pre-scaled)
    """
    S_prev, eta_prev = prev_state
    S_curr, eta_curr = curr_state
    added = S_curr.difference(S_prev)
    capital_cost = sum(lam[(period, e)] for e in added)
    service_cost = sum(gam[(period, e)] for e in S_curr)
    fleet_procure = fleet_cost[period] * (eta_curr - eta_prev)
    return capital_cost + service_cost + fleet_procure + mu_value


# ---------- DOMINANCE (Proposition 2) -- shared by both algorithms ----------
def dominates(
        state1: DP_State,
        state2: DP_State,
        label_state1: Cost,
        label_state2: Cost,
        lam: Capital_CostMap,
        gam: Service_CostMap,
        fleet_cost: FleetCostMap,
        period: int,
        P: int
) -> bool:
    """
    True if state1 dominates state2 (Proposition 2, any one condition).
    Precondition: period < P (dominance is undefined at the final stage).
    """
    if period >= P:
        raise ValueError(
            f"dominates() requires period < P (got period={period}, P={P})."
        )

    S1, eta1 = state1
    S2, eta2 = state2

    # Shared fleet catch-up term: f^{p+1} * max{0, eta2 - eta1}
    fleet_catchup = fleet_cost[period + 1] * max(0, eta2 - eta1)

    # Condition 1: E1 = E2, eta1 >= eta2, V1 <= V2
    if S1 == S2 and eta1 >= eta2 and label_state1 <= label_state2:
        return True

    # Condition 2: E1 subseteq E2
    #   V1 + Sum_{e in E2\E1} lambda_e^{p+1} + fleet_catchup < V2
    if S1.issubset(S2):
        missing_stations = S2.difference(S1)
        capital_cost_defer = sum(lam[(period + 1, e)] for e in missing_stations)
        if label_state1 + capital_cost_defer + fleet_catchup < label_state2:
            return True

    # Condition 3: E2 subseteq E1
    #   V1 + Sum_{t=p+1}^{P} Sum_{e in E1\E2} gamma_e^t + fleet_catchup < V2
    if S2.issubset(S1):
        extra_stations = S1.difference(S2)
        future_service_cost = sum(
            gam[(t, e)] for t in range(period + 1, P + 1) for e in extra_stations
        )
        if label_state1 + future_service_cost + fleet_catchup < label_state2:
            return True

    # Condition 4: neither E1 subseteq E2 nor E2 subseteq E1; U = E1 cap E2
    #   V1 + Sum_{t=p+1}^P Sum_{e in E1\U} gamma_e^t
    #      + Sum_{e in E2\U} lambda_e^{p+1} + fleet_catchup < V2
    if not S1.issubset(S2) and not S2.issubset(S1):
        U = S1.intersection(S2)
        A = S1.difference(U)  # unique to S1
        B = S2.difference(U)  # unique to S2
        future_service_A = sum(
            gam[(t, e)] for t in range(period + 1, P + 1) for e in A
        )
        capital_cost_B = sum(lam[(period + 1, e)] for e in B)
        if label_state1 + future_service_A + capital_cost_B + fleet_catchup < label_state2:
            return True

    return False


def filter_dominated_states(
        states_dict: Dict[DP_State, Cost],
        lam: Capital_CostMap,
        gam: Service_CostMap,
        fleet_cost: FleetCostMap,
        period: int,
        P: int
) -> Dict[DP_State, Cost]:
    """Return only the non-dominated states of states_dict, per Proposition 2."""
    items = list(states_dict.items())
    non_dominated = {}
    for state1, label1 in items:
        dominated = False
        for state2, label2 in items:
            if state1 == state2:
                continue
            if dominates(state2, state1, label2, label1, lam, gam, fleet_cost, period, P):
                dominated = True
                break
        if not dominated:
            non_dominated[state1] = label1
    return non_dominated


@dataclass
class DPResult:
    best_cost: Cost
    best_final_state: DP_State
    psi: Dict[int, Dict[DP_State, Cost]]           # value function per stage (pre-pruning where applicable)
    pred: Dict[int, Dict[DP_State, Optional[DP_State]]]
    path: List[Tuple[int, DP_State]]
    algorithm_type: str                              # "brute_force" or "labeling"
    state_reduction: Dict[int, Tuple[int, int]]       # period -> (generated, kept)
    candidates: Optional[Dict[int, Dict[DP_State, Cost]]] = None  # labeling only
    Pi: Optional[Dict[int, Dict[DP_State, Cost]]] = None          # labeling only


def print_optimal_path(res: DPResult) -> None:
    print("\n" + "=" * 70)
    print(f"OPTIMAL EXPANSION PATH ({res.algorithm_type.upper()})")
    print("=" * 70)
    for period, state in res.path:
        S, eta = state
        psi_val = res.psi.get(period, {}).get(state, 0.0)
        print(f"  p={period}: stations={set(S)}, fleet_size={eta}, psi^{period} = ${psi_val:,.2f}")
    print(f"\n  TOTAL MINIMUM SYSTEM COST = ${res.best_cost:,.2f}")
    print("=" * 70)


def print_state_reduction_summary(res: DPResult) -> None:
    if res.algorithm_type != "labeling":
        return
    print("\n" + "=" * 70)
    print("STATE SPACE REDUCTION SUMMARY (Labeling Algorithm)")
    print("=" * 70)
    print(f"{'Period':<10}{'Generated':<15}{'Kept':<15}{'Reduction %':<15}")
    print("-" * 70)
    tot_g, tot_k = 0, 0
    for period in sorted(res.state_reduction):
        g, k = res.state_reduction[period]
        pct = ((g - k) / g * 100) if g else 0.0
        print(f"{period:<10}{g:<15}{k:<15}{pct:>13.1f}%")
        tot_g += g
        tot_k += k
    overall = ((tot_g - tot_k) / tot_g * 100) if tot_g else 0.0
    print("-" * 70)
    print(f"{'TOTAL':<10}{tot_g:<15}{tot_k:<15}{overall:>13.1f}%")
    print("=" * 70)


# def print_cost_breakdown(
#         res: DPResult,
#         lam: Capital_CostMap,
#         gam: Service_CostMap,
#         fleet_cost: FleetCostMap,
# ) -> None:
#
#     print("\n" + "=" * 100)
#     print(f"COST BREAKDOWN BY PERIOD ({res.algorithm_type.upper()})")
#     print("=" * 100)
#     print(f"{'Period':<8}{'New Stat.':<11}{'In-Serv.':<10}{'New ETs':<9}"
#           f"{'Infrastructure':>18}{'Lower-Level (mu)':>20}{'Total System Cost':>20}")
#     print("-" * 100)
#
#     total_infra = 0.0
#     total_mu = 0.0
#
#     for idx in range(1, len(res.path)):
#         period, curr_state = res.path[idx]
#         _, prev_state = res.path[idx - 1]
#         S_curr, eta_curr = curr_state
#         S_prev, eta_prev = prev_state
#
#         added = S_curr.difference(S_prev)
#         n_new_stations = len(added)
#         n_in_service = len(S_curr)
#         n_new_ets = eta_curr - eta_prev
#
#         capital_cost = sum(lam[(period, e)] for e in added)
#         service_cost = sum(gam[(period, e)] for e in S_curr)
#         fleet_procure_cost = fleet_cost[period] * n_new_ets
#         infrastructure_cost = capital_cost + service_cost + fleet_procure_cost
#
#         stage_total = res.psi[period][curr_state] - res.psi[period - 1][prev_state]
#         lower_level_cost = stage_total - infrastructure_cost
#
#         total_infra += infrastructure_cost
#         total_mu += lower_level_cost
#
#         print(f"{period:<8}{n_new_stations:<11}{n_in_service:<10}{n_new_ets:<9}"
#               f"${infrastructure_cost:>16,.2f}  ${lower_level_cost:>17,.2f}  "
#               f"${stage_total:>17,.2f}")
#
#     print("-" * 100)
#     grand_total = total_infra + total_mu
#     print(f"{'TOTAL':<8}{'':<11}{'':<10}{'':<9}"
#           f"${total_infra:>16,.2f}  ${total_mu:>17,.2f}  ${grand_total:>17,.2f}")
#     print("=" * 100)
#     print(f"  Sanity check: infrastructure + lower-level = ${grand_total:,.2f} "
#           f"(should equal TOTAL MINIMUM SYSTEM COST = ${res.best_cost:,.2f})")
#     print("=" * 100)


def print_cost_breakdown(
        res: DPResult,
        lam: Capital_CostMap,
        gam: Service_CostMap,
        fleet_cost: FleetCostMap,
) -> None:
    """
    For each period along the optimal path, prints:
      - New stations built and total stations in service that period
      - New ETs purchased that period
      - Capital cost (newly built stations only)
      - Fixed service cost (all in-service stations)
      - Fleet procurement cost
      - Infrastructure cost = capital + service + fleet procurement
      - Lower-level ET operating cost (D^p * mu^p), backed out as
        (total stage cost - infrastructure cost) -- exact by construction.
      - Total system cost for that period (infrastructure + lower-level)

    lam/gam/fleet_cost must be the SAME cost maps used to produce res.
    """
    print("\n" + "=" * 140)
    print(f"COST BREAKDOWN BY PERIOD ({res.algorithm_type.upper()})")
    print("=" * 140)
    print(f"{'Period':<8}{'New Stat.':<11}{'In-Serv.':<10}{'New ETs':<9}"
          f"{'Capital Cost':>16}{'Service Cost':>16}{'Fleet Procure':>16}"
          f"{'Infrastructure':>18}{'Lower-Level (mu)':>20}{'Total System Cost':>20}")
    print("-" * 140)

    total_capital = 0.0
    total_service = 0.0
    total_fleet = 0.0
    total_infra = 0.0
    total_mu = 0.0

    for idx in range(1, len(res.path)):
        period, curr_state = res.path[idx]
        _, prev_state = res.path[idx - 1]
        S_curr, eta_curr = curr_state
        S_prev, eta_prev = prev_state

        added = S_curr.difference(S_prev)
        n_new_stations = len(added)
        n_in_service = len(S_curr)
        n_new_ets = eta_curr - eta_prev

        capital_cost = sum(lam[(period, e)] for e in added)
        service_cost = sum(gam[(period, e)] for e in S_curr)
        fleet_procure_cost = fleet_cost[period] * n_new_ets
        infrastructure_cost = capital_cost + service_cost + fleet_procure_cost

        stage_total = res.psi[period][curr_state] - res.psi[period - 1][prev_state]
        lower_level_cost = stage_total - infrastructure_cost

        total_capital += capital_cost
        total_service += service_cost
        total_fleet += fleet_procure_cost
        total_infra += infrastructure_cost
        total_mu += lower_level_cost

        print(f"{period:<8}{n_new_stations:<11}{n_in_service:<10}{n_new_ets:<9}"
              f"${capital_cost:>14,.2f}  ${service_cost:>14,.2f}  ${fleet_procure_cost:>14,.2f}  "
              f"${infrastructure_cost:>16,.2f}  ${lower_level_cost:>17,.2f}  "
              f"${stage_total:>17,.2f}")

    print("-" * 140)
    grand_total = total_infra + total_mu
    print(f"{'TOTAL':<8}{'':<11}{'':<10}{'':<9}"
          f"${total_capital:>14,.2f}  ${total_service:>14,.2f}  ${total_fleet:>14,.2f}  "
          f"${total_infra:>16,.2f}  ${total_mu:>17,.2f}  ${grand_total:>17,.2f}")
    print("=" * 140)
    print(f"  Sanity check: capital + service + fleet = ${total_capital + total_service + total_fleet:,.2f} "
          f"(should equal TOTAL Infrastructure = ${total_infra:,.2f})")
    print(f"  Sanity check: infrastructure + lower-level = ${grand_total:,.2f} "
          f"(should equal TOTAL MINIMUM SYSTEM COST = ${res.best_cost:,.2f})")
    print("=" * 140)



def print_non_dominated_states(res: DPResult) -> None:
    """
    Prints every non-dominated state kept at each stage (Pi[period]), sorted
    by cost ascending. Intended to be called right after
    driver.verify_dominance_pruning(). Only meaningful for labeling results
    (brute force does not perform dominance pruning).
    """
    if res.algorithm_type != "labeling" or res.Pi is None:
        print("\n(Non-dominated state listing is only available for labeling results.)")
        return
    print("\n" + "=" * 70)
    print("NON-DOMINATED STATES PER STAGE")
    print("=" * 70)
    for period in sorted(res.Pi.keys()):
        states = res.Pi[period]
        note = "no pruning at final stage" if period == max(res.Pi.keys()) else "after dominance pruning"
        print(f"\nPeriod p={period}: {len(states)} non-dominated state(s) ({note})")
        for state, cost in sorted(states.items(), key=lambda kv: kv[1]):
            print(f"    {state_to_key(state):<45} psi^{period} = ${cost:,.2f}")
    print("=" * 70)


def export_non_dominated_states(res: DPResult, out_path: str) -> None:
    """
    Write the non-dominated states at each period (Pi[period]) to a CSV file,
    one column per period, one state key per cell. Periods with fewer
    surviving states than the max leave the remaining cells blank.
    Period 0 (the trivial initial state) is excluded.

    For brute-force results (Pi is None), the full reachable state set per
    period (psi) is written instead.
    """
    source = res.Pi if res.Pi is not None else res.psi
    periods = [p for p in sorted(source.keys()) if p != 0]

    columns = []
    for period in periods:
        states_dict = source[period]
        cells = [state_to_key(state)
                 for state, _ in sorted(states_dict.items(), key=lambda kv: kv[1])]
        columns.append((f"period {period}", cells))

    max_rows = max((len(cells) for _, cells in columns), default=0)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([header for header, _ in columns])
        for i in range(max_rows):
            w.writerow([cells[i] if i < len(cells) else "" for _, cells in columns])

    total = sum(len(cells) for _, cells in columns)
    print(f"Non-dominated states ({total} total across {len(periods)} periods) exported to {out_path}")

def export_dp_values_grid(
        path: str,
        states_asc: List[DP_State],
        P: int,
        psi: Dict[int, Dict[DP_State, Cost]],
        Pi: Optional[Dict[int, Dict[DP_State, Cost]]] = None,
) -> None:
    """
    Writes the computed DP value of every state, per period, to a CSV grid --
    header row + one state-label column, in the same row/column format as
    bf_export_state_grid()/bf_load_mu_grid() (state label in column A, then
    one column per period), so it's easy to place side by side with the mu
    grid you filled in.

    Brute force (Pi=None): every state reachable in period p, i.e. present
    in psi[p], has its psi^p(S^p, eta^p) value exported. States not
    reachable in period p (infeasible mu, or unreachable from any
    period-(p-1) predecessor) are left blank.

    Labeling (Pi provided): a cell is populated ONLY if the state survived
    dominance pruning at that period, i.e. is a member of Pi[p]. This
    guarantees every exported value is the label value V^p(E^p, eta^p) of a
    non-dominated state, never a candidate that was later dominated and
    discarded. At the final period P, Pi[P] equals the full candidate set
    (Proposition 2 dominance is undefined at p=P), so that column shows
    every state reachable in period P.
    """
    source = Pi if Pi is not None else psi
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state"] + [f"Period {p} Value" for p in range(1, P + 1)])
        for state in reversed(states_asc):
            row = [state_to_key(state)]
            for p in range(1, P + 1):
                val = source.get(p, {}).get(state, None)
                row.append("" if val is None else f"{val:.6f}")
            w.writerow(row)


def compare_results(res_bf: DPResult, res_lbl: DPResult) -> None:
    """
    Compare a finalized brute-force DPResult and a finalized labeling DPResult.
    Call this yourself after both run_bruteforce(...) and
    run_labeling_finalize(...) have returned their results (or after loading
    both .pkl files back in) -- it is not called automatically.
    """
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"Brute force best cost : ${res_bf.best_cost:,.2f}")
    print(f"Labeling best cost    : ${res_lbl.best_cost:,.2f}")
    same = abs(res_bf.best_cost - res_lbl.best_cost) < 1e-6
    print(f"Same optimal cost     : {same}")

    periods = sorted(set(res_bf.state_reduction) & set(res_lbl.state_reduction))

    print("\n" + "-" * 70)
    print("STATES KEPT PER PERIOD (Brute Force vs Labeling)")
    print("(BF: states reachable with a real mu solve -- no pruning exists.")
    print(" Labeling: states surviving Proposition 2 dominance pruning.)")
    print("-" * 70)
    print(f"{'Period':<10}{'BF Kept':<14}{'Lbl Kept':<14}"
          f"{'Reduction (#)':<16}{'Reduction (%)':<15}")
    print("-" * 70)

    bf_vals, lbl_vals, diff_vals, pct_vals = [], [], [], []
    for p in periods:
        bf_kept = res_bf.state_reduction[p][1]
        lbl_kept = res_lbl.state_reduction[p][1]
        diff = bf_kept - lbl_kept
        pct = (diff / bf_kept * 100) if bf_kept else 0.0
        print(f"{p:<10}{bf_kept:<14}{lbl_kept:<14}{diff:<16}{pct:>13.1f}%")
        bf_vals.append(bf_kept)
        lbl_vals.append(lbl_kept)
        diff_vals.append(diff)
        pct_vals.append(pct)

    n = len(periods)
    avg_bf = sum(bf_vals) / n if n else 0.0
    avg_lbl = sum(lbl_vals) / n if n else 0.0
    avg_diff = sum(diff_vals) / n if n else 0.0
    avg_pct = sum(pct_vals) / n if n else 0.0

    print("-" * 70)
    print(f"{'AVERAGE':<10}{avg_bf:<14.1f}{avg_lbl:<14.1f}{avg_diff:<16.1f}{avg_pct:>13.1f}%")

    print(f"\nAverage states kept per period (brute force)           : {avg_bf:.1f}")
    print(f"Average states kept per period (labeling, post-pruning): {avg_lbl:.1f}")
    print(f"Average reduction in states carried forward             : {avg_pct:.1f}%")
    print("=" * 70)


from typing import Dict, List, Tuple


def count_paths_by_stage(res: "DPResult") -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Returns (paths_evaluated, paths_surviving), each a dict mapping
    period -> {state: path_count}, counting distinct feasible
    S^0 -> ... -> S^p sequences using only prior-stage SURVIVORS as valid
    predecessors at every step (never a dominated/discarded state).
    """
    periods = sorted(res.psi.keys())
    p0 = periods[0]
    is_labeling = res.algorithm_type == "labeling"
    P = periods[-1]

    paths_evaluated: Dict[int, Dict] = {p0: {s: 1 for s in res.psi[p0].keys()}}
    paths_surviving: Dict[int, Dict] = {p0: dict(paths_evaluated[p0])}

    for period in periods[1:]:
        prev_surv = paths_surviving[period - 1]

        pc_eval = {}
        for s in res.psi[period].keys():
            total = 0
            for s_prev, cnt in prev_surv.items():
                if transition_feasible(s_prev, s):
                    total += cnt
            pc_eval[s] = total
        paths_evaluated[period] = pc_eval

        if is_labeling and res.Pi is not None and period in res.Pi and period != P:
            paths_surviving[period] = {s: pc_eval[s] for s in res.Pi[period].keys()}
        else:
            # brute force (never prunes) or the final stage (Proposition 2
            # pruning is skipped there) -- evaluated == surviving.
            paths_surviving[period] = pc_eval

    return paths_evaluated, paths_surviving


def print_path_count_comparison(res_bf: "DPResult", res_lbl: "DPResult") -> None:
    bf_eval, bf_surv = count_paths_by_stage(res_bf)
    lbl_eval, lbl_surv = count_paths_by_stage(res_lbl)

    periods = sorted((set(bf_surv) & set(lbl_surv)) - {0})

    print("\n" + "=" * 100)
    print("PATH COUNT COMPARISON (Initial Stage -> Each Stage)")
    print("Lbl 'Evaluated' = paths reaching every candidate computed pre-pruning (Generated).")
    print("Lbl 'Surviving'  = subset of those reaching non-dominated states (Kept); this is")
    print("                   what actually seeds the next stage and can still reach period P.")
    print("=" * 100)
    print(f"{'Period':<8}{'BF Paths':<16}{'Lbl Evaluated':<18}{'Lbl Surviving':<18}{'Reduction % (BF vs Lbl Surviving)':<10}")
    print("-" * 100)
    for p in periods:
        bf_total = sum(bf_surv[p].values())
        lbl_total_eval = sum(lbl_eval[p].values())
        lbl_total_surv = sum(lbl_surv[p].values())
        pct = ((bf_total - lbl_total_surv) / bf_total * 100) if bf_total else 0.0
        print(f"{p:<8}{bf_total:<16,}{lbl_total_eval:<18,}{lbl_total_surv:<18,}{pct:>10.4f}%")

    P = max(periods)
    bf_final = sum(bf_surv[P].values())
    lbl_final = sum(lbl_surv[P].values())  # == sum(lbl_eval[P].values()); no pruning at final stage
    pct_final = ((bf_final - lbl_final) / bf_final * 100) if bf_final else 0.0

    print("-" * 100)
    print(f"\nTotal complete initial -> final paths, brute force : {bf_final:,}")
    print(f"Total complete initial -> final paths, labeling    : {lbl_final:,}")
    print(f"Reduction in total path count                      : {pct_final:.4f}%")
    print("=" * 100)

def plot_dp_network(res: DPResult, out_path: str) -> None:
    """
    Visualize the DP state network and highlight the optimal expansion path.
    Draws exactly the states each algorithm actually carried: the full
    feasible lattice per stage for brute force, or only the surviving
    (non-dominated) states per stage for labeling. out_path is the full file
    path (including filename) where the .svg is written.
    """
    is_labeling = res.algorithm_type == "labeling"
    per_period_states: Dict[int, List[DP_State]] = {}
    for period, table in res.psi.items():
        if is_labeling and res.Pi is not None and period in res.Pi:
            per_period_states[period] = list(res.Pi[period].keys())
        else:
            per_period_states[period] = list(table.keys())

    all_states_seen: List[DP_State] = []
    for states in per_period_states.values():
        for s in states:
            if s not in all_states_seen:
                all_states_seen.append(s)
    all_states_seen.sort(key=lambda st: (len(st[0]), tuple(sorted(st[0])), st[1]))
    state_row = {s: i for i, s in enumerate(all_states_seen)}

    path_color = "purple" if is_labeling else "orange"
    G = nx.DiGraph()
    pos: Dict = {}
    labels: Dict = {}
    periods_sorted = sorted(per_period_states)
    for period in periods_sorted:
        for state in per_period_states[period]:
            node = (period, state)
            G.add_node(node)
            pos[node] = (period, state_row.get(state, 0))
            labels[node] = state_to_key(state)

    for i in range(1, len(periods_sorted)):
        p_prev, p_curr = periods_sorted[i - 1], periods_sorted[i]
        for prev_state in per_period_states[p_prev]:
            for curr_state in per_period_states[p_curr]:
                if transition_feasible(prev_state, curr_state):
                    G.add_edge((p_prev, prev_state), (p_curr, curr_state))

    path_nodes = [(p, s) for p, s in res.path]
    path_edges = [
        ((p1, s1), (p2, s2))
        for (p1, s1), (p2, s2) in zip(res.path[:-1], res.path[1:])
    ]

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()),
                            node_color="white", edgecolors="black", node_size=150, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()),
                            edge_color="lightgray", arrows=False, width=0.6, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=path_nodes,
                            node_color=path_color, edgecolors="black", node_size=250)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                            edge_color=path_color, width=2.5, arrows=True, arrowstyle="-|>")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    ax = plt.gca()
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax.set_xticks(periods_sorted)
    ax.set_xticklabels([str(p) for p in periods_sorted])

    note = "non-dominated states only" if is_labeling else "full feasible state space"
    plt.xlabel("DP Stage (Planning period)")
    plt.ylabel("DP State (station configuration, fleet size)")
    plt.title(f"DP State Space Network - {res.algorithm_type.upper()} ({note}) - "
              f"Optimal Charging Infrastructure Planning Path")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"Network plot saved to {out_path}")


"For intermediate case analysis"
def plot_dp_network_3d(
        res,
        allowed_fleet_sizes,
        E_bar,                          # full candidate station set
        E0,                             # base in-service stations
        out_path: str,
        configs_universe: Optional[Set[StationConfig]] = None,
        period_floor_source: Optional[Dict[int, List[DP_State]]] = None,   # <-- NEW
) -> None:

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import offset_copy

    is_labeling = res.algorithm_type == "labeling"

    # ---- per-period states to draw: EVERY state evaluated in that stage
    # (res.psi[p] = all candidates whose lower-level problem was solved,
    # before dominance pruning), not only the non-dominated survivors. ----
    per_period_states = {period: list(table.keys()) for period, table in res.psi.items()}

    # ---- survivors of dominance pruning (res.Pi[p]). Only these are expanded
    # into stage p+1, so they are the only valid tails of transition edges. ----
    survivors = {}
    for period in per_period_states:
        if is_labeling and res.Pi is not None and period in res.Pi:
            survivors[period] = set(res.Pi[period].keys())
        else:
            survivors[period] = set(per_period_states[period])

    PRUNED_ALPHA = 1.0   # opacity of evaluated-but-dominated nodes; e.g. 0.35 to fade them

    periods_sorted = sorted(per_period_states)
    P = max(periods_sorted)
    initial_period = min(periods_sorted)

    fleet_sorted = sorted(allowed_fleet_sizes)
    NODE_EI_OFFSET = 0.3  # shifts all node fleet-size positions away from the
                            # z-axis origin, so the smallest fleet size (ei=0)
                            # no longer sits directly on the axis line
    eta_index = {eta: i + NODE_EI_OFFSET for i, eta in enumerate(fleet_sorted)}
    n_e = len(fleet_sorted)

    if configs_universe is not None:
        sorted_configs = sorted(
            configs_universe,
            key=lambda S: (len(S), tuple(sorted(S, key=lambda x: int(x))))
        )
    else:
        sorted_configs = sorted(
            feasible_station_configs(E_bar, E0),
            key=lambda S: (len(S), tuple(sorted(S, key=lambda x: int(x))))
        )
    config_level = {S: i for i, S in enumerate(sorted_configs)}
    level_min = 0
    level_max = max(len(sorted_configs) - 1, 0)


    BASE_FLEET_COLOR_BY_VALUE = {
        5: '#FFA500',  # orange
        6: '#228B22',  # green
        7: '#1E90FF',  # blue
        8: '#DC143C',  # red
        9: '#8B4513',  # brown
        10: '#006400',  # dark green
        11: '#9467bd',  # purple
        12: '#ff9896',  # light red
        13: '#f7b6d2',  # light pink
        14: '#c5b0d5',  # light purple
        15: '#ffbb78',  # light orange
        16: '#c49c94',  # light brown
        17: '#e377c2',  # pink
        18: '#aec7e8',  # light blue
        19: '#7f7f7f',  # gray
        20: '#bcbd22',  # olive
        21: '#dbdb8d',  # light olive
        22: '#17becf',  # cyan
        23: '#9edae5',  # light cyan
        24: '#393b79',  # dark indigo
        25: '#8B4513',  # saddle brown (from your original)
    }
    EXTRA_PALETTE = ['#008B8B', '#556B2F', '#4682B4',
                      '#A0522D', '#2F4F4F', '#B8860B', '#800000', ]
    fleet_color_by_value = {}
    extra_i = 0
    for eta in fleet_sorted:
        if eta in BASE_FLEET_COLOR_BY_VALUE:
            fleet_color_by_value[eta] = BASE_FLEET_COLOR_BY_VALUE[eta]
        else:
            fleet_color_by_value[eta] = EXTRA_PALETTE[extra_i % len(EXTRA_PALETTE)]
            extra_i += 1

    PATH_COLOR = '#5B0EA6'

    angle = np.radians(35)
    v_p = np.array([1.0, 0.0])
    v_s = np.array([0.0, 1.0])
    v_e = np.array([np.cos(angle), np.sin(angle)])

    NODE_R = 0.35 #0.45 #

    # Same fixed spacing as the sample illustration -- sp_p, sp_s, sp_e are
    # plain constants, not dynamically adjusted.
    sp_p = 10.0 #10.0 #
    sp_s = 1.0 #1.0 #0.6
    sp_e =  1.4 #1.4 #

    JITTER_R = 0.11 * 0.4

    # ---- small deliberate downward offset for stage-1 nodes only (cosmetic) ----
    STAGE1_Y_OFFSET = -0.00
    stage1_period = periods_sorted[1] if len(periods_sorted) > 1 else None

    period_floor = {}
    for period in periods_sorted:
        if period_floor_source is not None and period in period_floor_source:
            source_states = period_floor_source[period]
        else:
            source_states = per_period_states[period]
        levels_here = ([config_level[S] for (S, _eta) in source_states]
                       + [config_level[S] for (S, _eta) in per_period_states[period]])
        period_floor[period] = min(levels_here) if levels_here else level_min

    # False -> every period uses the same vertical scale (level 0 at the x-axis),
    # so a station configuration sits at the same height in every period, as in
    # the original brute-force/labeling figures. True -> re-base each period to
    # its lowest drawn configuration (the old period_floor_source behaviour).
    REBASE_FLOORS = False
    if not REBASE_FLOORS:
        period_floor = {period: level_min for period in periods_sorted}

    def proj(p, level, ei):
        floor = period_floor.get(p, level_min)
        base = p * sp_p * v_p + (level - floor) * sp_s * v_s + ei * sp_e * v_e
        if stage1_period is not None and p == stage1_period:
            base = base + STAGE1_Y_OFFSET * v_s
        return base

    # ---- group real states by (level, eta_idx). With global per-config
    # levels this slot is now unique per state, so jitter should not
    # normally trigger -- kept only as a defensive fallback. ----
    def slot_positions(period):
        groups = {}
        for state in per_period_states[period]:
            S, eta = state
            slot = (config_level[S], eta_index[eta])
            groups.setdefault(slot, []).append(state)

        out = {}
        for slot, states in groups.items():
            base_xy = proj(period, *slot)
            n = len(states)
            entries = []
            if n == 1:
                entries.append((states[0], base_xy))
            else:
                for k, state in enumerate(states):
                    theta = 2 * np.pi * k / n
                    offset = JITTER_R * np.array([np.cos(theta), np.sin(theta)])
                    entries.append((state, base_xy + offset))
            out[slot] = entries
        return out

    period_slot_positions = {period: slot_positions(period) for period in periods_sorted}

    EDGE_LW = 0.9
    EDGE_COL = '#aaaaaa'

    fig, ax = plt.subplots(figsize=(35, 24), facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')

    O = proj(initial_period, level_min, 0) + np.array([-0.2, -0.4]) - np.array([sp_p, 0.0])

    # exact real state -> rendered xy lookup, for edge drawing and path highlighting
    state_xy = {}
    for period in periods_sorted:
        for slot, entries in period_slot_positions[period].items():
            for state, xy in entries:
                state_xy[(period, state)] = xy

    # ---- background transition edges (real feasibility, using actual
    # rendered positions) ----
    fan_segs = []
    for i in range(len(periods_sorted) - 1):
        p0, p1 = periods_sorted[i], periods_sorted[i + 1]
        for prev_state in survivors[p0]:          # only survivors are expanded
            for curr_state in per_period_states[p1]:
                if not transition_feasible(prev_state, curr_state):
                    continue
                fan_segs.append([state_xy[(p0, prev_state)], state_xy[(p1, curr_state)]])

    ax.add_collection(LineCollection(fan_segs, colors=EDGE_COL,
                                      linewidths=EDGE_LW, alpha=0.45, zorder=1))

    # ---- optimal path: match by REAL STATE identity, not projected slot ----
    opt_real_states = set(res.path)  # {(period, state), ...}

    # ---- nodes per period ----
    for period in reversed(periods_sorted):
        zb = (P - period + 1) * 30

        if period != initial_period:
            c0 = proj(period, level_min, 0)
            c1 = proj(period, level_min, n_e - 1)
            c2 = proj(period, level_max, n_e - 1)
            c3 = proj(period, level_max, 0)
            poly = mpatches.Polygon([c0, c1, c2, c3], closed=True,
                                     facecolor='#c8c4dc', edgecolor='none',
                                     alpha=0.12, zorder=zb)
            ax.add_patch(poly)

        for slot, entries in period_slot_positions[period].items():
            for state, xy in entries:
                is_path_node = (period, state) in opt_real_states
                S, eta = state
                if is_path_node:
                    fill = PATH_COLOR
                    edge_color = PATH_COLOR
                    edge_lw = 1.0
                else:
                    fill = fleet_color_by_value[eta]
                    edge_color = '#222'
                    edge_lw = 0.8
                circ = mpatches.Circle(xy, radius=NODE_R, facecolor=fill,
                                        edgecolor=edge_color, linewidth=edge_lw,
                                        alpha=1.0 if state in survivors[period] else PRUNED_ALPHA,
                                        zorder=950 if is_path_node else zb + 3)
                ax.add_patch(circ)

    # ---- optimal path edges (via real rendered positions) ----
    path_xy = [state_xy[(p, s)] for p, s in res.path]
    for i in range(len(path_xy) - 1):
        a, b = path_xy[i], path_xy[i + 1]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=PATH_COLOR, lw=4.5,
                alpha=0.5, zorder=895, solid_capstyle='round')

    aw = dict(arrowstyle='->', color='#000', lw=2.2, mutation_scale=18)

    # ---- x-axis (DP stage) ----
    tip_p = np.array([proj(P, level_min, 0)[0] + sp_p * 0.8, O[1]])
    ax.annotate('', xy=tip_p, xytext=O, arrowprops=aw, zorder=999,
                annotation_clip=False)
    mid_p = np.array([(O[0] + tip_p[0]) / 2, O[1]])

    # ---- z-axis (fleet size) ----
    tip_e = O + (n_e - 1 + 0.8) * sp_e * v_e
    ax.annotate('', xy=tip_e, xytext=O, arrowprops=aw, zorder=999,
                annotation_clip=False)
    rot_e = np.degrees(np.arctan2(v_e[1], v_e[0]))
    tip_e_label = O + (n_e - 1 + 1.1) * sp_e * v_e
    ax.text(tip_e_label[0] + 0.1, tip_e_label[1] + 0.05,
            'Fleet Size ($\\eta^p$)', ha='left', va='bottom',
            fontsize=25, rotation=rot_e)

    SHOW_ZTICK_LABELS = False  # set True to draw the fleet-size tick values again
    perp_e = np.array([v_e[1], -v_e[0]])
    TICK_LABEL_GAP = 0.38
    TICK_LABEL_ROTATION_EXTRA = 30  # extra tilt on top of rot_e to reduce label crowding
    if SHOW_ZTICK_LABELS:
        for ei, eta_val in enumerate(fleet_sorted):
            tick_pos = O + ei * sp_e * v_e
            anchor = tick_pos + (0.35 * sp_e * v_e if ei == 0 else 0)
            label_pos = anchor + perp_e * TICK_LABEL_GAP
            ax.text(label_pos[0], label_pos[1], str(eta_val),
                    ha='center', va='center', fontsize=12, color='#333',
                    rotation=rot_e + TICK_LABEL_ROTATION_EXTRA, zorder=999)

    # ---- y-axis (distinct station configurations) ----
    tip_s = O + (level_max - level_min + 0.8) * sp_s * v_s
    ax.annotate('', xy=tip_s, xytext=O, arrowprops=aw, zorder=999,
                annotation_clip=False)
    mid_s = O + (level_max - level_min) / 2 * sp_s * v_s
    ax.text(mid_s[0] - 0.4, mid_s[1], 'Charging Station Config. ($|S^p|$)',
            ha='right', va='center', fontsize=25, rotation=90)

    ax.autoscale_view()
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.set_xlim(xl[0] - 4.0, xl[1] + 2.5)
    ax.set_ylim(O[1] - 0.5, yl[1] + 2.5)   # clip everything below the x-axis; labels are placed in points
    plt.tight_layout()

    # =========================================================================
    # x-axis ticks, arrows and labels -- laid out in POINTS (not data units),
    # so spacing no longer depends on how tall the state lattice is. Font size
    # is shrunk just enough that the widest label fits in one period's width.
    # =========================================================================
    TICK_FS, LABEL_FS, STATE_FS, TITLE_FS = 30, 23, 25, 25   # max font sizes
    MIN_FS = 9                 # never shrink below this
    FILL = 0.95 #0.85                # share of a period's width a label may use
    GAP = 6                    # points between stacked items
    ARROW_LEN = 30 #26             # points (at full font size)

    ann = {}                   # period -> (label, state_str, arrow_color)
    for idx in range(1, len(res.path)):
        period, curr_state = res.path[idx]
        _, prev_state = res.path[idx - 1]
        S_curr, eta_curr = curr_state
        S_prev, eta_prev = prev_state
        n_new_stations = len(S_curr.difference(S_prev))
        n_new_ets = eta_curr - eta_prev
        if n_new_stations == 0 and n_new_ets == 0:
            label = "No change"
        elif n_new_stations == 0:
            label = f"{n_new_ets} new ET" + ("s" if n_new_ets != 1 else "")
        elif n_new_ets == 0:
            label = f"{n_new_stations} new station" + ("s" if n_new_stations != 1 else "")
        else:
            label = (f"{n_new_ets} new ET{'s' if n_new_ets != 1 else ''} and\n"
                     f"{n_new_stations} new station{'s' if n_new_stations != 1 else ''}")
        state_str = ("({" + ", ".join(sorted(S_curr, key=lambda x: int(x))) + "}, "
                     + str(eta_curr) + ")")
        ann[period] = (label, state_str, fleet_color_by_value[eta_curr])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px2pt = 72.0 / fig.dpi
    slot_pt = (ax.transData.transform((sp_p, 0))[0]
               - ax.transData.transform((0, 0))[0]) * px2pt

    def _size_pt(s, fs, **kw):
        t = ax.text(0, 0, s, fontsize=fs, **kw)
        bb = t.get_window_extent(renderer)
        t.remove()
        return bb.width * px2pt, bb.height * px2pt

    widest = max([_size_pt(l, LABEL_FS, style='italic')[0] for l, _, _ in ann.values()]
                 + [_size_pt(s, STATE_FS)[0] for _, s, _ in ann.values()]
                 + [1.0])
    scale = min(1.0, FILL * slot_pt / widest)
    label_fs = max(MIN_FS, LABEL_FS * scale)
    state_fs = max(MIN_FS, STATE_FS * scale)
    tick_fs = max(MIN_FS, min(TICK_FS, TICK_FS * max(scale, 0.6)))
    #arrow_len = max(12, ARROW_LEN * scale)
    arrow_len = ARROW_LEN

    tick_h = _size_pt("$10$", tick_fs)[1]
    label_h = max([_size_pt(l, label_fs, style='italic')[1] for l, _, _ in ann.values()] + [0])
    state_h = max([_size_pt(s, state_fs)[1] for _, s, _ in ann.values()] + [0])

    y_tick = -GAP
    y_arrow_top = y_tick - tick_h - GAP
    y_arrow_bot = y_arrow_top - arrow_len
    y_label = y_arrow_bot - GAP
    y_state = y_label - label_h - GAP
    y_title = y_state - state_h - 3 * GAP

    def _at(dy):
        return offset_copy(ax.transData, fig=fig, x=0, y=dy, units='points')

    ax.text(O[0], O[1], f'${initial_period - 1}$', ha='center', va='top',
            fontsize=tick_fs * 0.7, color='#333', zorder=999,
            transform=_at(y_tick), clip_on=False)
    for period in periods_sorted:
        x = proj(period, level_min, 0)[0]
        ax.text(x, O[1], f'${period}$', ha='center', va='top', fontsize=tick_fs,
                color='#333', zorder=999, transform=_at(y_tick), clip_on=False)
        if period not in ann:
            continue
        label, state_str, arrow_color = ann[period]
        ax.annotate('', xy=(x, O[1]), xycoords=_at(y_arrow_bot),
                    xytext=(x, O[1]), textcoords=_at(y_arrow_top),
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                    lw=3.2 * max(scale, 0.6),
                                    mutation_scale=18 * max(scale, 0.6)),
                    zorder=999, annotation_clip=False)
        ax.text(x, O[1], label, ha='center', va='top', fontsize=label_fs,
                color='#222', style='italic', zorder=999,
                transform=_at(y_label), clip_on=False)
        ax.text(x, O[1], state_str, ha='center', va='top', fontsize=state_fs,
                color='#222', zorder=999, transform=_at(y_state), clip_on=False)

    ax.text(mid_p[0], mid_p[1], 'DP Stage  (Planning Period)', ha='center', va='top',
            fontsize=TITLE_FS, zorder=999, transform=_at(y_title), clip_on=False)

    fmt = out_path.rsplit('.', 1)[-1]
    plt.savefig(out_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"3D-style DP network plot saved to {out_path}  "
          f"(x-axis label font scale = {scale:.2f})")


# =============================================================================
# 1. BRUTE FORCE DP  (Section 4.2.1)
#    Full state space every stage. Requires one upfront mu grid covering
#    EVERY (period, state) pair.
# =============================================================================

def bf_export_state_grid(
        path: str,
        states_asc: List[DP_State],
        P: int,
        infeasible_fleet_sizes_by_period: Optional[Dict[int, Set[FleetSize]]] = None,
) -> None:
    """
    Writes a template grid with a header row and ONE leading state-label
    column, e.g.:

        state,                              Period 1 ET Cost, Period 2 ET Cost, ...
        ({1, 10, 2, 3, 4, 5, 6, 7, 8, 9}, 12),               ,                0, ...
        ...

    Rows = reversed(states_asc) (largest station-config first), one row per
    state -- the state label is written ONCE in column A, not repeated
    across every period column.

    infeasible_fleet_sizes_by_period: optional {period: {fleet sizes}}.
    For every state whose fleet size eta is in
    infeasible_fleet_sizes_by_period.get(period, set()), that state's cell
    for that period is pre-filled with 0 (marking it infeasible for the
    routing-only lower-level problem in that period) instead of being left
    blank for you to fill in.

    Example:
        bf_export_state_grid(
            path, states_asc, P,
            infeasible_fleet_sizes_by_period={
                3: {14},        # fleet size 14 infeasible in period 3
                4: {14},
                5: {6, 8},
            },
        )
    """
    infeasible_fleet_sizes_by_period = infeasible_fleet_sizes_by_period or {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state"] + [f"Period {p} ET Cost" for p in range(1, P + 1)])
        for state in reversed(states_asc):
            _, eta = state
            row = [state_to_key(state)]
            for p in range(1, P + 1):
                infeasible_etas = infeasible_fleet_sizes_by_period.get(p, set())
                row.append(0 if eta in infeasible_etas else "")
            w.writerow(row)


def bf_load_mu_grid(path: str, states_asc: List[DP_State], P: int) -> Dict[Tuple[int, DP_State], Cost]:
    """
    Reads the filled-in mu grid CSV back in for brute force. Matches this
    file's format to bf_export_state_grid(): a header row, then one row per
    state with the state label in column A followed by P period-cost
    columns. Rows are matched back to states by the state-label TEXT in
    column A (not by row position), so it's safe if you sort/reorder rows
    in your spreadsheet tool -- as long as the label text itself is left
    untouched, it will still be matched to the correct state.
    """
    states_written = list(reversed(states_asc))
    key_to_state = {state_to_key(s): s for s in states_written}
    mu: Dict[Tuple[int, DP_State], Cost] = {}

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty.")

    header, data_rows = rows[0], rows[1:]
    if not header or header[0].strip().lower() != "state":
        raise ValueError(f"{path}: expected first column header 'state', got {header[:1]!r}.")
    if len(header) != P + 1:
        raise ValueError(f"{path}: expected {P + 1} columns (state + {P} periods), got {len(header)}.")
    if len(data_rows) != len(states_written):
        raise ValueError(
            f"{path} has {len(data_rows)} data row(s) but {len(states_written)} states were expected."
        )

    seen_keys = set()
    for row_num, row in enumerate(data_rows, start=2):
        if len(row) != P + 1:
            raise ValueError(f"Row {row_num} must have {P + 1} columns, got {len(row)}.")
        key = row[0].strip()
        if key not in key_to_state:
            raise ValueError(f"Row {row_num}: unrecognized state label {key!r}.")
        if key in seen_keys:
            raise ValueError(f"Row {row_num}: state label {key!r} appears more than once.")
        seen_keys.add(key)
        state = key_to_state[key]
        for p, cell in enumerate(row[1:], start=1):
            cell = cell.strip()
            if cell == "":
                raise ValueError(f"Missing mu at row {row_num} ('{key}'), period p={p}.")
            val = float(cell)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite mu at row {row_num} ('{key}'), period p={p}.")
            mu[(p, state)] = val

    missing_states = set(key_to_state) - seen_keys
    if missing_states:
        raise ValueError(
            f"{len(missing_states)} state(s) missing from {path}, e.g. {next(iter(missing_states))}."
        )

    return mu


def _bf_run_core(
        E_bar: Set[Candidate_station],
        E0: Set[Candidate_station],
        allowed_fleet_sizes: List[FleetSize],
        P: int,
        lam: Capital_CostMap,
        gam: Service_CostMap,
        fleet_cost: FleetCostMap,
        mu_grid: Dict[Tuple[int, DP_State], Cost],
) -> DPResult:
    eta_0 = min(allowed_fleet_sizes)
    states_asc = feasible_states(E_bar, E0, allowed_fleet_sizes)
    initial_state: DP_State = (frozenset(E0), eta_0)

    print(f"\n{'=' * 70}\nBRUTE FORCE DP\n{'=' * 70}")
    print(f"Total DP states per stage: {len(states_asc)}")
    print(f"Initial state: {state_to_key(initial_state)}\nPlanning periods: {P}\n")

    missing = [(p, s) for p in range(1, P + 1) for s in states_asc if (p, s) not in mu_grid]
    if missing:
        raise ValueError(f"mu missing for {len(missing)} (period, state) pairs.")
    for p in range(1, P + 1):
        for e in E0:
            assert lam[(p, e)] == 0.0, f"lam[(p={p}, e={e})] must be 0.0 for legacy stations."
    assert initial_state in states_asc

    psi: Dict[int, Dict[DP_State, Cost]] = {0: {initial_state: 0.0}}
    pred: Dict[int, Dict[DP_State, Optional[DP_State]]] = {0: {initial_state: None}}
    state_reduction: Dict[int, Tuple[int, int]] = {}

    for period in range(1, P + 1):
        psi[period] = {}
        pred[period] = {}
        skipped = 0
        for curr_state in states_asc:
            if mu_grid.get((period, curr_state), 0.0) == 0.0:
                skipped += 1
                continue
            best_val, best_prev = None, None
            for prev_state, v_prev in psi[period - 1].items():
                if not transition_feasible(prev_state, curr_state):
                    continue
                c = v_prev + stage_transition_cost(
                    period, prev_state, curr_state, lam, gam, fleet_cost,
                    mu_grid[(period, curr_state)]
                )
                # Tie-break: on equal cost, prefer the predecessor with the
                # SMALLER fleet size (i.e., delay fleet growth as long as
                # possible when cost-indifferent), rather than depending on
                # incidental dict iteration order.
                if (best_val is None or c < best_val
                        or (c == best_val and prev_state[1] < best_prev[1])):
                    best_val, best_prev = c, prev_state
            if best_val is not None:
                psi[period][curr_state] = best_val
                pred[period][curr_state] = best_prev

        if not psi[period]:
            raise RuntimeError(f"No reachable states at period p={period}.")
        state_reduction[period] = (len(states_asc), len(psi[period]))
        print(f"  p={period}: {len(psi[period])} reachable ({skipped} infeasible/skipped).")

    last = psi[P]
    # Tie-break: on equal total cost, prefer the final state with the
    # SMALLER fleet size (delay fleet growth when cost-indifferent).
    best_final_state = min(last, key=lambda s: (last[s], s[1]))
    best_cost = last[best_final_state]

    path: List[Tuple[int, DP_State]] = []
    cur = best_final_state
    for period in range(P, -1, -1):
        path.append((period, cur))
        if period == 0:
            break
        cur = pred[period][cur]
    path.reverse()

    return DPResult(best_cost, best_final_state, psi, pred, path,
                     "brute_force", state_reduction)


_FILENAME_RE = re.compile(r"Period_(\d+)_FleetSize_(\d+)\.csv$")


def load_bf_states_grid(path: str):
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows[0], rows[1:]


def save_bf_states_grid(path: str, header, data) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def discover_period_fleet_files(folder: str) -> Dict[Tuple[int, int], str]:
    """
    Scans `folder` for files named Period_{p}_FleetSize_{eta}.csv and returns
    {(period, fleet_size): full_path}. Use this instead of hand-listing every
    file if they're all sitting in one directory.
    """
    found = {}
    for fname in os.listdir(folder):
        m = _FILENAME_RE.search(fname)
        if m:
            period, eta = int(m.group(1)), int(m.group(2))
            found[(period, eta)] = os.path.join(folder, fname)
    return found


def load_solved_file(path: str) -> Dict[FrozenSet[str], Dict[str, float]]:
    """
    Reads one Period_{p}_FleetSize_{eta}.csv file and returns
    {station_config: {"period_cost": ..., "comp_time": ...}}, keyed by the
    STATION SET (the fleet size is fixed for the whole file -- it's baked
    into every row's DP State label already, and matches the filename).
    """
    result: Dict[FrozenSet[str], Dict[str, float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"DP State", "Period Cost", "Comp Time (s)"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"{path}: missing expected column(s) {missing_cols}, "
                              f"found {reader.fieldnames}")
        for row in reader:
            label = row["DP State"]
            config, _eta = parse_state_label(label)
            cost_str = row["Period Cost"].strip()
            time_str = row["Comp Time (s)"].strip()
            if cost_str == "":
                raise ValueError(f"{path}: missing Period Cost for state {label!r}")
            if config in result:
                raise ValueError(f"{path}: duplicate station configuration "
                                  f"{sorted(config)} (check for repeated rows).")
            result[config] = {
                "period_cost": float(cost_str),
                "comp_time": float(time_str) if time_str not in ("", "None") else 0.0,
            }
    return result


def populate_bf_states_grid_and_time_grid(
        grid_path: str,
        period_fleet_files: Dict[Tuple[int, int], str],
        cost_output_path: str,
        time_output_path: str,
) -> None:
    """
    Fills bf_states_grid.csv's blank cells using solved (period, fleet size)
    data -- DIRECTLY matched by (station_config, fleet_size), since every
    fleet size has now actually been solved individually (no broadcasting).

    Simultaneously builds a parallel computation-time grid, same shape as
    the cost grid, populated with each state's Comp Time (s) instead of
    Period Cost. The time grid is built fresh every run (it isn't a
    "preserve prior work" artifact like the cost grid) -- it always reflects
    whatever solved data is available in period_fleet_files, filling every
    cell it can regardless of what was already in bf_states_grid.csv.

    Rules for the COST grid:
      - Cells already containing 0 (pre-marked infeasible) are NEVER touched.
      - Cells already containing a non-zero value are left untouched
        (safe to re-run without overwriting prior work).

    period_fleet_files: {(period, fleet_size): path_to_solved_csv}, e.g.
        {(1, 6): "Period_1_FleetSize_6.csv", (1, 7): "Period_1_FleetSize_7.csv", ...}
        Use discover_period_fleet_files(folder) to build this automatically.
    """
    header, data = load_bf_states_grid(grid_path)
    row_configs_etas = [parse_state_label(row[0]) for row in data]

    period_col_index: Dict[int, int] = {}
    for col_idx in range(1, len(header)):
        m = re.match(r"Period (\d+) ET Cost", header[col_idx])
        if m:
            period_col_index[int(m.group(1))] = col_idx

    # Time grid: fresh header/rows, mirroring the cost grid's 0s (infeasible
    # states have no solve time either) but otherwise starting blank.
    time_header = [header[0]] + [f"Period {p} Comp Time (s)" for p in sorted(period_col_index)]
    time_col_for_period = {p: i + 1 for i, p in enumerate(sorted(period_col_index))}
    time_data = []
    for row in data:
        new_row = [row[0]] + ["0" if row[period_col_index[p]].strip() == "0" else ""
                               for p in sorted(period_col_index)]
        time_data.append(new_row)

    # Pre-load every solved file's contents, grouped by period
    files_by_period: Dict[int, Dict[int, Dict[FrozenSet[str], Dict[str, float]]]] = {}
    for (period, eta), path in period_fleet_files.items():
        files_by_period.setdefault(period, {})[eta] = load_solved_file(path)

    n_cost_filled = 0
    n_skipped_zero = 0
    n_skipped_prefilled = 0
    n_missing = 0
    n_time_filled = 0

    for period, col in period_col_index.items():
        eta_data = files_by_period.get(period, {})
        time_col = time_col_for_period[period]
        period_filled = 0

        for row_idx, row in enumerate(data):
            config, eta = row_configs_etas[row_idx]
            cell = row[col].strip()
            solved = eta_data.get(eta, {}).get(config)

            if cell == "0":
                n_skipped_zero += 1
            elif cell != "":
                n_skipped_prefilled += 1
            elif solved is None:
                n_missing += 1
            else:
                row[col] = f"{solved['period_cost']:.6f}"
                n_cost_filled += 1
                period_filled += 1

            # Time grid always reflects solved data when available, regardless
            # of whether the cost cell above was already filled coming in.
            if solved is not None:
                time_data[row_idx][time_col] = f"{solved['comp_time']:.6f}"
                n_time_filled += 1

        print(f"Period {period}: {period_filled} cost cell(s) filled "
              f"(fleet sizes solved: {sorted(eta_data.keys())}).")

    save_bf_states_grid(cost_output_path, header, data)
    save_bf_states_grid(time_output_path, time_header, time_data)

    print("\n" + "=" * 60)
    print(f"Total cost cells filled   : {n_cost_filled}")
    print(f"Preserved 0s (infeasible) : {n_skipped_zero}")
    print(f"Already filled (skipped)  : {n_skipped_prefilled}")
    if n_missing:
        print(f"WARNING - missing (config, eta) match: {n_missing} cell(s) left blank")
    print(f"Total time cells filled   : {n_time_filled}")
    print(f"Cost grid saved to        : {cost_output_path}")
    print(f"Computation-time grid saved to: {time_output_path}")
    print("=" * 60)


# =============================================================================
# 2. LABELING ALGORITHM  (Section 4.2.2 - 4.2.3)
#    Staged / checkpointed. Only states expanded from Pi[p-1] are candidates
#    at stage p, so mu is only ever needed for those states. Pi[0] is defined
#    as the singleton initial state, which unifies stage 1 with the general
#    recursion (Step 1 in the paper is the p=1 special case of Step p).
# =============================================================================

@dataclass
class LabelingCheckpoint:
    E_bar: Set[Candidate_station]
    E0: Set[Candidate_station]
    allowed_fleet_sizes: List[FleetSize]
    P: int
    lam: Capital_CostMap
    gam: Service_CostMap
    fleet_cost: FleetCostMap
    psi: Dict[int, Dict[DP_State, Cost]] = field(default_factory=dict)      # pre-pruning values, per stage
    pred: Dict[int, Dict[DP_State, Optional[DP_State]]] = field(default_factory=dict)
    Pi: Dict[int, Dict[DP_State, Cost]] = field(default_factory=dict)       # non-dominated (or final-stage full) set
    candidates: Dict[int, Dict[DP_State, Cost]] = field(default_factory=dict)  # == psi, kept separately for clarity
    pending_candidates: Dict[int, List[DP_State]] = field(default_factory=dict)  # awaiting mu import
    state_reduction: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    last_completed_stage: int = 0  # 0 means only the initial state is known
    infeasible_fleet_sizes_by_period: Dict[int, Set[FleetSize]] = field(default_factory=dict)


class LabelingDriver:
    """Internal engine for the labeling algorithm. You normally don't touch
    this class directly -- use run_labeling_export_stage(), run_labeling_
    import_stage(), and run_labeling_finalize() below instead, which handle
    loading/saving the checkpoint file for you."""

    def __init__(self, ckpt: LabelingCheckpoint):
        self.ckpt = ckpt
        self.eta_0 = min(ckpt.allowed_fleet_sizes)
        self.initial_state: DP_State = (frozenset(ckpt.E0), self.eta_0)
        if ckpt.last_completed_stage == 0 and 0 not in ckpt.Pi:
            ckpt.Pi[0] = {self.initial_state: 0.0}
            ckpt.psi[0] = {self.initial_state: 0.0}
            ckpt.pred[0] = {self.initial_state: None}

    @classmethod
    def new(cls, E_bar, E0, allowed_fleet_sizes, P, lam, gam, fleet_cost,
            infeasible_fleet_sizes_by_period: Optional[Dict[int, Set[FleetSize]]] = None) -> "LabelingDriver":
        for p in range(1, P + 1):
            for e in E0:
                assert lam[(p, e)] == 0.0, f"lam[(p={p}, e={e})] must be 0.0 for legacy stations."
        assert all(eta >= 0 for eta in allowed_fleet_sizes)
        ckpt = LabelingCheckpoint(set(E_bar), set(E0), sorted(allowed_fleet_sizes), P, lam, gam, fleet_cost,
                                   infeasible_fleet_sizes_by_period=infeasible_fleet_sizes_by_period or {})
        return cls(ckpt)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.ckpt, f)

    @classmethod
    def load(cls, path: str) -> "LabelingDriver":
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        return cls(ckpt)

    def stage_generate_candidates(self, period: int) -> List[DP_State]:
        """Expand every state in Pi[period-1] by every subset U of remaining
        candidate stations and every feasible (non-decreasing) fleet size.
        For period=1, Pi[0] = {initial_state}, so this reduces exactly to
        Step 1 of the algorithm."""
        c = self.ckpt
        if period < 1 or period > c.P:
            raise ValueError(f"period must be in 1..{c.P}, got {period}")
        if (period - 1) not in c.Pi:
            raise RuntimeError(
                f"Stage {period - 1} has not been computed yet; "
                f"call run_labeling_import_stage({period - 1}, ...) first."
            )

        candidates: Set[DP_State] = set()
        for (S_prev, eta_prev) in c.Pi[period - 1].keys():
            remaining = c.E_bar.difference(S_prev)
            for U in powerset(remaining):
                S_curr = S_prev.union(U)
                for eta_curr in c.allowed_fleet_sizes:
                    if eta_curr < eta_prev:
                        continue
                    candidates.add((S_curr, eta_curr))

        cand_list = sorted(candidates, key=lambda st: (len(st[0]), tuple(sorted(st[0])), st[1]),
                            reverse=True)
        c.pending_candidates[period] = cand_list
        return cand_list

    def export_stage_candidates(self, period: int, path: str) -> List[DP_State]:
        """Generate (if needed) and write this stage's candidate states to a
        CSV template for external mu computation. Row order matters -- it is
        what import_stage_mu_and_advance uses to match values back.

        Any candidate whose fleet size eta is in
        self.ckpt.infeasible_fleet_sizes_by_period.get(period, set()) --
        set once when this driver was created via LabelingDriver.new(...) --
        has its mu_Dp_scaled cell pre-filled with 0, exactly like
        bf_export_state_grid() does for brute force, so the two approaches
        always agree on which states are infeasible in which period."""
        cand_list = self.stage_generate_candidates(period)
        infeasible_etas = self.ckpt.infeasible_fleet_sizes_by_period.get(period, set())
        n_prefilled = 0
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["state", "mu_Dp_scaled"])
            for state in cand_list:
                _, eta = state
                if eta in infeasible_etas:
                    w.writerow([state_to_key(state), 0])
                    n_prefilled += 1
                else:
                    w.writerow([state_to_key(state), ""])
        print(f"Stage {period}: exported {len(cand_list)} candidate state(s) to {path}"
              + (f" ({n_prefilled} pre-filled 0 as infeasible fleet size)" if n_prefilled else ""))
        return cand_list

    def import_stage_mu_and_advance(self, period: int, mu_csv_path: str) -> None:
        """Read the mu column for this stage's candidates, compute each
        candidate's value, then apply Proposition 2 dominance pruning UNLESS
        period == P."""
        c = self.ckpt
        if period not in c.pending_candidates:
            self.stage_generate_candidates(period)
        cand_list = c.pending_candidates[period]
        key_to_state = {state_to_key(s): s for s in cand_list}

        with open(mu_csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
        if rows and rows[0][:1] == ["state"]:
            rows = rows[1:]  # drop header if present
        if len(rows) != len(cand_list):
            raise ValueError(
                f"mu CSV has {len(rows)} data row(s) but stage {period} has "
                f"{len(cand_list)} candidate state(s). Re-export and refill."
            )

        # Matched by state-label TEXT in column A (not by row position), same
        # convention as bf_load_mu_grid -- so rows may be freely reordered
        # (e.g. sorted in a spreadsheet) as long as the label text itself is
        # left untouched.
        mu_by_state: Dict[DP_State, Cost] = {}
        for row_num, row in enumerate(rows, start=1):
            key = row[0].strip()
            if key not in key_to_state:
                raise ValueError(
                    f"{mu_csv_path} row {row_num}: unrecognized state label {key!r} "
                    f"for stage {period}."
                )
            state = key_to_state[key]
            if state in mu_by_state:
                raise ValueError(
                    f"{mu_csv_path} row {row_num}: state label {key!r} appears more than once."
                )
            cell = row[-1].strip()
            if cell == "":
                raise ValueError(f"Missing mu value in {mu_csv_path} for stage {period}, state {key!r}.")
            val = float(cell)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite mu value in {mu_csv_path} for stage {period}, state {key!r}.")
            mu_by_state[state] = val

        missing_states = [s for s in cand_list if s not in mu_by_state]
        if missing_states:
            raise ValueError(
                f"{mu_csv_path}: {len(missing_states)} candidate state(s) missing for "
                f"stage {period}, e.g. {state_to_key(missing_states[0])!r}."
            )

        stage_values: Dict[DP_State, Cost] = {}
        stage_pred: Dict[DP_State, DP_State] = {}
        for curr_state in cand_list:
            mu_val = mu_by_state[curr_state]
            if mu_val == 0.0:
                continue  # infeasible in this period, per convention
            best_val, best_prev = None, None
            for prev_state, v_prev in c.Pi[period - 1].items():
                if not transition_feasible(prev_state, curr_state):
                    continue
                cand_val = v_prev + stage_transition_cost(
                    period, prev_state, curr_state, c.lam, c.gam, c.fleet_cost, mu_val
                )
                # Tie-break: on equal cost, prefer the predecessor with the
                # SMALLER fleet size (delay fleet growth as long as possible
                # when cost-indifferent), matching the brute-force tie-break
                # rather than depending on incidental dict iteration order.
                if (best_val is None or cand_val < best_val
                        or (cand_val == best_val and prev_state[1] < best_prev[1])):
                    best_val, best_prev = cand_val, prev_state
            if best_val is not None:
                stage_values[curr_state] = best_val
                stage_pred[curr_state] = best_prev

        if not stage_values:
            raise RuntimeError(f"No reachable states at period p={period}.")

        c.psi[period] = stage_values
        c.pred[period] = stage_pred
        c.candidates[period] = dict(stage_values)

        if period < c.P:
            pruned = filter_dominated_states(stage_values, c.lam, c.gam, c.fleet_cost, period, c.P)
            c.Pi[period] = pruned
            print(f"Stage {period}: {len(stage_values)} candidate(s) evaluated -> "
                  f"{len(pruned)} non-dominated state(s) kept.")
        else:
            c.Pi[period] = stage_values
            print(f"Stage {period} (final): {len(stage_values)} candidate(s) evaluated, "
                  f"no pruning applied.")

        c.state_reduction[period] = (len(stage_values), len(c.Pi[period]))
        c.last_completed_stage = period
        c.pending_candidates.pop(period, None)

    def finalize(self) -> DPResult:
        c = self.ckpt
        if c.last_completed_stage != c.P:
            raise RuntimeError(
                f"Cannot finalize: last completed stage is {c.last_completed_stage}, "
                f"expected {c.P}. Continue importing remaining stages first."
            )
        last = c.Pi[c.P]
        # Tie-break: on equal total cost, prefer the final state with the
        # SMALLER fleet size (delay fleet growth when cost-indifferent).
        best_final_state = min(last, key=lambda s: (last[s], s[1]))
        best_cost = last[best_final_state]

        path: List[Tuple[int, DP_State]] = []
        cur = best_final_state
        for period in range(c.P, -1, -1):
            path.append((period, cur))
            if period == 0:
                break
            cur = c.pred[period][cur]
        path.reverse()

        return DPResult(
            best_cost=best_cost,
            best_final_state=best_final_state,
            psi=c.psi,
            pred=c.pred,
            path=path,
            algorithm_type="labeling",
            state_reduction=c.state_reduction,
            candidates=c.candidates,
            Pi=c.Pi,
        )

    def verify_dominance_pruning(self) -> bool:
        c = self.ckpt
        all_ok = True
        print("\n" + "=" * 70)
        print("DOMINANCE VERIFICATION (against actual problem instance)")
        print("=" * 70)
        for period in range(1, c.P):
            if period not in c.candidates:
                continue
            candidates = c.candidates[period]
            kept = c.Pi[period]
            dropped = {s: v for s, v in candidates.items() if s not in kept}

            for s1, v1 in kept.items():
                for s2, v2 in kept.items():
                    if s1 == s2:
                        continue
                    if dominates(s1, s2, v1, v2, c.lam, c.gam, c.fleet_cost, period, c.P):
                        print(f"  [FAIL] p={period}: {state_to_key(s1)} dominates "
                              f"surviving state {state_to_key(s2)}.")
                        all_ok = False

            unjustified = []
            for s_drop, v_drop in dropped.items():
                justified = any(
                    dominates(s_keep, s_drop, v_keep, v_drop, c.lam, c.gam, c.fleet_cost, period, c.P)
                    for s_keep, v_keep in kept.items()
                )
                if not justified:
                    unjustified.append(s_drop)

            if unjustified:
                print(f"  [FAIL] p={period}: {len(unjustified)} dropped state(s) unjustified, "
                      f"e.g. {state_to_key(unjustified[0])}.")
                all_ok = False
            else:
                print(f"  [OK]   p={period}: {len(dropped)} dropped state(s) justified; "
                      f"{len(kept)} survivor(s) mutually non-dominated.")
        print("=" * 70)
        print("DOMINANCE VERIFICATION: " + ("PASSED" if all_ok else "FAILED"))
        print("=" * 70)
        return all_ok


# =============================================================================
# PROBLEM DEFINITION
# =============================================================================
# Every tunable input for the case study lives in the CONFIG block directly
# below, as named module-level constants, so each value you set is visible and
# tracked in ONE place. `_problem_definition()` further down holds no literals
# of its own -- it only assembles these constants into the exact tuple both
# pipelines consume. Editing happens here, not in the function.
#
# Both the brute-force and the labeling pipeline call `_problem_definition()`
# themselves (including when you drive the labeling stages across several
# separate "Run" clicks, each of which re-imports this module), so keeping the
# values here -- rather than in the __main__ block -- is what guarantees every
# entry point runs on the identical instance.
# -----------------------------------------------------------------------------

# ---- Station-specific costs, keyed by station ID:
#          station_id -> [ capital cost (lambda), fixed service cost (gamma) ]
# Stations "1"-"5" are shared by the intermediate and full-scale cases;
# "6"-"7" are full-scale only. Base in-service stations (those in E0) are
# forced to capital cost 0 in every period regardless of what is listed here.

STATION_COST_PARAMS: Dict[str, List[float]] = {
    "1": [650.0, 350.0],
    "2": [580.0, 280.0],
    "3": [560.0, 260.0],
    "4": [576.0, 276.0],
    "5": [557.0, 257.0],
    "6": [648.0, 348.0],
    "7": [578.0, 278.0],
}

# ---- Candidate station set (E_bar), base in-service set (E0), planning
# horizon (number of periods), and the fleet sizes the lower level may use.
# ---- Fleet procurement cost per ET. It varies by period: period 1 pays
# FLEET_UNIT_COST, and every later period pays FLEET_UNIT_DECREMENT less:
#     fleet_cost[p] = FLEET_UNIT_COST - (p - 1) * FLEET_UNIT_DECREMENT
FLEET_UNIT_COST: float        = 650000.0
FLEET_UNIT_DECREMENT: float   = 5000.0

# ---- Fleet sizes that are infeasible for the routing-only lower-level problem
# in a given period (too few trucks to serve that period's demand). Every
# (period, state) whose fleet size is listed here has its mu pre-filled with 0
# and is treated as infeasible by BOTH pipelines. Use {} if none apply.

INFEASIBLE_FLEET_SIZES_BY_PERIOD: Dict[int, Set[int]] = {
    2:  {5},
    3:  {5, 6},
    4:  {5, 6, 7, 8},
    5:  {5, 6, 7, 8, 9},
}

E_BAR: Set[str]                = {"1", "2", "3", "4", "5"}
E0: Set[str]                   = {"1"}
NUM_PERIODS: int              = 5
ALLOWED_FLEET_SIZES: List[int] = [5, 6, 7, 8, 9, 10]

# -----------------------------------------------------------------------------

def _build_cost_maps(E_bar: Set[str], E0: Set[str], P: int,
                      lam_build, gam_serve, fleet_unit: float,
                      fleet_unit_decrement: float = 5000.0):
    """
    lam_build / gam_serve: EITHER a single scalar applied to every candidate
    station (original behavior), OR a dict {station_id: value} giving a
    station-specific capital cost (lam_build) / fixed service cost
    (gam_serve). Legacy in-service stations (those in E0) always get
    lam = 0.0 in every period, regardless of what lam_build supplies for
    that station, exactly as before.

    fleet_unit / fleet_unit_decrement: fleet_cost now varies by period
    (DP stage) instead of being a single constant across the whole horizon.
    Period 1 uses fleet_unit; each subsequent period decreases by a fixed
    amount fleet_unit_decrement (default $5,000), i.e.
        fleet_cost[p] = fleet_unit - (p - 1) * fleet_unit_decrement
    """

    def _lookup(param, station):
        if isinstance(param, dict):
            return param[station]
        return param

    lam: Capital_CostMap = {
        (p, e): (0.0 if e in E0 else _lookup(lam_build, e))
        for p in range(1, P + 1) for e in E_bar
    }
    gam: Service_CostMap = {
        (p, e): _lookup(gam_serve, e)
        for p in range(1, P + 1) for e in E_bar
    }
    fleet_cost: FleetCostMap = {
        p: fleet_unit - (p - 1) * fleet_unit_decrement
        for p in range(1, P + 1)
    }
    return lam, gam, fleet_cost


def _problem_definition():
    """Assemble the CONFIG constants above into the exact inputs both
    pipelines consume:
        (E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost)
    This function holds NO literals -- to change the instance, edit the CONFIG
    block above, not here. Fresh copies of the mutable containers are returned
    so callers cannot accidentally mutate the module-level constants.

    INFEASIBLE_FLEET_SIZES_BY_PERIOD is a separate CONFIG constant (it feeds
    the mu templates, not the cost maps) -- import it directly where needed.
    """
    lam_build = {e: STATION_COST_PARAMS[e][0] for e in E_BAR}
    gam_serve = {e: STATION_COST_PARAMS[e][1] for e in E_BAR}

    lam, gam, fleet_cost = _build_cost_maps(
        E_BAR, E0, NUM_PERIODS,
        lam_build=lam_build,
        gam_serve=gam_serve,
        fleet_unit=FLEET_UNIT_COST,
        fleet_unit_decrement=FLEET_UNIT_DECREMENT,
    )

    return (set(E_BAR), set(E0), NUM_PERIODS,
            list(ALLOWED_FLEET_SIZES), lam, gam, fleet_cost)

# =============================================================================
# PUBLIC FUNCTIONS -- call these directly with file paths. No terminal needed.
# =============================================================================

def run_bruteforce_export(out_path: str,
                           infeasible_fleet_sizes_by_period: Optional[Dict[int, Set[FleetSize]]] = None) -> None:
    E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost = _problem_definition()
    states_asc = feasible_states(E_bar, E0, allowed_fleet_sizes)
    bf_export_state_grid(out_path, states_asc, P, infeasible_fleet_sizes_by_period)
    print(f"Exported {len(states_asc)} states x {P} periods to {out_path}")


def run_bruteforce(mu_path: str,
                    checkpoint_path: str = "bf_result.pkl",
                    values_out_path: Optional[str] = None) -> DPResult:      # <-- new param
    E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost = _problem_definition()
    states_asc = feasible_states(E_bar, E0, allowed_fleet_sizes)
    mu_grid = bf_load_mu_grid(mu_path, states_asc, P)
    t0 = time.time()
    res = _bf_run_core(E_bar, E0, allowed_fleet_sizes, P, lam, gam, fleet_cost, mu_grid)
    elapsed = time.time() - t0
    print_optimal_path(res)
    print_cost_breakdown(res, lam, gam, fleet_cost)
    if values_out_path is not None:                                        # <-- new block
        export_dp_values_grid(values_out_path, states_asc, P, res.psi)
        print(f"Brute-force psi values exported to {values_out_path}")
    print(f"Brute force time: {elapsed:.4f}s")
    with open(checkpoint_path, "wb") as f:
        pickle.dump((res, elapsed), f)
    return res

def _write_stage_states_txt(txt_path: str, period: int,
                            cand_list: List[DP_State]) -> None:
    """Write a plain-text listing of a stage's candidate states -- one
    state key per line, in the same order as the exported CSV -- alongside
    the CSV, as a quick human-readable reference."""
    with open(txt_path, "w") as f:
        f.write(f"# Labeling stage p={period}: {len(cand_list)} candidate state(s)\n")
        f.write("# format: ({stations}, fleet_size)\n")
        for state in cand_list:
            f.write(state_to_key(state) + "\n")
    print(f"Stage {period}: also wrote {len(cand_list)} state(s) to {txt_path}")


def run_labeling_export_stage(period: int, out_path: str,
                               checkpoint_path: str = "lbl_checkpoint.pkl",
                               infeasible_fleet_sizes_by_period: Optional[Dict[int, Set[FleetSize]]] = None
                               ) -> List[DP_State]:

    E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost = _problem_definition()
    try:
        driver = LabelingDriver.load(checkpoint_path)
        if (infeasible_fleet_sizes_by_period is not None
                and infeasible_fleet_sizes_by_period != driver.ckpt.infeasible_fleet_sizes_by_period):
            print(f"Note: infeasible_fleet_sizes_by_period argument ignored -- checkpoint at "
                  f"{checkpoint_path} already has infeasibility rules baked in from when it "
                  f"was first created (period 1). Delete the checkpoint and start over if you "
                  f"need to change them.")
    except FileNotFoundError:
        driver = LabelingDriver.new(E_bar, E0, allowed_fleet_sizes, P, lam, gam, fleet_cost,
                                     infeasible_fleet_sizes_by_period=infeasible_fleet_sizes_by_period)
    cand_list = driver.export_stage_candidates(period, out_path)
    _write_stage_states_txt(os.path.splitext(out_path)[0] + ".txt", period, cand_list)
    driver.save(checkpoint_path)
    return cand_list


def run_labeling_import_stage(period: int, mu_path: str,
                               checkpoint_path: str = "lbl_checkpoint.pkl") -> None:
    """
    Import the solved mu values for one labeling stage (mu_path is the CSV
    you filled in from the file run_labeling_export_stage() just wrote),
    compute that stage's values, and apply dominance pruning (skipped
    automatically at the final period).

    Example (stage 1):
        run_labeling_import_stage(
            period=1,
            mu_path="C:/Users/chino/Results/lbl_stage1_mu.csv",
            checkpoint_path="C:/Users/chino/Results/lbl_checkpoint.pkl",
        )
    """
    driver = LabelingDriver.load(checkpoint_path)
    driver.import_stage_mu_and_advance(period, mu_path)
    driver.save(checkpoint_path)


def _labeling_stage_paths(labeling_dir: str, period: int,
                          states_name: str, mu_name: str) -> Tuple[str, str]:
    """Per-stage (states CSV, mu CSV) paths, built by substituting "{p}" in
    the name templates with the period number."""
    return (os.path.join(labeling_dir, states_name.format(p=period)),
            os.path.join(labeling_dir, mu_name.format(p=period)))


def _mu_column_filled(mu_path: str, expected_rows: Optional[int] = None) -> Tuple[bool, str]:
    """Light pre-check that the GAMS-filled mu CSV for one labeling stage is
    ready: it exists, has the expected number of data rows, and every data
    row has a non-blank value in its last column. Returns (ok, reason).
    Full validation still happens later in import_stage_mu_and_advance()."""
    if not os.path.exists(mu_path):
        return False, "file not found"
    with open(mu_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0][:1] == ["state"]:
        rows = rows[1:]
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        return False, "file has no data rows"
    if expected_rows is not None and len(rows) != expected_rows:
        return False, f"has {len(rows)} data row(s), expected {expected_rows}"
    for i, r in enumerate(rows, start=1):
        if r[-1].strip() == "":
            return False, f"row {i} has a blank mu value"
    return True, "ready"


# --- Fill a stage's mu column from the already-solved brute-force grid -------
# Lets run_labeling_stages() skip the external GAMS solve entirely: the
# brute-force run already solves D^p * mu^p for every (station config, fleet
# size) in every period, so a labeling stage's mu values can just be copied
# out of bf_mu_grid_filled.csv. (Ported from scripts/Combine_data.py.)

_STATE_LABEL_RE = re.compile(r"^\(\{(.*?)\},\s*(\d+)\)$")


def parse_state_label(label: str) -> DP_State:
    """Inverse of state_to_key(): '({1, 3, 5}, 13)' -> (frozenset({'1','3','5'}), 13).
    Matching is on the station SET, so label ordering / whitespace never matters."""
    m = _STATE_LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"Could not parse state label: {label!r}")
    stations_str, eta_str = m.groups()
    stations = frozenset(s.strip() for s in stations_str.split(",") if s.strip())
    return stations, int(eta_str)


def load_bf_grid_period_costs(bf_grid_path: str, period: int) -> Dict[DP_State, str]:
    """Read the brute-force mu grid and return {(config, eta): cost_str} for the
    'Period {period} ET Cost' column. Costs are kept as raw strings (so "0"
    stays "0" and formatted decimals aren't re-rounded)."""
    period_col = f"Period {period} ET Cost"
    lookup: Dict[DP_State, str] = {}
    with open(bf_grid_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if period_col not in fields:
            raise ValueError(
                f"{bf_grid_path}: column {period_col!r} not found. Columns: {fields}"
            )
        state_col = fields[0]
        for row in reader:
            lookup[parse_state_label(row[state_col])] = (row[period_col] or "").strip()
    return lookup


def fill_stage_mu_from_bf_grid(
        states_csv_path: str,
        bf_grid_path: str,
        period: int,
        mu_out_path: str,
) -> Tuple[int, int, int]:
    """
    Build a labeling stage's filled mu CSV from the already-solved brute-force
    grid instead of a fresh GAMS solve.

    Reads the stage's exported candidate states (states_csv_path -- the file
    run_labeling_export_stage() wrote), looks up each state's cost in
    bf_grid_path's 'Period {period} ET Cost' column (matched on the parsed
    (station_set, fleet_size)), and writes the states plus a filled
    'mu_Dp_scaled' column to mu_out_path.

    A cell that already holds a value in the states file (including a 0
    pre-filled for an infeasible fleet size) is kept as-is. States with no
    match in the grid for this period are left blank and counted.

    Returns (n_filled, n_kept, n_missing).
    """
    period_costs = load_bf_grid_period_costs(bf_grid_path, period)

    with open(states_csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{states_csv_path} is empty.")
    header, data = rows[0], rows[1:]
    try:
        mu_col = header.index("mu_Dp_scaled")
    except ValueError:
        raise ValueError(f"{states_csv_path}: no 'mu_Dp_scaled' column in header {header}")

    n_filled = n_kept = n_missing = 0
    out_rows: List[List[str]] = []
    for row in data:
        if not row or not row[0].strip():
            continue
        while len(row) <= mu_col:
            row.append("")
        if row[mu_col].strip() != "":
            n_kept += 1
        else:
            cost = period_costs.get(parse_state_label(row[0]))
            if not cost:
                n_missing += 1
            else:
                row[mu_col] = cost
                n_filled += 1
        out_rows.append(row)

    with open(mu_out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)

    print(f"Stage {period}: mu filled from {os.path.basename(bf_grid_path)} "
          f"(period {period} column) -- {n_filled} filled, {n_kept} kept, "
          f"{n_missing} missing  ->  {mu_out_path}")
    return n_filled, n_kept, n_missing


def _problem_signature_mismatch(ckpt: "LabelingCheckpoint", problem) -> List[str]:
    """Return human-readable reasons the checkpoint was built on a different
    problem instance than `problem` (the tuple returned by
    _problem_definition()). Empty list == compatible."""
    E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost = problem
    diffs: List[str] = []
    if set(ckpt.E_bar) != set(E_bar):
        diffs.append(f"E_bar: checkpoint {sorted(ckpt.E_bar)} vs current {sorted(E_bar)}")
    if set(ckpt.E0) != set(E0):
        diffs.append(f"E0: checkpoint {sorted(ckpt.E0)} vs current {sorted(E0)}")
    if ckpt.P != P:
        diffs.append(f"P: checkpoint {ckpt.P} vs current {P}")
    if sorted(ckpt.allowed_fleet_sizes) != sorted(allowed_fleet_sizes):
        diffs.append(f"allowed_fleet_sizes: checkpoint {sorted(ckpt.allowed_fleet_sizes)} "
                     f"vs current {sorted(allowed_fleet_sizes)}")
    if dict(ckpt.lam) != dict(lam):
        diffs.append("lam (station capital-cost map) differs")
    if dict(ckpt.gam) != dict(gam):
        diffs.append("gam (station service-cost map) differs")
    if dict(ckpt.fleet_cost) != dict(fleet_cost):
        diffs.append(f"fleet_cost: checkpoint {dict(ckpt.fleet_cost)} "
                     f"vs current {dict(fleet_cost)}")
    return diffs


def _assert_checkpoint_matches_problem(ckpt: "LabelingCheckpoint",
                                       checkpoint_path: str, problem) -> None:
    diffs = _problem_signature_mismatch(ckpt, problem)
    if diffs:
        raise RuntimeError(
            "Stale labeling checkpoint.\n    " + checkpoint_path + "\n"
            "was built on a different problem instance than the current "
            "_problem_definition():\n"
            + "".join(f"  - {d}\n" for d in diffs)
            + "Its cached per-stage values would be wrong for this instance. "
              "Delete that file (and the lbl_stage*_states.csv / lbl_stage*_mu.csv "
              "beside it), or call run_labeling_stages(..., fresh=True) to clear "
              "them automatically."
        )


def _clear_labeling_artifacts(labeling_dir: str, checkpoint_path: str,
                              periods: Iterable[int],
                              states_name: str, mu_name: str) -> None:
    """fresh=True helper: remove the checkpoint plus this run's per-stage
    states / mu / txt files so the labeling run rebuilds from scratch."""
    removed: List[str] = []
    candidates = [checkpoint_path]
    for p in periods:
        states_path = os.path.join(labeling_dir, states_name.format(p=p))
        candidates.append(states_path)
        candidates.append(os.path.splitext(states_path)[0] + ".txt")
        candidates.append(os.path.join(labeling_dir, mu_name.format(p=p)))
    for path in candidates:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    if removed:
        print(f"fresh=True: removed {len(removed)} existing labeling artifact(s):")
        for path in removed:
            print(f"    {path}")
    else:
        print("fresh=True: no existing labeling artifacts to remove.")


def run_labeling_stages(
        periods: Iterable[int],
        labeling_dir: str = "../Results/labeling_approach",
        checkpoint_path: Optional[str] = None,
        infeasible_fleet_sizes_by_period: Optional[Dict[int, Set[FleetSize]]] = None,
        states_name: str = "lbl_stage{p}_states.csv",
        mu_name: str = "lbl_stage{p}_mu.csv",
        bf_grid_path: Optional[str] = None,
        stop_on_missing_mu: bool = True,
        resume: bool = True,
        fresh: bool = False,
) -> List[int]:
    """
    Automate the per-stage  export -> (solve mu in GAMS) -> import  loop for
    a whole range of labeling stages, so you no longer write out one
    run_labeling_export_stage / run_labeling_import_stage pair per period.

    periods
        Any iterable of consecutive, increasing stage numbers -- e.g.
        range(1, P + 1) or [1, 2, 3, 4, 5].
    labeling_dir
        Folder that holds (and receives) the per-stage CSVs and the
        checkpoint. Per-stage file names come from states_name / mu_name with
        "{p}" replaced by the period number, so stage 3 reads/writes
        {labeling_dir}/lbl_stage3_states.csv and {labeling_dir}/lbl_stage3_mu.csv.
    checkpoint_path
        Defaults to  {labeling_dir}/lbl_checkpoint.pkl .
    infeasible_fleet_sizes_by_period
        Passed straight to run_labeling_export_stage. Only the first stage
        (the one that creates the checkpoint) actually stores it; later
        stages reuse the baked-in rules.
    bf_grid_path
        Optional path to the solved brute-force grid (bf_mu_grid_filled.csv).
        When given, each stage's mu CSV is filled automatically from that
        grid's 'Period {p} ET Cost' column (via fill_stage_mu_from_bf_grid)
        instead of needing a separate GAMS solve -- so the whole labeling run
        goes through end to end in one call. The stage mu CSV is rebuilt from
        the grid every time the stage is (re)processed, so an out-of-date mu
        file is never reused. Leave as None to keep the manual "fill the mu
        CSV in GAMS yourself" workflow.
    stop_on_missing_mu
        True  -> when a stage's filled mu CSV is not there yet, print which
                 file to produce (in GAMS, or by supplying bf_grid_path) and
                 stop cleanly. Re-run this function with the SAME arguments to
                 resume at that stage.
        False -> raise FileNotFoundError instead.
    resume
        True  -> stages already imported into the checkpoint are skipped.
        False -> re-export and re-import every requested stage (only valid
                 when it does not create a gap in the checkpoint).
    fresh
        True  -> before doing anything, delete any existing checkpoint AND the
                 lbl_stage{p}_states.csv / _mu.csv / _states.txt files for the
                 requested periods, so the whole labeling run is rebuilt from
                 scratch on the CURRENT _problem_definition() and the CURRENT
                 bf_grid_path. Use this whenever you have changed the problem
                 definition, the infeasible-fleet-size rules, or the solved
                 brute-force grid since the last run.
        False -> keep and resume from whatever is already on disk (default).

    Safety: if an existing checkpoint is found and its baked-in problem
    instance (E_bar, E0, P, allowed fleet sizes, cost maps) does not match the
    current _problem_definition(), this raises instead of silently resuming a
    stale run -- delete the checkpoint or pass fresh=True.

    Returns the list of stages successfully imported during THIS call. Once
    the final stage (p == P) has been imported, call run_labeling_finalize().
    """
    periods = [int(p) for p in periods]
    if not periods:
        raise ValueError("periods is empty -- pass e.g. range(1, P + 1).")
    if periods != list(range(periods[0], periods[0] + len(periods))):
        raise ValueError(f"periods must be consecutive and increasing, got {periods}.")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(labeling_dir, "lbl_checkpoint.pkl")

    problem = _problem_definition()
    P = problem[2]

    if fresh:
        _clear_labeling_artifacts(labeling_dir, checkpoint_path, periods,
                                  states_name, mu_name)

    last_done = 0
    if os.path.exists(checkpoint_path):
        try:
            existing_ckpt = LabelingDriver.load(checkpoint_path).ckpt
        except Exception as exc:  # corrupt / unreadable checkpoint
            print(f"Warning: could not read {checkpoint_path} ({exc}); "
                  f"treating it as if no stage has been imported yet.")
        else:
            _assert_checkpoint_matches_problem(existing_ckpt, checkpoint_path, problem)
            last_done = existing_ckpt.last_completed_stage

    to_run = list(periods)
    if resume and last_done > 0:
        already = [p for p in to_run if p <= last_done]
        to_run = [p for p in to_run if p > last_done]
        if already:
            print(f"Resume: stage(s) {already} already imported "
                  f"(checkpoint last_completed_stage = {last_done}); skipping.")

    if not to_run:
        print("run_labeling_stages: nothing to do -- every requested stage is "
              f"already imported (last completed = {last_done}).")
        return []

    if to_run[0] > last_done + 1:
        raise ValueError(
            f"Gap: checkpoint has stages 1..{last_done} imported, so the next "
            f"runnable stage is {last_done + 1}, but the first stage to run "
            f"here is {to_run[0]}. Extend the range down to {last_done + 1}."
        )
    if to_run[-1] > P:
        raise ValueError(
            f"Requested stage {to_run[-1]} exceeds P = {P} from the problem definition."
        )

    os.makedirs(labeling_dir, exist_ok=True)
    imported: List[int] = []

    for p in to_run:
        states_path, mu_path = _labeling_stage_paths(labeling_dir, p, states_name, mu_name)

        print("\n" + "-" * 70)
        print(f"LABELING STAGE {p}/{P}  --  export candidate states  ->  {states_path}")
        print("-" * 70)
        cand_list = run_labeling_export_stage(
            period=p,
            out_path=states_path,
            checkpoint_path=checkpoint_path,
            infeasible_fleet_sizes_by_period=infeasible_fleet_sizes_by_period,
        )

        if bf_grid_path is not None:
            # Filling from the already-solved grid is instant, so always
            # rebuild this stage's mu CSV from it rather than trusting a mu
            # file that may already be on disk (it could be stale -- left over
            # from an earlier grid, an earlier problem instance, or a partial
            # run). Stages already imported into the checkpoint never reach
            # this point (the resume logic above drops them first).
            fill_stage_mu_from_bf_grid(states_path, bf_grid_path, p, mu_path)

        ok, reason = _mu_column_filled(mu_path, expected_rows=len(cand_list))
        if not ok:
            grid_hint = (
                f"       (bf_grid_path was {bf_grid_path!r}; check that grid has a\n"
                f"        'Period {p} ET Cost' value for every state in the stage file.)\n"
                if bf_grid_path is not None else ""
            )
            note = (
                f"\nLABELING STAGE {p}: mu file not ready ({reason}).\n"
                f"  1. Solve the lower-level VRP in GAMS for the candidate states in:\n"
                f"       {states_path}\n"
                f"{grid_hint}"
                f"  2. Save the filled 'mu_Dp_scaled' column as:\n"
                f"       {mu_path}\n"
                f"  3. Re-run run_labeling_stages(...) with the SAME arguments -- it\n"
                f"     will skip stages 1..{p - 1} and resume at stage {p}.\n"
            )
            if stop_on_missing_mu:
                print(note)
                if imported:
                    print(f"Imported during this call: {imported}")
                return imported
            raise FileNotFoundError(note)

        print(f"LABELING STAGE {p}/{P}  --  import mu  <-  {mu_path}")
        run_labeling_import_stage(
            period=p,
            mu_path=mu_path,
            checkpoint_path=checkpoint_path,
        )
        imported.append(p)

    print("\n" + "=" * 70)
    print(f"run_labeling_stages: imported stage(s) {imported}.")
    if imported and imported[-1] == P:
        print("All P stages imported -- call run_labeling_finalize(...) next.")
    print("=" * 70)
    return imported


def run_labeling_finalize(checkpoint_path: str = "lbl_checkpoint.pkl",
                           result_path: str = "lbl_result.pkl",
                           nd_states_path: Optional[str] = None,
                           values_out_path: Optional[str] = None) -> DPResult:
    driver = LabelingDriver.load(checkpoint_path)
    t0 = time.time()
    res = driver.finalize()
    elapsed = time.time() - t0
    print_optimal_path(res)
    print_cost_breakdown(res, driver.ckpt.lam, driver.ckpt.gam, driver.ckpt.fleet_cost)
    print_state_reduction_summary(res)
    if values_out_path is not None:
        # Only states that actually survived pruning (i.e., appear in Pi at
        # some period) are included -- NOT the full brute-force-sized lattice.
        touched_states = set()
        for p in range(1, driver.ckpt.P + 1):
            touched_states.update(res.Pi.get(p, {}).keys())
        states_asc = sorted(
            touched_states,
            key=lambda st: (len(st[0]), tuple(sorted(st[0])), st[1])
        )
        export_dp_values_grid(values_out_path, states_asc, driver.ckpt.P,
                               res.psi, Pi=res.Pi)
        print(f"Labeling label values exported to {values_out_path} "
              f"({len(states_asc)} non-dominated state(s) across all periods)")
    print(f"Labeling finalize time: {elapsed:.4f}s")
    driver.verify_dominance_pruning()
    print_non_dominated_states(res)
    if nd_states_path is not None:
        export_non_dominated_states(res, nd_states_path)
    with open(result_path, "wb") as f:
        pickle.dump((res, elapsed), f)
    return res

# =============================================================================
# WORKED EXAMPLE -- edit the paths below to your own folders, then run
# whichever block you need TODAY by uncommenting it (leave the rest
# commented out). No terminal, no flags -- just click "Run" on this file,
# or run these same lines in a Jupyter cell / Python console.
# =============================================================================

if __name__ == "__main__":

    # ---------------- WORKING DIRECTORY ----------------
    # This script lives in <project root>/DP_benchmark_scripts/, but every path
    # below (DP_benchmark_data/..., Results/...) is relative to the PROJECT
    # ROOT. Switch there first so the script runs correctly no matter which
    # working directory the IDE / terminal launches it from.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ---------------- SHARED: problem instance ----------------
    # All tunable values are the module-level CONFIG constants near the top of
    # this file (STATION_COST_PARAMS, E_BAR, E0, NUM_PERIODS,
    # ALLOWED_FLEET_SIZES, FLEET_UNIT_COST, FLEET_UNIT_DECREMENT, and
    # INFEASIBLE_FLEET_SIZES_BY_PERIOD). Edit them there. The unpacking below
    # is only so the plot_dp_network_3d() calls at the end of this block have
    # the values on hand; every pipeline function re-reads the same CONFIG
    # itself, so brute force and labeling always run on the identical instance.
    E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost = _problem_definition()


    # ---------------- BRUTE FORCE ----------------

    # Step 1: export the template (0 pre-filled per INFEASIBLE_FLEET_SIZES_BY_PERIOD
    # above), then go fill in the remaining blank cells externally in GAMS,
    # save the filled version as bf_mu_grid_filled.csv
    run_bruteforce_export(
        out_path="DP_benchmark_data/brute_force_solution/bf_states_grid.csv",
        infeasible_fleet_sizes_by_period=INFEASIBLE_FLEET_SIZES_BY_PERIOD,
    )

    # Step 2 (runs automatically, BEFORE labeling and brute force below):
    # combine every solved Period_{p}_FleetSize_{eta}.csv in the
    # brute_force_solution folder into bf_mu_grid_filled.csv, plus the
    # matching bf_computation_time_grid.csv. Both are rebuilt from the
    # bf_states_grid.csv template on every run, so re-running is safe and
    # always reflects the current per-period files.
    period_fleet_files = discover_period_fleet_files("DP_benchmark_data/brute_force_solution")
    print(f"Discovered {len(period_fleet_files)} (period, fleet size) file(s): "
          f"{sorted(period_fleet_files.keys())}")
    populate_bf_states_grid_and_time_grid(
        grid_path="DP_benchmark_data/brute_force_solution/bf_states_grid.csv",
        period_fleet_files=period_fleet_files,
        cost_output_path="DP_benchmark_data/brute_force_solution/bf_mu_grid_filled.csv",
        time_output_path="DP_benchmark_data/brute_force_solution/bf_computation_time_grid.csv",
    )

    # ---------------- LABELING ----------------
    # One call replaces the whole per-stage export/import ladder. For every
    # stage p in `periods` it runs run_labeling_export_stage(p), fills
    # lbl_stage{p}_mu.csv, then run_labeling_import_stage(p).
    #
    # With bf_grid_path set (below), each stage's mu column is filled
    # automatically from the already-solved brute-force grid, so the whole
    # labeling run finishes in this one call -- no separate GAMS step.
    # Drop bf_grid_path to go back to filling each lbl_stage{p}_mu.csv in
    # GAMS yourself (the call then stops at the first unfilled stage and
    # resumes from there on the next run).
    # fresh=True wipes any existing checkpoint + lbl_stage*_states/_mu/.txt
    # files for these periods first, so the run is rebuilt entirely on the
    # CURRENT _problem_definition() / INFEASIBLE_FLEET_SIZES_BY_PERIOD / grid.
    # Set it whenever you have changed any of those since the last run. Leave
    # it False (default) to resume an interrupted run. Do NOT use fresh=True
    # together with the manual "fill mu in GAMS" workflow (bf_grid_path=None)
    # -- it would delete the mu CSVs you filled by hand.
    run_labeling_stages(
        periods=range(1, P + 1),
        labeling_dir="DP_benchmark_data/labeling_approach",
        checkpoint_path="DP_benchmark_data/labeling_approach/lbl_checkpoint.pkl",
        infeasible_fleet_sizes_by_period=INFEASIBLE_FLEET_SIZES_BY_PERIOD,
        bf_grid_path="DP_benchmark_data/brute_force_solution/bf_mu_grid_filled.csv",
        fresh=True,
    )

    #---------------Brute force & Labeling approach----------------------------
    res_bf = run_bruteforce(
        mu_path="DP_benchmark_data/brute_force_solution/bf_mu_grid_filled.csv",
        checkpoint_path="DP_benchmark_data/brute_force_solution/bf_result.pkl",
        values_out_path="DP_benchmark_data/brute_force_solution/bf_node_values_grid.csv",
    )

    # Once all P periods are imported, finalize:
    res_lbl = run_labeling_finalize(
        checkpoint_path="DP_benchmark_data/labeling_approach/lbl_checkpoint.pkl",
        result_path="DP_benchmark_data/labeling_approach/lbl_result.pkl",
        nd_states_path="DP_benchmark_data/labeling_approach/non_dominated_states.csv",
        values_out_path="DP_benchmark_data/labeling_approach/lbl_label_values_grid.csv",
    )

    #---- shared config universe (for consistent levels across both figures) ----
    used_configs = set()
    for source_psi in (res_bf.psi, res_lbl.psi):
        for period_table in source_psi.values():
            for (S, _eta) in period_table.keys():
                used_configs.add(S)

    # # ---- shared per-period floor source: union of states each result
    # actually occupies in each period, so both figures re-base to the
    # same floor per period. ----
    period_floor_source = {}
    lbl_states_by_period = {
        p: (res_lbl.Pi.get(p, {}) if res_lbl.Pi is not None else res_lbl.psi.get(p, {}))
        for p in set(res_bf.psi.keys()) | set(res_lbl.psi.keys())
    }
    for p in set(res_bf.psi.keys()) | set(res_lbl.psi.keys()):
        combined = set(res_bf.psi.get(p, {}).keys()) | set(lbl_states_by_period.get(p, {}).keys())
        period_floor_source[p] = list(combined)

    plot_dp_network_3d(
        res_bf,
        allowed_fleet_sizes=allowed_fleet_sizes,
        E_bar=E_bar,
        E0=E0,
        out_path="Results/Optimal_path_brute_force_benchmark.pdf",
        configs_universe=used_configs,
        period_floor_source=period_floor_source,
    )

    plot_dp_network_3d(
        res_lbl,
        allowed_fleet_sizes=allowed_fleet_sizes,
        E_bar=E_bar,
        E0=E0,
        out_path="Results/Optimal_path_labeling_benchmark.pdf",
        configs_universe=used_configs,
        period_floor_source=period_floor_source,
    )

    # ---------------- COMPARE (after both pipelines are done) ----------------

    compare_results(res_bf, res_lbl)
    print_path_count_comparison(res_bf, res_lbl)

    pass