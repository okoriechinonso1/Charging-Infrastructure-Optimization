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
#              LABELING ALGORITHM  (Sections 4.2.2 - 4.2.3)
#                 run_labeling_export_stage(period, out_path, checkpoint_path)
#                 run_labeling_import_stage(period, mu_path, checkpoint_path)
#                 run_labeling_finalize(checkpoint_path, result_path)
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
# SHARED TYPES AND UTILITIES
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
    stations), ordered by |S| ascending then lexicographic. Via the
    Pi[0]={initial} unification, this coincides with the labeling algorithm's
    stage-1 candidate set -- see stage_generate_candidates."""
    configs = [S for S in powerset(E_bar) if frozenset(E0).issubset(S)]
    configs.sort(key=lambda S: (len(S), tuple(sorted(S))))
    return configs


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


# ---------- DOMINANCE (Proposition 2) ----------
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
    psi: Dict[int, Dict[DP_State, Cost]]           # value function per stage (pre-pruning)
    pred: Dict[int, Dict[DP_State, Optional[DP_State]]]
    path: List[Tuple[int, DP_State]]
    algorithm_type: str                              # "labeling"
    state_reduction: Dict[int, Tuple[int, int]]       # period -> (generated, kept)
    candidates: Optional[Dict[int, Dict[DP_State, Cost]]] = None
    Pi: Optional[Dict[int, Dict[DP_State, Cost]]] = None


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
    driver.verify_dominance_pruning().
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
    header row + one state-label column (state label in column A, then
    one column per period).

    A cell is populated ONLY if the state survived dominance pruning at that
    period, i.e. is a member of Pi[p]. This guarantees every exported value
    is the label value V^p(E^p, eta^p) of a non-dominated state, never a
    candidate that was later dominated and discarded. At the final period P,
    Pi[P] equals the full candidate set (Proposition 2 dominance is undefined
    at p=P), so that column shows every state reachable in period P.
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


def plot_dp_network(res: DPResult, out_path: str) -> None:
    """
    Visualize the DP state network and highlight the optimal expansion path.
    Draws only the surviving (non-dominated) states per stage. out_path is
    the full file path (including filename) where the .svg is written.
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

    is_labeling = res.algorithm_type == "labeling"

    # ---- per-period states actually carried (same convention as plot_dp_network) ----
    per_period_states = {}
    for period, table in res.psi.items():
        if is_labeling and res.Pi is not None and period in res.Pi:
            per_period_states[period] = list(res.Pi[period].keys())
        else:
            per_period_states[period] = list(table.keys())

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

    NODE_R = 0.40 #0.35

    # Same fixed spacing as the sample illustration -- sp_p, sp_s, sp_e are
    # plain constants, not dynamically adjusted.
    sp_p = 10.0 #10.0 #
    sp_s = 1.0 #1.0 #0.6
    sp_e =  1.0 #1.4 #

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
        levels_here = [config_level[S] for (S, _eta) in source_states]
        period_floor[period] = min(levels_here) if levels_here else level_min

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
        for prev_state in per_period_states[p0]:
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
                                        zorder=950 if is_path_node else zb + 3)
                ax.add_patch(circ)

        bot = proj(period, level_min, 0)
        ax.text(bot[0], O[1] - 0.15, f'${period}$', ha='center', va='top',
                fontsize=30, color='#333', zorder=zb + 4)

    # ---- per-period "what changed" annotation + in-service station config ----
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

        bot = proj(period, level_min, 0)
        arrow_color = fleet_color_by_value[eta_curr]
        zb = (P - period + 1) * 30


    ########################################################################################
        arrow_top = (bot[0], O[1] - 1.0)  # was 1.2
        arrow_bottom = (bot[0], O[1] - 3.0)  # was 2.6 -- arrow length now 1.0
        ax.annotate('', xy=arrow_bottom, xytext=arrow_top,
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                    lw=3.2, mutation_scale=18),
                    zorder=zb + 4)
        ax.text(bot[0], O[1] - 3.8, label, ha='center', va='top',  # was 4.6
                fontsize=23, color='#222', zorder=zb + 4, style='italic')
        ax.text(bot[0], O[1] - 5.3, state_str, ha='center', va='top',  # was 8.2
                fontsize=25, color='#222', zorder=zb + 4)
    ax.text(O[0], O[1] - 0.15, f'${initial_period - 1}$', ha='center', va='top',
            fontsize=20, color='#333', zorder=999)

    # ---- optimal path edges (via real rendered positions) ----
    path_xy = [state_xy[(p, s)] for p, s in res.path]
    for i in range(len(path_xy) - 1):
        a, b = path_xy[i], path_xy[i + 1]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=PATH_COLOR, lw=4.5,
                alpha=0.5, zorder=895, solid_capstyle='round')

    aw = dict(arrowstyle='->', color='#000', lw=2.2, mutation_scale=18)

    """
    ---- x-axis (DP stage) ----
    """
    tip_p = np.array([proj(P, level_min, 0)[0] + sp_p * 0.8, O[1]])
    ax.annotate('', xy=tip_p, xytext=O, arrowprops=aw, zorder=999,
                annotation_clip=False)
    mid_p = np.array([(O[0] + tip_p[0]) / 2, O[1]])
    # ax.text(mid_p[0], mid_p[1] - 8.0, 'DP Stage  (Planning Period)',
    #         ha='center', va='top', fontsize=25)

    ax.text(mid_p[0], mid_p[1] - 9.0, 'DP Stage  (Planning Period)',  # was 11.8
            ha='center', va='top', fontsize=25)

    # ##########################################################################################
    #     arrow_top = (bot[0], O[1] - 1.0)  # was 1.2
    #     arrow_bottom = (bot[0], O[1] - 3.0)  # was 2.6 -- arrow length now 1.0
    #     ax.annotate('', xy=arrow_bottom, xytext=arrow_top,
    #                 arrowprops=dict(arrowstyle='->', color=arrow_color,
    #                                 lw=3.2, mutation_scale=18),
    #                 zorder=zb + 4)
    #     ax.text(bot[0], O[1] - 3.8, label, ha='center', va='top',  # was 4.6
    #             fontsize=23, color='#222', zorder=zb + 4, style='italic')
    #     ax.text(bot[0], O[1] - 5.3, state_str, ha='center', va='top',  # was 8.2
    #             fontsize=25, color='#222', zorder=zb + 4)
    #
    # ax.text(O[0], O[1] - 0.15, f'${initial_period - 1}$', ha='center', va='top',
    #         fontsize=16, color='#333', zorder=999)
    #
    # # ---- optimal path edges (via real rendered positions) ----
    # path_xy = [state_xy[(p, s)] for p, s in res.path]
    # for i in range(len(path_xy) - 1):
    #     a, b = path_xy[i], path_xy[i + 1]
    #     ax.plot([a[0], b[0]], [a[1], b[1]], color=PATH_COLOR, lw=4.5,
    #             alpha=0.5, zorder=895, solid_capstyle='round')
    #
    # aw = dict(arrowstyle='->', color='#000', lw=2.2, mutation_scale=18)
    #
    # # ---- x-axis (DP stage) ----
    # tip_p = np.array([proj(P, level_min, 0)[0] + sp_p * 0.8, O[1]])
    # ax.annotate('', xy=tip_p, xytext=O, arrowprops=aw, zorder=999,
    #             annotation_clip=False)
    # mid_p = np.array([(O[0] + tip_p[0]) / 2, O[1]])
    #
    # ax.text(mid_p[0], mid_p[1] - 10.2, 'DP Stage  (Planning Period)',  # was 9.0
    #         ha='center', va='top', fontsize=25)
    ##############################################################################

    # ---- z-axis (fleet size) -- with actual value ticks ----
    tip_e = O + (n_e - 1 + 0.8) * sp_e * v_e
    ax.annotate('', xy=tip_e, xytext=O, arrowprops=aw, zorder=999,
                annotation_clip=False)
    rot_e = np.degrees(np.arctan2(v_e[1], v_e[0]))
    tip_e_label = O + (n_e - 1 + 1.1) * sp_e * v_e
    ax.text(tip_e_label[0] + 0.1, tip_e_label[1] + 0.05,
            'Fleet Size ($\\eta^p$)', ha='left', va='bottom',
            fontsize=25, rotation=rot_e)
    #
    # perp_e = np.array([v_e[1], -v_e[0]])
    # TICK_LABEL_GAP = 0.38
    # for ei, eta_val in enumerate(fleet_sorted):
    #     tick_pos = O + ei * sp_e * v_e
    #     anchor = tick_pos + (0.35 * sp_e * v_e if ei == 0 else 0)  # push "6" up the line first
    #     label_pos = anchor + perp_e * TICK_LABEL_GAP
    #     ax.text(label_pos[0], label_pos[1], str(eta_val),
    #             ha='center', va='center', fontsize=20, color='#333',
    #             rotation=rot_e, zorder=999)

    SHOW_ZTICK_LABELS = False  # set True to draw the fleet-size tick values again
    perp_e = np.array([v_e[1], -v_e[0]])
    TICK_LABEL_GAP = 0.38
    TICK_LABEL_ROTATION_EXTRA = 30  # extra tilt on top of rot_e to reduce label crowding
    if SHOW_ZTICK_LABELS:
        for ei, eta_val in enumerate(fleet_sorted):
            tick_pos = O + ei * sp_e * v_e
            anchor = tick_pos + (0.35 * sp_e * v_e if ei == 0 else 0)  # push "6" up the line first
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
    ax.set_ylim(yl[0] - 5.5, yl[1] + 2.5)

    plt.tight_layout()
    fmt = out_path.rsplit('.', 1)[-1]
    plt.savefig(out_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"3D-style DP network plot saved to {out_path}")



# =============================================================================
# LABELING ALGORITHM  (Section 4.2.2 - 4.2.3)
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
        has its mu_Dp_scaled cell pre-filled with 0 (infeasible in this
        period)."""
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

        # Matched by state-label TEXT in column A (not by row position) -- so
        # rows may be freely reordered (e.g. sorted in a spreadsheet) as long
        # as the label text itself is left untouched.
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
                # when cost-indifferent), rather than depending on incidental
                # dict iteration order.
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
# of its own -- it only assembles these constants into the exact tuple the
# labeling pipeline consumes. Editing happens here, not in the function.
#
# The labeling pipeline calls `_problem_definition()` itself (including when
# you drive the labeling stages across several separate "Run" clicks, each of
# which re-imports this module), so keeping the values here -- rather than in
# the __main__ block -- is what guarantees every entry point runs on the
# identical instance.
# -----------------------------------------------------------------------------

# ---- Station-specific costs, keyed by station ID:
#          station_id -> [ capital cost (lambda), fixed service cost (gamma) ]
# Stations "1"-"5" are shared by the intermediate and full-scale cases;
# "6"-"7" are full-scale only. Base in-service stations (those in E0) are
# forced to capital cost 0 in every period regardless of what is listed here.
STATION_COST_PARAMS: Dict[str, List[float]] = {
    "1": [650000.0, 350000.0],
    "2": [580000.0, 280000.0],
    "3": [560000.0, 260000.0],
    "4": [576000.0, 276000.0],
    "5": [557000.0, 257000.0],
    "6": [648000.0, 348000.0],
    "7": [578000.0, 278000.0],
}

# ---- Candidate station set (E_bar), base in-service set (E0), planning
# horizon (number of periods), and the fleet sizes the lower level may use.
E_BAR: Set[str]                = {"1", "2", "3", "4", "5", "6", "7"}
E0: Set[str]                   = {"1"}
NUM_PERIODS: int              = 10
ALLOWED_FLEET_SIZES: List[int] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# ---- Fleet procurement cost per ET. It varies by period: period 1 pays
# FLEET_UNIT_COST, and every later period pays FLEET_UNIT_DECREMENT less:
#     fleet_cost[p] = FLEET_UNIT_COST - (p - 1) * FLEET_UNIT_DECREMENT
FLEET_UNIT_COST: float        = 650000.0
FLEET_UNIT_DECREMENT: float   = 5000.0

# ---- Fleet sizes that are infeasible for the routing-only lower-level problem
# in a given period (too few trucks to serve that period's demand). Every
# (period, state) whose fleet size is listed here has its mu pre-filled with 0
# and is treated as infeasible. Use {} if none apply.
INFEASIBLE_FLEET_SIZES_BY_PERIOD: Dict[int, Set[int]] = {
    3:  {5, 6},
    4:  {5, 6, 7},
    5:  {5, 6, 7, 8, 9},
    6:  {5, 6, 7, 8, 9},
    7:  {5, 6, 7, 8, 9, 10, 11},
    8:  {5, 6, 7, 8, 9, 10, 11, 12, 13},
    9:  {5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
    10: {5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
}


# ---- ALTERNATE PRESET: 'Intermediate Case' (5 stations, 5 periods).
# To use it, comment out the four full-scale lines above (E_BAR ...
# ALLOWED_FLEET_SIZES) plus INFEASIBLE_FLEET_SIZES_BY_PERIOD, and uncomment:
# E_BAR               = {"1", "2", "3", "4", "5"}
# E0                  = {"1"}
# NUM_PERIODS         = 5
# ALLOWED_FLEET_SIZES = [5, 6, 7, 8, 9, 10]
# INFEASIBLE_FLEET_SIZES_BY_PERIOD = {
#     2: {5},
#     3: {5, 6},
#     4: {5, 6, 7, 8},
#     5: {5, 6, 7, 8, 9},
# }


# ---- Settings for the AUTOMATED labeling pipeline, i.e. calling
# run_labeling_stages(..., charging_opt_config=...) so each stage's mu is
# SOLVED directly via scripts/run_charging_optimization.py (GAMS) instead of
# a manual step. See CHARGING_OPT_CONFIG below, and its use in the __main__
# block at the bottom of this file.
DAYS_PER_PERIOD: float = 1000  # 500 working days per 3-year planning period

# ---- File locations. Built from this script's own location, so they work
# whatever the working directory the IDE / terminal launches it from:
#   <project>/DP_FullScale_data/{period_name}/   per-period inputs
#   <project>/DP_FullScale_data/labeling_approach/   labeling results + checkpoint
#   <project>/Figures/   figures
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "DP_FullScale_data")
LABELING_DIR = os.path.join(DATA_ROOT, "labeling_approach")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "Figures")

CHARGING_OPT_PERIOD_FOLDER_TEMPLATE = os.path.join(DATA_ROOT, "{period_name}", "augmented_routes_for_charging")
CHARGING_OPT_ARC_PARAMS_PATH        = os.path.join(DATA_ROOT, "{period_name}", "charging_arcs_params.csv")
# Physical parameters forwarded to run_charging_optimization.solve_dp_states.
# Leave this empty ({}) to use solve_dp_states' own defaults instead.
CHARGING_OPT_SOLVER_KWARGS: Dict = {
    "driving_range": 150,
    "tour_duration": 480,
    "vehicle_capacity": 20000,
    "depot": "151",
    "lowest_range": 10,
    "delta_min": 10,
    "delta_max": 240,
    "soc_reduction_rate": 1.0,
    "big_m": 1000,
    "cost_per_mile": 6.50,
}


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
    """Assemble the CONFIG constants above into the exact inputs the
    labeling pipeline consumes:
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
                               checkpoint_path: str = os.path.join(LABELING_DIR, "lbl_checkpoint.pkl"),
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
                               checkpoint_path: str = os.path.join(LABELING_DIR, "lbl_checkpoint.pkl")) -> None:
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


# --- Fill a stage's mu column by SOLVING the lower-level problem directly ----
# This runs scripts/run_charging_optimization.py's GAMS solve for this
# stage's candidate states, so the whole labeling run can go end to end
# without a manual GAMS step.

def fill_stage_mu_from_charging_opt(
        states_csv_path: str,
        mu_out_path: str,
        period: int,
        period_folder: str,
        charging_arc_params_path: str,
        days_per_period: float,
        num_routes: Optional[int] = None,
        solver_kwargs: Optional[Dict] = None,
) -> Tuple[int, int]:
    """
    Build a labeling stage's filled mu CSV by SOLVING the lower-level
    charging-second optimization (scripts/run_charging_optimization.py)
    directly -- no manual GAMS step needed.

    Reads the stage's exported candidate states (states_csv_path -- the file
    run_labeling_export_stage() wrote). A row that ALREADY carries a value in
    its mu_Dp_scaled cell -- the 0 pre-filled by export_stage_candidates()
    for any candidate whose fleet size was declared infeasible for this
    period via infeasible_fleet_sizes_by_period -- is kept as-is and is NOT
    solved: the "0 = infeasible fleet size" convention used everywhere else
    in this module is respected here too, so an infeasible state's GAMS
    solve is skipped entirely rather than solved and then overwritten.

    Every other (blank-cell) row's (station config, fleet size) is solved via
    run_charging_optimization.solve_dp_states(..., strict=True) -- a
    genuinely-attempted state that fails to solve raises immediately rather
    than silently writing a 0 (which would be indistinguishable from the
    infeasibility convention downstream). Its 'total_period_cost' (already
    D^p * mu^p, scaled by days_per_period) is written into mu_Dp_scaled.

    num_routes: leave as None (the default) to auto-detect how many routes
    exist for this period by scanning period_folder (see
    run_charging_optimization.discover_route_numbers) -- preferred, since it
    can never drift out of sync with what's actually on disk. Pass an
    explicit int only to deliberately override/cap it.

    solver_kwargs: optional dict of extra keyword arguments forwarded to
    solve_dp_states (driving_range, tour_duration, vehicle_capacity, depot,
    lowest_range, delta_min, delta_max, soc_reduction_rate, big_m,
    cost_per_mile) -- omit to use solve_dp_states' own defaults.

    Returns (n_solved, n_kept_prefilled).
    """
    # Lazy import (needs GAMS) using a bare module name, matching this
    # project's scripts/-relative import convention (see CLAUDE.md's note on
    # CG_Algorithm_Main.py) -- run_charging_optimization.py itself imports
    # its GAMS/data-loading helpers the same bare way, so it only resolves
    # correctly with the scripts/ folder itself on sys.path, not just the
    # project root. This module (Charging_infrastructure_DP_Algorithm_v8.py)
    # lives in scripts/ too, so we add ITS OWN directory, which works
    # regardless of the caller's current working directory.
    import sys
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import Run_charging_optimization as _rco

    with open(states_csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{states_csv_path} is empty.")
    header, data = rows[0], rows[1:]
    try:
        mu_col = header.index("mu_Dp_scaled")
    except ValueError:
        raise ValueError(f"{states_csv_path}: no 'mu_Dp_scaled' column in header {header}")

    out_rows: List[List[str]] = []
    to_solve: List[Tuple[List[int], int]] = []
    label_row_index: Dict[str, int] = {}
    n_kept = 0

    for row in data:
        if not row or not row[0].strip():
            continue
        row = list(row)
        while len(row) <= mu_col:
            row.append("")
        label = row[0].strip()
        if row[mu_col].strip() != "":
            n_kept += 1  # already pre-filled (0 = infeasible fleet size) -- skip solving
        else:
            stations, fleet = parse_state_label(label)
            to_solve.append(([int(s) for s in stations], fleet))
            label_row_index[label] = len(out_rows)
        out_rows.append(row)

    n_solved = 0
    if to_solve:
        solved = _rco.solve_dp_states(
            period=period,
            dp_states=to_solve,
            period_folder=period_folder,
            num_routes=num_routes,
            charging_arc_params_path=charging_arc_params_path,
            days_per_period=days_per_period,
            strict=True,
            **(solver_kwargs or {}),
        )
        for label, row_idx in label_row_index.items():
            if label not in solved:
                raise RuntimeError(
                    f"{states_csv_path}: state {label!r} was not returned by "
                    f"solve_dp_states (this should not happen with strict=True)."
                )
            out_rows[row_idx][mu_col] = f"{solved[label]['total_period_cost']:.6f}"
            n_solved += 1

    with open(mu_out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)

    print(f"Stage {period}: mu solved via run_charging_optimization -- "
          f"{n_solved} solved, {n_kept} kept (pre-filled infeasible)  ->  {mu_out_path}")
    return n_solved, n_kept


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
        labeling_dir: str = LABELING_DIR,
        checkpoint_path: Optional[str] = None,
        infeasible_fleet_sizes_by_period: Optional[Dict[int, Set[FleetSize]]] = None,
        states_name: str = "lbl_stage{p}_states.csv",
        mu_name: str = "lbl_stage{p}_mu.csv",
        charging_opt_config: Optional[Dict] = None,
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
    charging_opt_config
        Optional dict that, when given, SOLVES each stage's candidate states
        directly via scripts/run_charging_optimization.py (GAMS) instead of
        waiting on a manually-filled CSV -- so the whole labeling run,
        including the lower-level solve, goes through end to end in one
        call. States whose mu cell was already pre-filled 0 by
        export_stage_candidates() (infeasible fleet size for that period,
        per infeasible_fleet_sizes_by_period) are left as-is and never
        solved. Required keys:
            period_folder_template : str, e.g.
                "{period_name}/augmented_routes_for_charging_5"
            period_names            : {period: period_name} dict, e.g.
                run_charging_optimization.PERIOD_NAMES
            charging_arc_params_path : str -- may contain "{period_name}",
                which is filled in per period exactly like
                period_folder_template (a path without the placeholder is
                used unchanged for every period)
            days_per_period          : float
        Optional keys:
            num_routes : int -- leave unset (the default) to auto-detect how
                many routes exist per period by scanning period_folder for
                'route_<N>_aug_travel.csv' files (see
                run_charging_optimization.discover_route_numbers), which can
                never drift out of sync with what's actually on disk. Pass
                an explicit int only to deliberately override/cap it.
            solver_kwargs : dict forwarded to solve_dp_states() for the
                physical parameters (driving_range, tour_duration,
                vehicle_capacity, depot, lowest_range, delta_min, delta_max,
                soc_reduction_rate, big_m, cost_per_mile) -- omit to use
                solve_dp_states' own defaults.
    stop_on_missing_mu
        True  -> when a stage's filled mu CSV is not there yet, print which
                 file to produce (in GAMS, or by supplying
                 charging_opt_config) and stop cleanly. Re-run this function
                 with the SAME arguments to resume at that stage.
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
                 charging_opt_config. Use this whenever you have changed the
                 problem definition, the infeasible-fleet-size rules, or the
                 lower-level settings since the last run.
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

    if charging_opt_config is not None:
        required_keys = ("period_folder_template", "period_names",
                          "charging_arc_params_path", "days_per_period")
        missing_keys = [k for k in required_keys if k not in charging_opt_config]
        if missing_keys:
            raise ValueError(f"charging_opt_config missing required key(s): {missing_keys}")

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

        if charging_opt_config is not None:
            # Solved fresh every time this stage is (re)processed, rather
            # than trusting a mu file that may already be on disk (it could
            # be stale). Stages already imported into the checkpoint never
            # reach this point (the resume logic above drops them first).
            period_name = charging_opt_config["period_names"][p]
            period_folder = charging_opt_config["period_folder_template"].format(
                period_name=period_name
            )
            arc_params_path = charging_opt_config["charging_arc_params_path"].format(
                period_name=period_name
            )
            fill_stage_mu_from_charging_opt(
                states_csv_path=states_path,
                mu_out_path=mu_path,
                period=p,
                period_folder=period_folder,
                num_routes=charging_opt_config.get("num_routes"),
                charging_arc_params_path=arc_params_path,
                days_per_period=charging_opt_config["days_per_period"],
                solver_kwargs=charging_opt_config.get("solver_kwargs"),
            )

        ok, reason = _mu_column_filled(mu_path, expected_rows=len(cand_list))
        if not ok:
            note = (
                f"\nLABELING STAGE {p}: mu file not ready ({reason}).\n"
                f"  1. Solve the lower-level VRP in GAMS for the candidate states in:\n"
                f"       {states_path}\n"
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


def run_labeling_finalize(checkpoint_path: str = os.path.join(LABELING_DIR, "lbl_checkpoint.pkl"),
                           result_path: str = os.path.join(LABELING_DIR, "lbl_result.pkl"),
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
        # some period) are included -- NOT the full feasible lattice.
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

    # ---------------- SHARED: problem instance ----------------
    # All tunable values are the module-level CONFIG constants near the top of
    # this file (STATION_COST_PARAMS, E_BAR, E0, NUM_PERIODS,
    # ALLOWED_FLEET_SIZES, FLEET_UNIT_COST, FLEET_UNIT_DECREMENT, and
    # INFEASIBLE_FLEET_SIZES_BY_PERIOD). Edit them there. The unpacking below
    # is only so the plot_dp_network_3d() call at the end of this block has
    # the values on hand; every pipeline function re-reads the same CONFIG
    # itself, so every entry point runs on the identical instance.
    E_bar, E0, P, allowed_fleet_sizes, lam, gam, fleet_cost = _problem_definition()

    # ---------------- LABELING ----------------
    # One call replaces the whole per-stage export/import ladder. For every
    # stage p in `periods` it runs run_labeling_export_stage(p), fills
    # lbl_stage{p}_mu.csv, then run_labeling_import_stage(p).
    #
    # Two ways to fill each stage's mu -- pick ONE:
    #   (a) charging_opt_config (ACTIVE below) -- SOLVES every stage directly
    #       via scripts/run_charging_optimization.py (GAMS), fully automated,
    #       no manual step needed. DAYS_PER_PERIOD,
    #       CHARGING_OPT_PERIOD_FOLDER_TEMPLATE, CHARGING_OPT_ARC_PARAMS_PATH,
    #       and CHARGING_OPT_SOLVER_KWARGS above are what feed this -- edit
    #       THOSE constants, not this dict.
    #   (b) omit it -- fills each lbl_stage{p}_mu.csv in GAMS yourself (the
    #       call then stops at the first unfilled stage and resumes from
    #       there on the next run).
    #
    # fresh=True wipes any existing checkpoint + lbl_stage*_states/_mu/.txt
    # files for these periods first, so the run is rebuilt entirely on the
    # CURRENT _problem_definition() / INFEASIBLE_FLEET_SIZES_BY_PERIOD /
    # charging_opt_config. Set it whenever you have changed any of those
    # since the last run. Leave it False (default) to resume an interrupted
    # run. Do NOT use fresh=True together with the manual "fill mu in GAMS
    # yourself" workflow (option (b)) -- it would delete the mu CSVs you
    # filled by hand.
    import Run_charging_optimization as run_charging_optimization  # bare import; scripts/ is this file's own directory

    CHARGING_OPT_CONFIG = {
        "period_folder_template": CHARGING_OPT_PERIOD_FOLDER_TEMPLATE,
        "period_names": run_charging_optimization.PERIOD_NAMES,
        "charging_arc_params_path": CHARGING_OPT_ARC_PARAMS_PATH,
        "days_per_period": DAYS_PER_PERIOD,
        # "num_routes": 5,  # uncomment to override auto-detection per period
        "solver_kwargs": CHARGING_OPT_SOLVER_KWARGS,
    }

    run_labeling_stages(
        periods=range(1, P + 1),
        labeling_dir=LABELING_DIR,
        checkpoint_path=os.path.join(LABELING_DIR, "lbl_checkpoint.pkl"),
        infeasible_fleet_sizes_by_period=INFEASIBLE_FLEET_SIZES_BY_PERIOD,
        charging_opt_config=CHARGING_OPT_CONFIG,
        fresh=True,
    )

    # Once all P periods are imported, finalize:
    res_lbl = run_labeling_finalize(
        checkpoint_path=os.path.join(LABELING_DIR, "lbl_checkpoint.pkl"),
        result_path=os.path.join(LABELING_DIR, "lbl_result.pkl"),
        nd_states_path=os.path.join(LABELING_DIR, "non_dominated_states.csv"),
        values_out_path=os.path.join(LABELING_DIR, "lbl_label_values_grid.csv"),
    )

    # ---- config universe (for consistent levels in the figure) ----
    used_configs = set()
    for period_table in res_lbl.psi.values():
        for (S, _eta) in period_table.keys():
            used_configs.add(S)

    # ---- per-period floor source: states the labeling result actually
    # occupies in each period (non-dominated set Pi[p]). ----
    period_floor_source = {
        p: list((res_lbl.Pi.get(p, {}) if res_lbl.Pi is not None else res_lbl.psi.get(p, {})).keys())
        for p in res_lbl.psi.keys()
    }

    plot_dp_network_3d(
        res_lbl,
        allowed_fleet_sizes=allowed_fleet_sizes,
        E_bar=E_bar,
        E0=E0,
        out_path=os.path.join(FIGURES_DIR, "Optimal_path_labeling_fullscale_v2.pdf"),
        configs_universe=used_configs,
        period_floor_source=period_floor_source,
    )

    pass