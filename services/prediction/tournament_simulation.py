"""
Monte Carlo tournament simulation for international tournaments.

Flow per iteration:
  1. If no fixed bracket: randomly assign teams to groups of 4
  2. Simulate round-robin group stage (points system)
  3. Top 2 from each group advance to knockout
  4. Simulate knockout bracket (no draws — penalty shootout = 50/50 on draw)

Format (v1 simplified): 32 teams → 8 groups of 4 → 16 qualifiers → R16 → QF → SF → Final
If fewer than 32 teams are available, the list is padded with dummy entries at 1500 Elo
that almost never advance, keeping top-team probabilities accurate.
"""

import numpy as np
from itertools import combinations
from typing import Optional

from football_core.feature_engineering import build_features_qualy, warm_cache

# ── Module-level probability matrix cache (keyed by sorted team tuple) ────────
_prob_cache: dict = {}

DUMMY_PREFIX = "_dummy_"
N_TEAMS = 32
N_GROUPS = 8


# ─────────────────────────────────────────────
# PROBABILITY COMPUTATION
# ─────────────────────────────────────────────

def _match_probs(team_a: str, team_b: str, model, year: Optional[str]) -> tuple:
    """
    Returns (p_a_wins, p_draw, p_b_wins) for a neutral-ground match.
    Uses is_qualifier=0 (tournament context, neutral ground).
    Falls back to balanced prior if feature build fails.
    """
    if team_a.startswith(DUMMY_PREFIX) or team_b.startswith(DUMMY_PREFIX):
        # Dummy teams are very weak — strong bias toward the real team
        if team_a.startswith(DUMMY_PREFIX):
            return 0.10, 0.20, 0.70
        return 0.70, 0.20, 0.10

    try:
        features = build_features_qualy(
            team_a, team_b, year=year, league_id="16", is_qualifier=0
        )
        probs = model.predict_proba(features)[0]
        # Order from training: result_map = {'1':0, 'X':1, '2':2}
        # → probs = [P(team_a wins), P(draw), P(team_b wins)]
        return float(probs[0]), float(probs[1]), float(probs[2])
    except Exception:
        return 0.40, 0.30, 0.30


def build_probability_matrix(teams: list, model, year: Optional[str] = None) -> dict:
    """
    Pre-compute all pairwise match probabilities.
    Result is cached at module level to avoid recomputing on repeated requests.
    """
    cache_key = tuple(sorted(teams))
    if cache_key in _prob_cache:
        return _prob_cache[cache_key]

    matrix = {}
    for a in teams:
        for b in teams:
            if a != b:
                matrix[(a, b)] = _match_probs(a, b, model, year)

    _prob_cache[cache_key] = matrix
    return matrix


# ─────────────────────────────────────────────
# GROUP STAGE
# ─────────────────────────────────────────────

def _simulate_group(group: list, prob_matrix: dict) -> list:
    """
    Simulate round-robin group (6 matches for 4 teams).
    Returns teams sorted by points DESC, then goal difference DESC.
    """
    points = {t: 0 for t in group}
    gd = {t: 0 for t in group}

    for a, b in combinations(group, 2):
        p1, px, p2 = prob_matrix.get((a, b), (0.40, 0.30, 0.30))
        r = np.random.random()
        if r < p1:
            points[a] += 3
            diff = int(np.random.poisson(1.2)) + 1
            gd[a] += diff
            gd[b] -= diff
        elif r < p1 + px:
            points[a] += 1
            points[b] += 1
        else:
            points[b] += 3
            diff = int(np.random.poisson(1.2)) + 1
            gd[b] += diff
            gd[a] -= diff

    return sorted(group, key=lambda t: (points[t], gd[t]), reverse=True)


# ─────────────────────────────────────────────
# KNOCKOUT
# ─────────────────────────────────────────────

def _ko_match(team_a: str, team_b: str, prob_matrix: dict) -> str:
    """Single elimination match. Draw redistributed 50/50 (penalties)."""
    p1, px, p2 = prob_matrix.get((team_a, team_b), (0.40, 0.30, 0.30))
    p_a = p1 + px * 0.5
    p_b = p2 + px * 0.5
    total = p_a + p_b
    return team_a if np.random.random() < p_a / total else team_b


def _simulate_knockout(bracket: list, prob_matrix: dict) -> str:
    """
    Run full knockout bracket until 1 winner.
    bracket length must be a power of 2 (guaranteed by the main flow: 16 teams).
    """
    current = list(bracket)
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            winner = _ko_match(current[i], current[i + 1], prob_matrix)
            next_round.append(winner)
        current = next_round
    return current[0]


# ─────────────────────────────────────────────
# MAIN SIMULATION ENTRY POINT
# ─────────────────────────────────────────────

def run_simulation(
    teams: list,
    model,
    n_simulations: int = 10_000,
    groups: Optional[dict] = None,
    year: Optional[str] = None,
) -> dict:
    """
    Monte Carlo WC tournament simulation (simplified 32-team format).

    Args:
        teams:         List of team names from DB (real teams only).
        model:         Loaded sklearn model with predict_proba.
        n_simulations: Number of Monte Carlo iterations.
        groups:        Optional fixed bracket {group_name: [t1, t2, t3, t4]}.
                       If None, random draw each iteration.
        year:          Season filter for feature building (usually None for intl teams).

    Returns:
        {team: win_probability} for all real teams (dummies excluded).
    """
    warm_cache()

    # ── Pad / trim to exactly N_TEAMS (32) ────────────────────────────────────
    real_teams = list(teams)
    n_real = len(real_teams)

    if n_real > N_TEAMS:
        # Keep top N_TEAMS; caller should pass pre-sorted list by Elo desc
        real_teams = real_teams[:N_TEAMS]

    padded = list(real_teams)
    dummy_count = N_TEAMS - len(padded)
    dummies = [f"{DUMMY_PREFIX}{i}" for i in range(dummy_count)]
    padded.extend(dummies)

    all_teams = padded

    # ── Pre-compute probability matrix ────────────────────────────────────────
    matrix = build_probability_matrix(all_teams, model, year)

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    wins = {t: 0 for t in real_teams}

    for _ in range(n_simulations):
        # Draw groups
        if groups and len(groups) == N_GROUPS:
            sim_groups = [list(v) for v in groups.values()]
        else:
            shuffled = all_teams.copy()
            np.random.shuffle(shuffled)
            sim_groups = [shuffled[i * 4:(i + 1) * 4] for i in range(N_GROUPS)]

        # Group stage → 16 qualifiers (top 2 per group)
        qualified = []
        for g in sim_groups:
            standings = _simulate_group(g, matrix)
            qualified.extend(standings[:2])

        # Shuffle qualified bracket (random seeding for R16)
        np.random.shuffle(qualified)

        # Knockout R16 → Final
        winner = _simulate_knockout(qualified, matrix)

        if winner in wins:
            wins[winner] += 1

    return {t: wins[t] / n_simulations for t in real_teams}
