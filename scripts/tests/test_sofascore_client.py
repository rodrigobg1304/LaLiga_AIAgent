"""
Integration tests for sofascore_client.py against the real Sofascore API.

Run from the scripts/ directory:
    pytest tests/test_sofascore_client.py -v
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from sofascore_client import (
    get_seasons_dict,
    get_seasons_dict_result,
    get_statistics_by_match,
    get_matches_by_round,
    get_matches_league_by_round,
    leagues_seasons_dict,
)

# ─── Test data ────────────────────────────────────────────────────────────────
LALIGA_LEAGUE_ID = 8
LALIGA_SEASON_ID = 77559
LALIGA_MATCH_ID  = 14083675
LALIGA_ROUND     = 9


# ─── get_seasons_dict ─────────────────────────────────────────────────────────

def test_get_seasons_dict_returns_200():
    response = get_seasons_dict(LALIGA_LEAGUE_ID)
    assert response.status_code == 200

def test_get_seasons_dict_contains_seasons():
    response = get_seasons_dict(LALIGA_LEAGUE_ID)
    data = response.json()
    assert 'seasons' in data
    assert len(data['seasons']) > 0


# ─── get_seasons_dict_result ──────────────────────────────────────────────────

def test_get_seasons_dict_result_populates_registry():
    result = get_seasons_dict_result(LALIGA_LEAGUE_ID)
    assert str(LALIGA_LEAGUE_ID) in result

def test_get_seasons_dict_result_contains_test_season():
    get_seasons_dict_result(LALIGA_LEAGUE_ID)
    season_ids = [
        season_id
        for entry in leagues_seasons_dict[str(LALIGA_LEAGUE_ID)]
        for season_id in entry
    ]
    assert LALIGA_SEASON_ID in season_ids

def test_bug_fix_no_cross_league_contamination():
    """
    Bug fix: calling get_seasons_dict_result for two leagues must not
    contaminate each other's season lists.
    """
    get_seasons_dict_result(8)   # LaLiga
    get_seasons_dict_result(17)  # Premier League

    laliga_ids  = {sid for e in leagues_seasons_dict['8']  for sid in e}
    premier_ids = {sid for e in leagues_seasons_dict['17'] for sid in e}

    assert laliga_ids.isdisjoint(premier_ids), (
        "Season IDs from different leagues are mixed — mutable state bug is back"
    )


# ─── get_statistics_by_match ──────────────────────────────────────────────────

def test_get_statistics_by_match_returns_200():
    response = get_statistics_by_match(LALIGA_MATCH_ID)
    assert response.status_code == 200

def test_get_statistics_by_match_has_statistics_key():
    response = get_statistics_by_match(LALIGA_MATCH_ID)
    assert 'statistics' in response.json()

def test_get_statistics_by_match_has_1st_and_2nd_periods():
    response = get_statistics_by_match(LALIGA_MATCH_ID)
    periods = [s['period'] for s in response.json()['statistics']]
    assert '1ST' in periods
    assert '2ND' in periods


# ─── get_matches_by_round ─────────────────────────────────────────────────────

def test_get_matches_by_round_returns_200():
    response = get_matches_by_round(LALIGA_LEAGUE_ID, LALIGA_SEASON_ID, LALIGA_ROUND)
    assert response.status_code == 200

def test_get_matches_by_round_contains_test_match():
    response = get_matches_by_round(LALIGA_LEAGUE_ID, LALIGA_SEASON_ID, LALIGA_ROUND)
    match_ids = [e['id'] for e in response.json().get('events', [])]
    assert LALIGA_MATCH_ID in match_ids, (
        f"Match {LALIGA_MATCH_ID} not found in round {LALIGA_ROUND} — check LALIGA_ROUND constant"
    )


# ─── get_matches_league_by_round ──────────────────────────────────────────────

def test_get_matches_league_by_round_returns_list():
    result = get_matches_league_by_round(LALIGA_LEAGUE_ID, LALIGA_SEASON_ID, LALIGA_ROUND)
    assert isinstance(result, list)
    assert len(result) > 0

def test_get_matches_league_by_round_row_structure():
    result = get_matches_league_by_round(LALIGA_LEAGUE_ID, LALIGA_SEASON_ID, LALIGA_ROUND)
    for row in result:
        assert len(row) == 5
        assert row[0] == LALIGA_SEASON_ID
        assert row[1] == LALIGA_ROUND
        assert isinstance(row[3], str)
        assert isinstance(row[4], str)
