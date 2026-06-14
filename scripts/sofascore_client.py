"""
Sofascore API client for fetching match statistics.

Uses tls-client (Go-based TLS fingerprinting) to bypass Sofascore WAF protection.

Usage:
    # 1. Populate seasons for each league you need
    get_seasons_dict_result(8)   # LaLiga
    get_seasons_dict_result(17)  # Premier League

    # 2. Collect statistics for a round
    df = collect_data_statistics_from(my_league=8, my_season=77559, my_round=27)
"""
import sys
from pathlib import Path
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import tls_client

sys.path.insert(0, str(Path(__file__).parent.parent / "football-core" / "src"))
from football_core.constants import SOFASCORE_ALL, LEAGUE_TIMEZONES

league_dict = SOFASCORE_ALL

# Shared tls_client session — reused across all requests for connection efficiency.
# chrome_120 fingerprint bypasses Sofascore's WAF (curl_cffi impersonation is blocked).
_session = tls_client.Session(client_identifier="chrome_120", random_tls_extension_order=True)

# Registry populated by calling get_seasons_dict_result() per league.
# Key: league_id (str) → Value: list of {season_id: year} dicts
leagues_seasons_dict: dict[str, list[dict]] = {}

# Sofascore status codes for fully completed matches
# 100 = FT (90 min), 110 = AET (After Extra Time), 120 = AP (After Penalties)
COMPLETED_STATUS_CODES = {100, 110, 120}


def get_seasons_dict(league_id):
    """Get seasons mapping id:year values for a given league. Returns raw HTTP response."""
    url = f'https://api.sofascore.com/api/v1/unique-tournament/{league_id}/seasons'
    return _session.get(url)


def get_seasons_dict_result(league_id):
    """
    Fetch seasons for a league and register them in leagues_seasons_dict.
    Returns the updated leagues_seasons_dict.

    Bug fix vs original notebook: league_seasons_array is now local to each call,
    so calling this for multiple leagues never contaminates other leagues' season lists.
    """
    my_seasons_result = get_seasons_dict(league_id=league_id)
    league_seasons_array = []  # local — avoids cross-league contamination
    for i in range(0, len(my_seasons_result.json()['seasons'])):
        league_seasons_array.append({
            my_seasons_result.json()['seasons'][i]['id']:
            my_seasons_result.json()['seasons'][i]['year']
        })
    leagues_seasons_dict[str(league_id)] = league_seasons_array
    return leagues_seasons_dict


def get_rounds(league_id, season_id):
    """Get all available rounds for a league/season. Returns raw HTTP response."""
    url = f'https://api.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/rounds'
    return _session.get(url)


def get_rounds_list(league_id, season_id):
    """
    Returns the list of round identifiers ready to use in get_matches_by_round().

    For regular leagues:   [1, 2, 3, ..., 38]
    For tournaments:       [1, 2, 3, '5/slug/round-of-16', '27/slug/quarterfinals', ...]

    Each entry with a slug is formatted as '{round}/slug/{slug}' to match
    the Sofascore URL format.
    """
    response = get_rounds(league_id, season_id)
    rounds = []
    for r in response.json().get('rounds', []):
        if 'slug' in r:
            rounds.append(f"{r['round']}/slug/{r['slug']}")
        else:
            rounds.append(r['round'])
    return rounds


def get_matches_by_round(league_id, season_id, round_id):
    """
    Get all matches for a given league/season/round.
    round_id can be int or slug string (e.g. '5/slug/round-of-16').
    Returns raw HTTP response.
    """
    url = f'https://api.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/events/round/{round_id}'
    return _session.get(url)


def get_matches_league_by_round(league_id, season_id, round_id):
    """
    Returns list of [season_id, round_id, match_id, home_team_slug, away_team_slug]
    for all matches in the given round.
    """
    matches_by_round_result = get_matches_by_round(league_id, season_id, round_id)
    matches_league_seasons_array = []
    for i in range(0, len(matches_by_round_result.json()['events'])):
        matches_league_seasons_array.append([
            season_id,
            round_id,
            matches_by_round_result.json()['events'][i]['id'],
            matches_by_round_result.json()['events'][i]['homeTeam']['slug'],
            matches_by_round_result.json()['events'][i]['awayTeam']['slug'],
        ])
    return matches_league_seasons_array


def get_statistics_by_match(match_id):
    """Get match statistics for a single match. Returns raw HTTP response."""
    url = f'https://api.sofascore.com/api/v1/event/{match_id}/statistics'
    return _session.get(url)


def build_match_metadata(event, my_league, my_round) -> dict:
    """
    Extract match metadata from a Sofascore event dict.
    Returns a dict ready for insert_match_metadata().

    MatchDate: UTC datetime from Sofascore startTimestamp.
    MatchDateLocal: same timestamp converted to local timezone via LEAGUE_TIMEZONES.
    homeScoreET / awayScoreET: goals in extra time (None if match ended at 90 min).
    homeScorePen / awayScorePen: penalty shootout score (None if no penalties).
    """
    ts = event['startTimestamp']
    match_date_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    tz_str = LEAGUE_TIMEZONES.get(str(my_league))
    if tz_str:
        match_date_local = match_date_utc.astimezone(ZoneInfo(tz_str)).strftime('%Y-%m-%d %H:%M:%S')
    else:
        match_date_local = None

    return {
        'MatchId':        event['id'],
        'LeagueId':       my_league,
        'SeasonId':       event['season']['id'] if 'season' in event else None,
        'Round':          str(my_round),
        'homeTeam':       event['homeTeam']['slug'],
        'awayTeam':       event['awayTeam']['slug'],
        'MatchDate':      match_date_utc.strftime('%Y-%m-%d %H:%M:%S'),
        'MatchDateLocal': match_date_local,
        'homeScore':      event['homeScore'].get('normaltime'),
        'awayScore':      event['awayScore'].get('normaltime'),
        'homeScoreET':    event['homeScore'].get('overtime'),
        'awayScoreET':    event['awayScore'].get('overtime'),
        'homeScorePen':   event['homeScore'].get('penalties'),
        'awayScorePen':   event['awayScore'].get('penalties'),
    }


def collect_round_metadata(my_league, my_season, my_round, my_match_id=None) -> list[dict]:
    """
    Collect match metadata for all completed matches in a round.
    If my_match_id is given, returns only the metadata for that specific match.

    Requires leagues_seasons_dict to be populated via get_seasons_dict_result() first.
    Returns a list of dicts ready for insert_match_metadata().
    """
    for seasons in leagues_seasons_dict[str(my_league)]:
        for season_key in seasons:
            if my_season == season_key:
                matches_result = get_matches_by_round(my_league, my_season, my_round)
                metadata_list = []
                for event in matches_result.json().get('events', []):
                    if event['status']['code'] not in COMPLETED_STATUS_CODES:
                        continue
                    if my_match_id is not None and event['id'] != my_match_id:
                        continue
                    metadata_list.append(build_match_metadata(event, my_league, my_round))
                return metadata_list

    print(f"Season {my_season} not found in leagues_seasons_dict for league {my_league}.")
    return []


def collect_round_fixtures(my_league, my_season, my_round) -> list[dict]:
    """
    Collect fixture metadata for ALL matches in a round (completed or not).
    Useful for upcoming rounds where scores are not yet available.

    For unplayed matches homeScore/awayScore may be absent — scores will be None.
    Returns a list of dicts ready for insert_match_metadata().
    """
    get_seasons_dict_result(my_league)
    matches_result = get_matches_by_round(my_league, my_season, my_round)
    fixtures = []
    for event in matches_result.json().get('events', []):
        ts = event['startTimestamp']
        match_date_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        tz_str = LEAGUE_TIMEZONES.get(str(my_league))
        match_date_local = (
            match_date_utc.astimezone(ZoneInfo(tz_str)).strftime('%Y-%m-%d %H:%M:%S')
            if tz_str else None
        )
        home_score = event.get('homeScore', {})
        away_score = event.get('awayScore', {})
        fixtures.append({
            'MatchId':        event['id'],
            'LeagueId':       my_league,
            'SeasonId':       event['season']['id'] if 'season' in event else None,
            'Round':          str(my_round),
            'homeTeam':       event['homeTeam']['slug'],
            'awayTeam':       event['awayTeam']['slug'],
            'MatchDate':      match_date_utc.strftime('%Y-%m-%d %H:%M:%S'),
            'MatchDateLocal': match_date_local,
            'homeScore':      home_score.get('normaltime'),
            'awayScore':      away_score.get('normaltime'),
            'homeScoreET':    home_score.get('overtime'),
            'awayScoreET':    away_score.get('overtime'),
            'homeScorePen':   home_score.get('penalties'),
            'awayScorePen':   away_score.get('penalties'),
        })
    return fixtures


def _build_match_statistics_df(event, my_league, my_round, season_year):
    """
    Internal helper. Builds the statistics DataFrame for a single completed event dict.
    Handles regular time (1ST, 2ND), extra time (ET1, ET2, ET) and penalties (P).
    Returns a DataFrame or raises KeyError/TimeoutError/ConnectTimeout on failure.
    """
    match_key     = event['id']
    home_team_key = event['homeTeam']['slug']
    away_team_key = event['awayTeam']['slug']

    # Goals — regular time
    goals_home_1st = event['homeScore']['period1']
    goals_away_1st = event['awayScore']['period1']
    goals_home_2nd = event['homeScore']['normaltime'] - goals_home_1st
    goals_away_2nd = event['awayScore']['normaltime'] - goals_away_1st

    # Goals — extra time (None if match ended in 90 min)
    goals_home_et = event['homeScore'].get('overtime')
    goals_away_et = event['awayScore'].get('overtime')

    # Goals — penalties (None if no penalty shootout)
    goals_home_pen = event['homeScore'].get('penalties')
    goals_away_pen = event['awayScore'].get('penalties')

    statistics_by_match_result = get_statistics_by_match(match_id=match_key)
    statistics_temp_df = pd.json_normalize(
        data=statistics_by_match_result.json()['statistics'],
        meta=['period'],
        record_path=['groups', ['statisticsItems']],
    )

    # Goals are not included in the statistics endpoint — add them manually
    goals_rows = [
        {'name': 'Goals', 'homeValue': goals_home_1st, 'awayValue': goals_away_1st,
         'homeTotal': np.nan, 'awayTotal': np.nan, 'period': '1ST'},
        {'name': 'Goals', 'homeValue': goals_home_2nd, 'awayValue': goals_away_2nd,
         'homeTotal': np.nan, 'awayTotal': np.nan, 'period': '2ND'},
    ]
    if goals_home_et is not None:
        goals_rows.append(
            {'name': 'Goals', 'homeValue': goals_home_et, 'awayValue': goals_away_et,
             'homeTotal': np.nan, 'awayTotal': np.nan, 'period': 'ET'}
        )
    if goals_home_pen is not None:
        goals_rows.append(
            {'name': 'Goals', 'homeValue': goals_home_pen, 'awayValue': goals_away_pen,
             'homeTotal': np.nan, 'awayTotal': np.nan, 'period': 'P'}
        )

    statistics_temp_df = pd.concat(
        [statistics_temp_df, pd.DataFrame(goals_rows)], ignore_index=True
    )

    # Keep all periods: regular (1ST, 2ND), extra time (ET1, ET2, ET), penalties (P)
    valid_periods = {'1ST', '2ND', 'ET1', 'ET2', 'ET', 'P'}
    statistics_temp_df = statistics_temp_df[statistics_temp_df['period'].isin(valid_periods)]

    statistics_temp_df['LeagueId'] = my_league
    statistics_temp_df['Year']     = season_year
    statistics_temp_df['SeasonId'] = event['season']['id'] if 'season' in event else None
    statistics_temp_df['MatchId']  = match_key
    statistics_temp_df['homeTeam'] = home_team_key
    statistics_temp_df['awayTeam'] = away_team_key
    statistics_temp_df['Round']    = my_round
    statistics_temp_df = statistics_temp_df[[
        'name', 'homeValue', 'awayValue', 'homeTotal', 'awayTotal',
        'period', 'LeagueId', 'Year', 'SeasonId', 'MatchId', 'Round',
        'homeTeam', 'awayTeam',
    ]]
    statistics_temp_df.reset_index(drop=True, inplace=True)
    return statistics_temp_df


def collect_match_statistics(my_league, my_season, my_round, my_match_id):
    """
    Collect statistics for a single match.
    Useful when only one specific match needs to be inserted.

    Requires leagues_seasons_dict to be populated via get_seasons_dict_result()
    before calling this function.

    Returns a DataFrame with the same columns as collect_data_statistics_from.
    """
    for seasons in leagues_seasons_dict[str(my_league)]:
        for season_key in seasons:
            if my_season == season_key:
                season_year = seasons.get(season_key)
                matches_by_round_result = get_matches_by_round(league_id=my_league, season_id=my_season, round_id=my_round)
                for event in matches_by_round_result.json()['events']:
                    if event['id'] != my_match_id:
                        continue
                    if event['status']['code'] not in COMPLETED_STATUS_CODES:
                        print(f'Match not completed ({event["status"]["description"]}): {event["homeTeam"]["slug"]} vs {event["awayTeam"]["slug"]}')
                        return pd.DataFrame()
                    try:
                        return _build_match_statistics_df(event, my_league, my_round, season_year)
                    except (KeyError, TimeoutError, Exception) as e:
                        print(f"Error collecting match {my_match_id}: {e}")
                        return pd.DataFrame()
                print(f"Match {my_match_id} not found in round {my_round} for league {my_league}.")
                return pd.DataFrame()

    print(f"Season {my_season} not found in leagues_seasons_dict for league {my_league}. Call get_seasons_dict_result({my_league}) first.")
    return pd.DataFrame()


def collect_data_statistics_from(my_league, my_season, my_round):
    """
    Main data collection function. Fetches statistics for all completed matches
    (Sofascore status code 100) in a given league/season/round.

    Requires leagues_seasons_dict to be populated via get_seasons_dict_result()
    before calling this function.

    Returns a DataFrame with columns:
        name, homeValue, awayValue, homeTotal, awayTotal, period,
        LeagueId, Year, SeasonId, MatchId, Round, homeTeam, awayTeam
    """
    statistics_df = pd.DataFrame()
    for seasons in leagues_seasons_dict[str(my_league)]:
        for season_key in seasons:
            if my_season == season_key:
                season_year = seasons.get(season_key)
                print(f'League: {league_dict.get(str(my_league))}, season_name: {season_year}, season_id: {my_season}')
                print(f'Round: {my_round}')
                matches_by_round_result = get_matches_by_round(league_id=my_league, season_id=my_season, round_id=my_round)

                for i in range(0, len(matches_by_round_result.json()['events'])):
                    time.sleep(10)
                    event = matches_by_round_result.json()['events'][i]
                    home_team_key = event['homeTeam']['slug']
                    away_team_key = event['awayTeam']['slug']

                    if event['status']['code'] in COMPLETED_STATUS_CODES:
                        try:
                            statistics_temp_df = _build_match_statistics_df(event, my_league, my_round, season_year)
                            statistics_df = pd.concat([statistics_df, statistics_temp_df], ignore_index=True)
                        except (KeyError, TimeoutError, Exception) as e:
                            print(f"Error in {home_team_key} vs {away_team_key}: {e}")
                    else:
                        error_status = event['status']['description']
                        print(f'Match {error_status}: {home_team_key} vs {away_team_key}')

                return statistics_df

    print(f"Season {my_season} not found in leagues_seasons_dict for league {my_league}. Call get_seasons_dict_result({my_league}) first.")
    return pd.DataFrame()
