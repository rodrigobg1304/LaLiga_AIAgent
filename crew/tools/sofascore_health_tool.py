"""
Sofascore validation tools for the data-collection crew.

Uses tls-client (Go-based TLS fingerprinting) consistent with sofascore_client.py.
"""
import tls_client
from crewai.tools import tool

# Sofascore status codes for fully completed matches (FT, AET, AP)
COMPLETED_STATUS_CODES = {100, 110, 120}

_session = tls_client.Session(client_identifier="chrome_120", random_tls_extension_order=True)


@tool("check_sofascore_health")
def check_sofascore_health() -> dict:
    """
    Check whether the Sofascore API is reachable by hitting the LaLiga seasons
    endpoint. Call this before any round-detection or data-collection step.

    Returns a dict: {"accessible": bool, "status_code": int|None, "error": str|None}.
    If accessible is False the API is blocked or unreachable — skip the run.
    """
    try:
        resp = _session.get("https://api.sofascore.com/api/v1/unique-tournament/8/seasons")
        if resp.status_code == 200:
            return {"accessible": True, "status_code": 200, "error": None}
        return {
            "accessible": False,
            "status_code": resp.status_code,
            "error": f"Unexpected HTTP {resp.status_code}",
        }
    except Exception as exc:
        return {"accessible": False, "status_code": None, "error": str(exc)}


@tool("check_round_status")
def check_round_status(league_id: int, season_id: int, round_id: int) -> dict:
    """
    Validate whether a specific round exists in Sofascore and whether all its
    matches have a final score (completed).

    Inputs:
      league_id  — Sofascore tournament ID (e.g. 8, 17, 23)
      season_id  — Sofascore season ID (e.g. 77559)
      round_id   — round number to check (integer)

    Returns a dict:
      exists          — False if the endpoint returns no events (round not yet played)
      all_finished    — True only when every match has a final status code
      match_count     — total matches in the round
      in_progress     — list of "homeTeam vs awayTeam" strings still in progress
      status          — "complete" | "in_progress" | "not_available" | "error"
      error           — error message string or None
    """
    url = (
        f"https://api.sofascore.com/api/v1/unique-tournament/{league_id}"
        f"/season/{season_id}/events/round/{round_id}"
    )
    try:
        resp = _session.get(url)
        if resp.status_code != 200:
            return {
                "exists": False, "all_finished": False, "match_count": 0,
                "in_progress": [], "status": "not_available",
                "error": f"HTTP {resp.status_code}",
            }
        events = resp.json().get("events", [])
        if not events:
            return {
                "exists": False, "all_finished": False, "match_count": 0,
                "in_progress": [], "status": "not_available", "error": None,
            }

        pending = [
            f"{e['homeTeam']['slug']} vs {e['awayTeam']['slug']}"
            for e in events
            if e.get("status", {}).get("code") not in COMPLETED_STATUS_CODES
        ]
        return {
            "exists": True,
            "all_finished": len(pending) == 0,
            "match_count": len(events),
            "in_progress": pending,
            "status": "complete" if not pending else "in_progress",
            "error": None,
        }
    except Exception as exc:
        return {
            "exists": False, "all_finished": False, "match_count": 0,
            "in_progress": [], "status": "error", "error": str(exc),
        }
