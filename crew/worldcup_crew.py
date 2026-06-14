"""
CrewAI crew for daily World Cup 2026 data collection.

Two agents collaborate in sequence:
  1. monitor_agent  — pre-flight Sofascore API health check
  2. fetcher_agent  — collects ALL available rounds via collect_tournaments.py

The World Cup uses knockout + group-stage rounds with slug identifiers
(e.g. 'round-of-32', 'quarterfinals'), so round detection via MAX(Round)+1
is not applicable. Instead, --all-rounds is used and INSERT IGNORE in
db_utils ensures daily runs never produce duplicates.

Run via main.py (scheduled daily at 09:00 Europe/Madrid) or directly:
    python worldcup_crew.py
"""
import os
from crewai import Agent, Task, Crew, Process, LLM

from tools.sofascore_health_tool import check_sofascore_health
from tools.subprocess_tool import collect_tournament_rounds

# 2026 World Cup: Sofascore tournament ID 16, season ID 58210
WORLDCUP_LEAGUE_ID = 16
WORLDCUP_SEASON_ID = 58210


def _llm() -> LLM:
    model_name = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
    if not model_name.startswith("anthropic/"):
        model_name = f"anthropic/{model_name}"
    return LLM(model=model_name, api_key=os.environ.get("ANTHROPIC_API_KEY"))


def build_worldcup_crew() -> Crew:
    llm = _llm()

    # ── Agents ────────────────────────────────────────────────────────────────

    monitor_agent = Agent(
        role="API Monitor",
        goal="Verify that the Sofascore API is accessible before any data collection begins.",
        backstory=(
            "You are a reliability engineer responsible for pre-flight checks. "
            "Before the pipeline runs, you confirm external APIs are reachable. "
            "If Sofascore is blocked or unreachable, you report it clearly so the "
            "rest of the pipeline can skip gracefully."
        ),
        tools=[check_sofascore_health],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    fetcher_agent = Agent(
        role="World Cup Data Fetcher",
        goal=(
            "Collect completed World Cup 2026 match statistics from Sofascore "
            "for ALL available rounds in a single daily run."
        ),
        backstory=(
            "You are the executor in the World Cup pipeline. You call collect_tournament_rounds "
            "once per day to sweep all completed rounds (group stage + knockout). "
            "Because INSERT IGNORE is used in the database, it is always safe to collect "
            "all rounds — duplicates are silently skipped and only new completed matches "
            "get inserted. Report the full stdout and any errors."
        ),
        tools=[collect_tournament_rounds],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    validation_task = Task(
        description=(
            "Call the check_sofascore_health tool. "
            "If accessible is True, output exactly: 'API_OK'. "
            "If accessible is False, output exactly: 'SOFASCORE_BLOCKED: <error message>'. "
            "Do not call any other tool."
        ),
        expected_output="Either 'API_OK' or 'SOFASCORE_BLOCKED: <reason>' as a single line.",
        agent=monitor_agent,
    )

    fetch_worldcup_task = Task(
        description=(
            "Check the validation_task output.\n"
            "- If it contains 'SOFASCORE_BLOCKED': output 'World Cup: Sofascore blocked, skipping.' "
            "and do not call any tool.\n"
            "- If it is 'API_OK': call collect_tournament_rounds with "
            f"league_id={WORLDCUP_LEAGUE_ID}, season_id={WORLDCUP_SEASON_ID}, "
            "tournament_type='qualifier'. "
            "Report the full result including stdout (rows inserted per round) and any errors."
        ),
        expected_output=(
            "Collection result dict with success/stdout/stderr showing rows inserted per round, "
            "or a skip message if Sofascore was blocked."
        ),
        agent=fetcher_agent,
        context=[validation_task],
    )

    # ── Crew ──────────────────────────────────────────────────────────────────

    return Crew(
        agents=[monitor_agent, fetcher_agent],
        tasks=[validation_task, fetch_worldcup_task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    crew = build_worldcup_crew()
    result = crew.kickoff()
    print("\n── World Cup crew final output ──")
    print(result)
