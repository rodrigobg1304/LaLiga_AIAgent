"""
CrewAI crew for automated football data collection + model retraining.

Four agents collaborate in sequence:
  1. monitor_agent        — pre-flight Sofascore API health check
  2. round_detector_agent — per-league round detection (MySQL) + Sofascore validation
  3. fetcher_agent        — invokes collect_leagues.py for each ready league
  4. training_agent       — notified by fetcher when collection ends; retrains all
                            production models autonomously

Run via main.py (scheduled) or directly:
    python crew.py
"""
import os
from crewai import Agent, Task, Crew, Process, LLM

from tools.mysql_tool import get_next_round, get_all_next_rounds
from tools.sofascore_health_tool import check_sofascore_health, check_round_status
from tools.subprocess_tool import collect_league_round
from tools.training_tool import run_training_script, get_training_pipeline


def _llm() -> LLM:
    model_name = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
    # LiteLLM requires the "anthropic/" prefix for Claude models.
    if not model_name.startswith("anthropic/"):
        model_name = f"anthropic/{model_name}"
    return LLM(model=model_name, api_key=os.environ.get("ANTHROPIC_API_KEY"))


def build_crew() -> Crew:
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

    round_detector_agent = Agent(
        role="Round Detector",
        goal=(
            "Determine the next unprocessed round for each league by querying MySQL, "
            "then verify that round is fully finished in Sofascore."
        ),
        backstory=(
            "You are a data engineer who knows the pipeline state. You query MySQL to "
            "find the highest round already stored per league, compute next_round = max + 1 "
            "(or 1 if no data exists), then hit Sofascore to confirm the round exists and "
            "all matches have a final score. Your output is a structured league-status list "
            "that the fetcher uses to decide whether to collect or skip."
        ),
        tools=[get_all_next_rounds, get_next_round, check_round_status],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=6,
    )

    fetcher_agent = Agent(
        role="Data Fetcher",
        goal="Collect match statistics from Sofascore for every league whose round is confirmed complete.",
        backstory=(
            "You are the executor in the pipeline. You receive the round-detection results "
            "and for each league marked 'proceed' you run the collection script. "
            "For leagues marked 'pending' or 'blocked' you log a skip message and do nothing. "
            "You report the detailed outcome (rows inserted, any errors) for every league."
        ),
        tools=[collect_league_round],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=6,
    )

    training_agent = Agent(
        role="Model Trainer",
        goal=(
            "Retrain all production ML models autonomously once the data collection "
            "phase has completed, regardless of how many leagues were updated."
        ),
        backstory=(
            "You are an ML engineer responsible for keeping production models up to date. "
            "You receive a notification from the Data Fetcher when the collection phase ends. "
            "You then run each training script in the correct order — RF 1X2, XGBoost, "
            "Over/Under goals, saves, and corners — reporting success or failure per script. "
            "You work autonomously: you do not ask for confirmation, you do not skip steps, "
            "and you continue even if one script fails so that as many models as possible "
            "are updated. If every league was skipped during collection (no new data), "
            "you still run training so the models incorporate any manual DB updates."
        ),
        tools=[get_training_pipeline, run_training_script],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=12,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    validation_task = Task(
        description=(
            "Call the check_sofascore_health tool. "
            "If accessible is True, output exactly: 'API_OK'. "
            "If accessible is False, output exactly: 'SOFASCORE_BLOCKED: <error message>'. "
            "Do not call any other tool."
        ),
        expected_output=(
            "Either 'API_OK' or 'SOFASCORE_BLOCKED: <reason>' as a single line."
        ),
        agent=monitor_agent,
    )

    round_detection_task = Task(
        description=(
            "Step 1: Check the validation_task output. "
            "If it contains 'SOFASCORE_BLOCKED', immediately output a JSON list where every "
            "league has status='blocked' — do not call any tools.\n\n"
            "Step 2 (only if API_OK): Call get_all_next_rounds to retrieve the next round for "
            "each league from MySQL.\n\n"
            "Step 3: For each league in the result, call check_round_status with its "
            "league_id, season_id, and next_round. Use the status field:\n"
            "  - 'complete'       → set league status to 'proceed'\n"
            "  - 'in_progress'    → set league status to 'pending' (matches still live)\n"
            "  - 'not_available'  → set league status to 'pending' (round not yet played)\n"
            "  - 'error'          → set league status to 'pending'\n\n"
            "Output a JSON array — one object per league — with these exact keys:\n"
            "  league_name, league_id, season_id, next_round, status\n\n"
            "League season IDs: LaLiga(8)=77559, Premier League(17)=76986, Serie A(23)=76457."
        ),
        expected_output=(
            "JSON array of three league-status objects:\n"
            '[{"league_name":"LaLiga","league_id":8,"season_id":77559,"next_round":29,"status":"proceed"},'
            '{"league_name":"Premier League","league_id":17,"season_id":76986,"next_round":27,"status":"pending"},'
            '{"league_name":"Serie A","league_id":23,"season_id":76457,"next_round":31,"status":"proceed"}]'
        ),
        agent=round_detector_agent,
        context=[validation_task],
    )

    fetch_laliga_task = Task(
        description=(
            "Read the round_detection_task output JSON. Find the entry where league_id = 8 (LaLiga).\n"
            "- If status is 'proceed': call collect_league_round(league_id=8, round_id=<next_round>). "
            "Report the full result including stdout.\n"
            "- If status is 'pending': output 'LaLiga round <next_round>: not ready yet, skipping.'\n"
            "- If status is 'blocked': output 'LaLiga: Sofascore blocked, skipping.'"
        ),
        expected_output=(
            "One of: collection result dict with success/stdout/stderr, "
            "or a skip message explaining the reason."
        ),
        agent=fetcher_agent,
        context=[round_detection_task],
    )

    fetch_premier_task = Task(
        description=(
            "Read the round_detection_task output JSON. Find the entry where league_id = 17 (Premier League).\n"
            "- If status is 'proceed': call collect_league_round(league_id=17, round_id=<next_round>). "
            "Report the full result including stdout.\n"
            "- If status is 'pending': output 'Premier League round <next_round>: not ready yet, skipping.'\n"
            "- If status is 'blocked': output 'Premier League: Sofascore blocked, skipping.'"
        ),
        expected_output=(
            "One of: collection result dict with success/stdout/stderr, "
            "or a skip message explaining the reason."
        ),
        agent=fetcher_agent,
        context=[round_detection_task],
    )

    fetch_seriea_task = Task(
        description=(
            "Read the round_detection_task output JSON. Find the entry where league_id = 23 (Serie A).\n"
            "- If status is 'proceed': call collect_league_round(league_id=23, round_id=<next_round>). "
            "Report the full result including stdout.\n"
            "- If status is 'pending': output 'Serie A round <next_round>: not ready yet, skipping.'\n"
            "- If status is 'blocked': output 'Serie A: Sofascore blocked, skipping.'"
        ),
        expected_output=(
            "One of: collection result dict with success/stdout/stderr, "
            "or a skip message explaining the reason."
        ),
        agent=fetcher_agent,
        context=[round_detection_task],
    )

    training_task = Task(
        description=(
            "The Data Fetcher has just finished the collection phase. This is your notification "
            "to start autonomous model retraining. Do not wait for any further instruction.\n\n"
            "Step 1: Call get_training_pipeline to retrieve the ordered list of scripts.\n\n"
            "Step 2: For each script in the list (in order), call run_training_script with its "
            "script_name. Do not skip any script. If one script fails (success=False), log the "
            "error and continue with the next script — partial retraining is better than none.\n\n"
            "Step 3: After all scripts have run, report a final summary with:\n"
            "  - Total scripts run\n"
            "  - How many succeeded vs failed\n"
            "  - For each script: name, success status, and the last lines of stdout\n"
            "  - Any errors encountered\n\n"
            "Context from the collection phase (for your report only — training runs regardless):\n"
            "  LaLiga fetch result:        {fetch_laliga_result}\n"
            "  Premier League fetch result: {fetch_premier_result}\n"
            "  Serie A fetch result:        {fetch_seriea_result}"
        ),
        expected_output=(
            "Training summary report:\n"
            "  Scripts run: 5/5\n"
            "  Succeeded: N  |  Failed: M\n"
            "  [Per-script: name, success, last stdout lines]\n"
            "  [Any errors]"
        ),
        agent=training_agent,
        context=[fetch_laliga_task, fetch_premier_task, fetch_seriea_task],
    )

    # ── Crew ──────────────────────────────────────────────────────────────────

    return Crew(
        agents=[monitor_agent, round_detector_agent, fetcher_agent, training_agent],
        tasks=[
            validation_task,
            round_detection_task,
            fetch_laliga_task,
            fetch_premier_task,
            fetch_seriea_task,
            training_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    crew = build_crew()
    result = crew.kickoff()
    print("\n── Crew final output ──")
    print(result)
