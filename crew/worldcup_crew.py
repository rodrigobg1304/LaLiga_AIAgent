"""
CrewAI crew for daily World Cup 2026 data collection + model retraining.

Three agents collaborate in sequence:
  1. monitor_agent   — pre-flight Sofascore API health check
  2. fetcher_agent   — collects ALL available rounds via collect_tournaments.py
  3. training_agent  — retrains World Cup / qualifier models (train_qualy.py)
                       autonomously once collection finishes

Run via main.py (scheduled daily at 09:00 Europe/Madrid) or directly:
    python worldcup_crew.py
"""
import os
from crewai import Agent, Task, Crew, Process, LLM

from tools.sofascore_health_tool import check_sofascore_health
from tools.subprocess_tool import collect_tournament_rounds
from tools.training_tool import get_worldcup_training_pipeline, run_training_script

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

    training_agent = Agent(
        role="World Cup Model Trainer",
        goal=(
            "Retrain the World Cup and qualifier ML models (1X2 + Over/Under) "
            "autonomously every day after data collection completes."
        ),
        backstory=(
            "You are an ML engineer keeping the international prediction models up to date. "
            "After the fetcher finishes, you run train_qualy.py which retrains all "
            "World Cup and qualifier models: 1X2, Over/Under Goals, Saves, and Corners. "
            "You work autonomously — no confirmation needed, no steps skipped. "
            "Even if collection found no new matches today, you still retrain so the "
            "models reflect any manual DB changes."
        ),
        tools=[get_worldcup_training_pipeline, run_training_script],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
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

    training_task = Task(
        description=(
            "The fetcher has finished. This is your signal to retrain the World Cup models.\n\n"
            "Step 1: Call get_worldcup_training_pipeline to get the list of scripts.\n\n"
            "Step 2: For each script (in order), call run_training_script with its script_name. "
            "Do not skip any script. If one fails, log the error and continue.\n\n"
            "Step 3: Report a final summary with: scripts run, succeeded vs failed, "
            "and the last lines of stdout for each script."
        ),
        expected_output=(
            "Training summary: scripts run, succeeded/failed count, "
            "per-script name + success status + last stdout lines."
        ),
        agent=training_agent,
        context=[fetch_worldcup_task],
    )

    # ── Crew ──────────────────────────────────────────────────────────────────

    return Crew(
        agents=[monitor_agent, fetcher_agent, training_agent],
        tasks=[validation_task, fetch_worldcup_task, training_task],
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
