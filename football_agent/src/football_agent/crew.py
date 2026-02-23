from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from football_agent.tools.tools import team_results_tool, goals_tool, stat_tool, standings_tool


@CrewBase
class FootballAgent:
    """Football Analytics Crew"""

    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def football_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["football_analyst"],
            tools=[team_results_tool, goals_tool, stat_tool, standings_tool],
            verbose=True,
            allow_delegation=False
        )

    @task
    def football_query_task(self) -> Task:
        return Task(
            config=self.tasks_config["football_query_task"]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )