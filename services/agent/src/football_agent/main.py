from football_agent.crew import FootballAgent
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

def run():
    while True:
        query = input("\n⚽ Pregunta: ").strip()
        if query.lower() in ("salir", "exit", "q"):
            print("\n¡Hasta luego! ⚽\n")
            break

        result = FootballAgent().crew().kickoff(inputs={"query": query})
        print(f"\n{result}")


if __name__ == "__main__":
    run()