from fastapi import FastAPI
from pydantic import BaseModel
import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv(".env")

from football_agent.crew import FootballAgent

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/agent")
def query_agent(req: QueryRequest):
    from football_agent.crew import FootballAgent
    result = FootballAgent().crew().kickoff(inputs={"query": req.query})
    return {"response": str(result)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)