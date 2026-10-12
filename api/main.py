from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # so TAVILY_API_KEY etc. can live in a local .env file

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.planner import plan_tasks
from api import runner, store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Autonomous Multi-Agent Researcher", lifespan=lifespan)


class GoalCreate(BaseModel):
    query: str = Field(min_length=3, max_length=500)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/goals")
def create_goal(payload: GoalCreate):
    store.init_db()
    goal = store.create_goal(payload.query)
    tasks = plan_tasks(payload.query, mock_mode=runner._mock_mode())
    for t in tasks:
        store.create_task(goal["id"], t)
    return {
        "id": goal["id"],
        "query": payload.query,
        "tasks": [t.model_dump(mode="json") for t in tasks],
    }


@app.get("/goals/{goal_id}")
def get_goal(goal_id: str):
    store.init_db()
    goal = store.get_goal(goal_id)
    if goal is None:
        raise HTTPException(404, "goal not found")
    goal["tasks"] = store.list_tasks(goal_id)
    return goal


@app.get("/goals/{goal_id}/report")
def get_report(goal_id: str):
    store.init_db()
    if store.get_goal(goal_id) is None:
        raise HTTPException(404, "goal not found")
    try:
        report = runner.build_report(goal_id)
    except KeyError:
        raise HTTPException(404, "goal not found")
    return {"goal_id": goal_id, "report": report}


@app.get("/tasks")
def list_tasks(goal_id: str | None = None):
    store.init_db()
    return store.list_tasks(goal_id)


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    store.init_db()
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(404, "task not found")
    return task


@app.post("/tasks/{task_id}/run")
def run_task(task_id: str, background: BackgroundTasks):
    store.init_db()
    if store.get_task(task_id) is None:
        raise HTTPException(404, "task not found")
    background.add_task(runner.run_task, task_id)
    return {"task_id": task_id, "status": "accepted"}


@app.post("/goals/{goal_id}/run")
def run_goal(goal_id: str, background: BackgroundTasks):
    """Run every pending task of a goal concurrently. Double-queueing is safe:
    runner.run_all_tasks() claims each task with one atomic UPDATE, so an
    already-running or finished task is skipped by the loser of the race."""
    store.init_db()
    if store.get_goal(goal_id) is None:
        raise HTTPException(404, "goal not found")
    queued = [
        t["id"]
        for t in store.list_tasks(goal_id)
        if t["status"] in ("pending", "needs_retry")
    ]
    background.add_task(runner.run_all_tasks, goal_id)
    return {"goal_id": goal_id, "queued": queued}


@app.get("/goals/{goal_id}/events")
def task_events(goal_id: str):
    """Per-task completion events (task granularity) for the dashboard to poll."""
    store.init_db()
    if store.get_goal(goal_id) is None:
        raise HTTPException(404, "goal not found")
    return runner.get_events(goal_id)
