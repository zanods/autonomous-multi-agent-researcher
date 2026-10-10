import time

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from api.main import app  # noqa: E402  (import loads .env as a side effect)
    # Delete AFTER the import: load_dotenv() re-populates keys otherwise.
    for key in (
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    return TestClient(app)


def test_goal_lifecycle(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/goals", json={"query": "Test query about laptop cooling"})
        assert r.status_code == 200
        body = r.json()
        goal_id = body["id"]
        assert len(body["tasks"]) == 3

        r = client.get("/tasks", params={"goal_id": goal_id})
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 3
        assert all(t["status"] == "pending" for t in tasks)
        tid = tasks[0]["id"]

        r = client.post(f"/tasks/{tid}/run")
        assert r.status_code == 200
        assert r.json()["status"] == "accepted"

        # background job runs the full research + fact-check flow
        status = None
        deadline = time.time() + 15
        while time.time() < deadline:
            status = client.get(f"/tasks/{tid}").json()["status"]
            if status == "done":
                break
            time.sleep(0.1)
        assert status == "done"

        r = client.get(f"/goals/{goal_id}/report")
        assert r.status_code == 200
        assert "Test query about laptop cooling" in r.json()["report"]


def test_run_missing_task_404(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        assert client.post("/tasks/nope/run").status_code == 404
        assert client.get("/tasks/nope").status_code == 404
        assert client.get("/goals/nope/report").status_code == 404


def test_goal_create_validates_query(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        assert client.post("/goals", json={"query": "ab"}).status_code == 422
