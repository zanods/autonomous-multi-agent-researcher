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


def _no_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for key in (
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_claim_task_is_atomic(tmp_path, monkeypatch):
    _no_keys(monkeypatch, tmp_path)
    from api import store
    from models.task import Task

    store.init_db()
    goal = store.create_goal("atomic query")
    task = Task(title="Atomic claim task")
    store.create_task(goal["id"], task)

    first = store.claim_task(task.id)
    assert first is not None and first["status"] == "in_progress"
    # second concurrent claim must fail — no double execution
    assert store.claim_task(task.id) is None
    assert store.claim_task("nonexistent") is None


def test_unverified_claims_roundtrip(tmp_path, monkeypatch):
    _no_keys(monkeypatch, tmp_path)
    from api import store
    from models.task import Task, TaskStatus

    store.init_db()
    goal = store.create_goal("claims query")
    task = Task(
        title="Claims task",
        status=TaskStatus.NEEDS_RETRY,
        unverified_claims=["claim A", "claim B"],
        tokens_in=100,
        tokens_out=50,
    )
    store.create_task(goal["id"], task)
    row = store.get_task(task.id)
    assert row["unverified_claims"] == ["claim A", "claim B"]
    assert row["tokens_in"] == 100 and row["tokens_out"] == 50


def test_runner_retry_uses_focus_claims(tmp_path, monkeypatch):
    _no_keys(monkeypatch, tmp_path)
    from api import runner, store
    from models.task import Task, TaskStatus

    store.init_db()
    goal = store.create_goal("retry query")
    task = Task(
        title="Retry focus task",
        status=TaskStatus.NEEDS_RETRY,
        unverified_claims=["unverified claim 1"],
    )
    store.create_task(goal["id"], task)

    captured = {}

    def fake_research(task, mock_mode=True, budget=None, focus_claims=None):
        captured["focus_claims"] = focus_claims
        task.result = "mock result"
        return task

    monkeypatch.setattr(runner, "research_task", fake_research)
    runner.run_task(task.id)

    # retry must narrow research to the failed claims (the Sprint 2 fix,
    # previously missing from the API path)
    assert captured["focus_claims"] == ["unverified claim 1"]
    row = store.get_task(task.id)
    assert row["status"] == "done"  # mock verify passes
    assert row["unverified_claims"] == []
    assert "tokens_in" in row and "tokens_out" in row


def test_run_all_endpoint(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/goals", json={"query": "Test query about laptop cooling"})
        goal_id = r.json()["id"]

        r = client.post(f"/goals/{goal_id}/run")
        assert r.status_code == 200
        assert len(r.json()["queued"]) == 3

        deadline = time.time() + 25
        tasks = []
        while time.time() < deadline:
            tasks = client.get("/tasks", params={"goal_id": goal_id}).json()
            if tasks and all(t["status"] == "done" for t in tasks):
                break
            time.sleep(0.2)
        assert tasks and all(t["status"] == "done" for t in tasks)

        # second call queues nothing — all tasks terminal
        r = client.post(f"/goals/{goal_id}/run")
        assert r.json()["queued"] == []

        assert client.post("/goals/nope/run").status_code == 404


def test_synthesizer_live_labels_failed_tasks(monkeypatch):
    from agents import synthesizer
    from models.task import Task, TaskStatus

    class FakeLLM:
        def invoke(self, prompt):
            assert "Good task" in prompt  # only verified tasks are synthesized
            assert "Bad task" not in prompt
            return type("Resp", (), {"content": "Synthesized report."})()

    monkeypatch.setattr(synthesizer, "get_llm", lambda: FakeLLM())
    done_task = Task(title="Good task", result="verified stuff",
                     status=TaskStatus.DONE)
    failed_task = Task(title="Bad task", status=TaskStatus.FAILED)
    report = synthesizer.synthesize("q", [done_task, failed_task], mock_mode=False)
    assert "Verification gaps" in report
    assert "Bad task" in report
