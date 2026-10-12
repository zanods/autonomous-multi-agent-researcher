"""Feature 4 — parallel tasks + task-granularity completion events."""

import threading
import time

from models.task import Task, TaskStatus

_API_KEYS = (
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)


def _no_keys(monkeypatch):
    for key in _API_KEYS:
        monkeypatch.delenv(key, raising=False)


def _goal_with_tasks(tmp_path, monkeypatch, n=3):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "parallel.db"))
    _no_keys(monkeypatch)
    from api import runner, store

    store.init_db()
    runner.clear_events()
    goal = store.create_goal("parallel test query")
    for i in range(n):
        store.create_task(goal["id"], Task(title=f"Parallel task {i}"))
    return runner, store, goal["id"]


def _slow_mocks(monkeypatch, calls, research_s=0.3, verify_s=0.2):
    """Fake research/verify that burn wall-clock time like real LLM calls."""
    from api import runner

    def fake_research(task, mock_mode=True, budget=None, focus_claims=None):
        calls["research"] += 1
        time.sleep(research_s)
        task.result = f"notes for {task.title}"
        return task

    def fake_verify(task, mock_mode=True, budget=None):
        calls["verify"] += 1
        time.sleep(verify_s)
        return True, "mock-verified", []

    monkeypatch.setattr(runner, "research_task", fake_research)
    monkeypatch.setattr(runner, "verify_task", fake_verify)


def test_run_all_tasks_runs_concurrently(tmp_path, monkeypatch):
    """3 tasks x 0.5s of fake I/O each: sequential >= 1.5s, concurrent ~0.5s."""
    runner, store, gid = _goal_with_tasks(tmp_path, monkeypatch, n=3)
    calls = {"research": 0, "verify": 0}
    _slow_mocks(monkeypatch, calls)

    start = time.monotonic()
    runner.run_all_tasks(gid, max_workers=3)
    elapsed = time.monotonic() - start

    assert elapsed < 1.1, f"took {elapsed:.2f}s — not concurrent"
    assert calls == {"research": 3, "verify": 3}
    assert all(t["status"] == "done" for t in store.list_tasks(gid))


def test_atomic_claim_survives_concurrent_double_run(tmp_path, monkeypatch):
    """Two threads racing run_task on the same task: exactly one processes it."""
    runner, store, gid = _goal_with_tasks(tmp_path, monkeypatch, n=1)
    calls = {"research": 0, "verify": 0}
    _slow_mocks(monkeypatch, calls)
    tid = store.list_tasks(gid)[0]["id"]

    threads = [threading.Thread(target=runner.run_task, args=(tid,)) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert calls["research"] == 1
    task = store.get_task(tid)
    assert task["status"] == "done"
    assert task["retries"] == 0


def test_task_completion_events_emitted(tmp_path, monkeypatch):
    runner, store, gid = _goal_with_tasks(tmp_path, monkeypatch, n=2)
    _slow_mocks(monkeypatch, {"research": 0, "verify": 0}, research_s=0.01, verify_s=0.01)

    runner.run_all_tasks(gid, max_workers=2)

    events = runner.get_events(gid)
    assert len(events) == 2
    assert {e["status"] for e in events} == {"done"}
    assert {e["task_id"] for e in events} == {t["id"] for t in store.list_tasks(gid)}
    assert all(e["goal_id"] == gid for e in events)
    # Other goals are filtered out.
    assert runner.get_events("nope") == []


def test_run_task_pipeline_retry_then_done(monkeypatch):
    """The extracted per-task pipeline preserves retry logic (max 2)."""
    import graph.workflow as wf_mod

    attempts = {"n": 0}

    def fake_research(task, mock_mode=True, budget=None, focus_claims=None):
        task.result = "some research notes here"
        return task

    def fake_verify(task, mock_mode=True, budget=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return False, "weak", ["some research notes here"]
        return True, "ok", []

    monkeypatch.setattr(wf_mod, "research_task", fake_research)
    monkeypatch.setattr(wf_mod, "verify_task", fake_verify)

    task = Task(title="Pipeline task")
    out = wf_mod.run_task_pipeline(task, mock_mode=True)
    assert out.status == TaskStatus.DONE
    assert out.retries == 1


def test_arun_tasks_concurrent(monkeypatch):
    """arun_tasks beats sequential wall-clock with slow fakes."""
    import asyncio

    import graph.workflow as wf_mod

    def fake_research(task, mock_mode=True, budget=None, focus_claims=None):
        time.sleep(0.3)
        task.result = "notes"
        return task

    def fake_verify(task, mock_mode=True, budget=None):
        return True, "ok", []

    monkeypatch.setattr(wf_mod, "research_task", fake_research)
    monkeypatch.setattr(wf_mod, "verify_task", fake_verify)

    tasks = [Task(title=f"Async task {i}") for i in range(3)]
    start = time.monotonic()
    results = asyncio.run(wf_mod.arun_tasks(tasks, mock_mode=True))
    elapsed = time.monotonic() - start

    assert elapsed < 0.8, f"took {elapsed:.2f}s — not concurrent"
    assert all(t.status == TaskStatus.DONE for t, _b in results)
    assert all(b is not None for _t, b in results)
