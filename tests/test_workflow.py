from graph.workflow import build_workflow
from models.task import TaskStatus


def test_end_to_end_mock_run():
    app = build_workflow()
    result = app.invoke(
        {
            "query": "Impact of high-TDP laptop cooling on sustained clock speeds",
            "tasks": [],
            "findings": [],
            "run_log": [],
            "current_task_id": None,
            "report": None,
            "mock_mode": True,
        }
    )
    assert result["report"] is not None
    assert len(result["tasks"]) == 3
    assert all(t.status == TaskStatus.DONE for t in result["tasks"])
    assert any("[planner]" in line for line in result["run_log"])
    assert any("[synthesizer]" in line for line in result["run_log"])
