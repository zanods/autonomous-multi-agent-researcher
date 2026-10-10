import pytest

from models.task import Task, TaskStatus


def test_happy_path_transitions():
    t = Task(title="Sample task")
    assert t.status == TaskStatus.PENDING
    t.transition(TaskStatus.IN_PROGRESS)
    t.transition(TaskStatus.RESEARCHED)
    t.transition(TaskStatus.VERIFIED)
    t.transition(TaskStatus.DONE)
    assert t.status == TaskStatus.DONE
    assert t.is_terminal()


def test_illegal_transition_raises():
    t = Task(title="Sample task")
    with pytest.raises(ValueError):
        t.transition(TaskStatus.DONE)  # PENDING -> DONE is not allowed


def test_retry_flow_respects_limit():
    t = Task(title="Sample task", max_retries=1)
    t.transition(TaskStatus.IN_PROGRESS)
    t.transition(TaskStatus.NEEDS_RETRY)
    assert t.can_retry()
    t.retries += 1
    assert not t.can_retry()
    t.transition(TaskStatus.IN_PROGRESS)
    t.transition(TaskStatus.FAILED)
    assert t.is_terminal()
