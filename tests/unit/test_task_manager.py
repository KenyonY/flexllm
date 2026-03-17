"""TaskManager 单元测试"""

import json

import pytest

from flexllm.agent.task_manager import TaskManager


@pytest.fixture
def tm(tmp_path):
    return TaskManager(tasks_dir=tmp_path / "tasks")


def test_create(tm):
    result = json.loads(tm.create("任务1", "描述1"))
    assert result["id"] == 1
    assert result["subject"] == "任务1"
    assert result["status"] == "pending"

    result2 = json.loads(tm.create("任务2"))
    assert result2["id"] == 2


def test_get(tm):
    tm.create("任务1")
    result = json.loads(tm.get(1))
    assert result["subject"] == "任务1"

    # 不存在的任务
    result = json.loads(tm.get(999))
    assert "error" in result


def test_update_status(tm):
    tm.create("任务1")
    result = json.loads(tm.update(1, status="in_progress"))
    assert result["status"] == "in_progress"

    result = json.loads(tm.update(1, status="completed"))
    assert result["status"] == "completed"


def test_update_not_found(tm):
    result = json.loads(tm.update(999, status="completed"))
    assert "error" in result


def test_dependencies(tm):
    tm.create("任务1")
    tm.create("任务2")

    # 任务2 依赖任务1
    tm.update(2, add_blocked_by=[1])

    t1 = json.loads(tm.get(1))
    t2 = json.loads(tm.get(2))
    assert 2 in t1["blocks"]
    assert 1 in t2["blockedBy"]


def test_complete_clears_dependencies(tm):
    tm.create("任务1")
    tm.create("任务2")
    tm.update(2, add_blocked_by=[1])

    # 完成任务1，应自动从任务2的 blockedBy 中移除
    tm.update(1, status="completed")
    t2 = json.loads(tm.get(2))
    assert 1 not in t2["blockedBy"]


def test_add_blocks(tm):
    tm.create("任务1")
    tm.create("任务2")

    # 任务1 阻塞任务2
    tm.update(1, add_blocks=[2])

    t1 = json.loads(tm.get(1))
    t2 = json.loads(tm.get(2))
    assert 2 in t1["blocks"]
    assert 1 in t2["blockedBy"]


def test_list_all(tm):
    tm.create("任务1")
    tm.create("任务2")
    tm.update(1, status="completed")

    result = tm.list_all()
    assert "#1" in result
    assert "#2" in result
    assert "[x]" in result
    assert "[ ]" in result
    assert "1/2 completed" in result


def test_list_empty(tm):
    assert tm.list_all() == "没有任务"


def test_delete(tm):
    tm.create("任务1")
    tm.create("任务2")
    tm.update(2, add_blocked_by=[1])

    # 删除任务1，应清理任务2的 blockedBy
    result = json.loads(tm.delete(1))
    assert result["deleted"] == 1

    t2 = json.loads(tm.get(2))
    assert 1 not in t2["blockedBy"]

    # 删除不存在的任务
    result = json.loads(tm.delete(999))
    assert "error" in result


def test_recover_next_id(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tm1 = TaskManager(tasks_dir=tasks_dir)
    tm1.create("任务1")
    tm1.create("任务2")

    # 重新加载，应恢复 next_id
    tm2 = TaskManager(tasks_dir=tasks_dir)
    result = json.loads(tm2.create("任务3"))
    assert result["id"] == 3
