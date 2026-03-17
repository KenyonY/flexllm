"""TodoTracker 单元测试"""

from flexllm.agent.todo_tracker import TodoTracker


def test_update_and_render():
    tracker = TodoTracker()
    result = tracker.update(
        [
            {"id": 1, "text": "做任务A", "status": "pending"},
            {"id": 2, "text": "做任务B", "status": "completed"},
        ]
    )
    assert "[ ] #1: 做任务A" in result
    assert "[x] #2: 做任务B" in result
    assert "1/2 completed" in result


def test_render_empty():
    tracker = TodoTracker()
    assert tracker.render() == "(no todos)"


def test_tick_no_items():
    tracker = TodoTracker(nag_interval=2)
    assert tracker.tick() is None


def test_tick_nag():
    tracker = TodoTracker(nag_interval=2)
    tracker.update([{"id": 1, "text": "任务", "status": "pending"}])

    # 第1轮不提醒
    assert tracker.tick() is None
    # 第2轮提醒
    reminder = tracker.tick()
    assert reminder is not None
    assert "1 个未完成" in reminder


def test_tick_no_nag_when_all_completed():
    tracker = TodoTracker(nag_interval=1)
    tracker.update([{"id": 1, "text": "任务", "status": "completed"}])

    assert tracker.tick() is None


def test_notify_used_resets():
    tracker = TodoTracker(nag_interval=2)
    tracker.update([{"id": 1, "text": "任务", "status": "pending"}])

    tracker.tick()  # round 1
    tracker.notify_used()  # 重置
    assert tracker.tick() is None  # round 1 again
    assert tracker.tick() is not None  # round 2


def test_update_resets_counter():
    tracker = TodoTracker(nag_interval=2)
    tracker.update([{"id": 1, "text": "任务", "status": "pending"}])

    tracker.tick()  # round 1
    # 更新后重置
    tracker.update([{"id": 1, "text": "任务", "status": "in_progress"}])
    assert tracker.tick() is None  # round 1 again


def test_in_progress_render():
    tracker = TodoTracker()
    tracker.update([{"id": 1, "text": "正在做", "status": "in_progress"}])
    result = tracker.render()
    assert "[>] #1: 正在做" in result
