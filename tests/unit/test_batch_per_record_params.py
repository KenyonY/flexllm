"""batch per-record params + system/user_template 统一语义

- convert_to_messages 对所有格式统一 system 兜底 / user_template 只套最后一条
- build_gen_params_list 剥离消息构造类键；缓存键随 per-record 参数变化
"""

from flexllm.cache.response_cache import ResponseCache, ResponseCacheConfig
from flexllm.cli.utils import convert_to_messages, detect_input_format
from flexllm.clients.batch_helpers import build_gen_params_list


def _convert(record, global_system=None, user_template=None):
    fmt, fields = detect_input_format(record)
    return convert_to_messages(record, fmt, fields, global_system, user_template)


# ========== 第一步：system 兜底 ==========


class TestSystemFallback:
    def test_openai_chat_no_inline_system_inserts_global(self):
        record = {"messages": [{"role": "user", "content": "hi"}]}
        msgs, _ = _convert(record, global_system="你是助手")
        assert msgs[0] == {"role": "system", "content": "你是助手"}
        assert msgs[1]["role"] == "user"

    def test_openai_chat_inline_system_preserved(self):
        record = {
            "messages": [
                {"role": "system", "content": "行内"},
                {"role": "user", "content": "hi"},
            ]
        }
        msgs, _ = _convert(record, global_system="全局")
        assert msgs[0]["content"] == "行内"
        assert sum(1 for m in msgs if m["role"] == "system") == 1

    def test_simple_inline_system_wins_over_global(self):
        record = {"q": "hi", "system": "行内"}
        msgs, _ = _convert(record, global_system="全局")
        assert msgs[0] == {"role": "system", "content": "行内"}
        assert sum(1 for m in msgs if m["role"] == "system") == 1

    def test_simple_no_inline_uses_global(self):
        record = {"q": "hi"}
        msgs, _ = _convert(record, global_system="全局")
        assert msgs[0] == {"role": "system", "content": "全局"}

    def test_no_system_config_unchanged(self):
        for record in (
            {"messages": [{"role": "user", "content": "hi"}]},
            {"q": "hi"},
            {"instruction": "do"},
        ):
            msgs, _ = _convert(record)
            assert all(m["role"] != "system" for m in msgs)


# ========== 第一步：user_template 只套最后一条 user ==========


class TestUserTemplate:
    def test_multiturn_only_last_user_templated(self):
        record = {
            "messages": [
                {"role": "user", "content": "第一轮"},
                {"role": "assistant", "content": "答"},
                {"role": "user", "content": "第二轮"},
            ]
        }
        msgs, _ = _convert(record, user_template="包装:{content}")
        assert msgs[0]["content"] == "第一轮"  # 历史轮原样
        assert msgs[2]["content"] == "包装:第二轮"  # 仅最后一条

    def test_multimodal_last_user_skipped(self):
        # 最后一条 user 为多模态(list)，跳过；模板套到上一条 str user
        record = {
            "messages": [
                {"role": "user", "content": "文本轮"},
                {"role": "assistant", "content": "答"},
                {"role": "user", "content": [{"type": "text", "text": "图"}]},
            ]
        }
        msgs, _ = _convert(record, user_template="X:{content}")
        assert msgs[0]["content"] == "X:文本轮"
        assert isinstance(msgs[2]["content"], list)

    def test_template_does_not_mutate_input_record(self):
        record = {"messages": [{"role": "user", "content": "hi"}]}
        _convert(record, user_template="T:{content}")
        assert record["messages"][0]["content"] == "hi"

    def test_simple_template_applied(self):
        record = {"q": "原文"}
        msgs, _ = _convert(record, user_template="译:{content}")
        assert msgs[-1]["content"] == "译:原文"


# ========== 第二步：params 字段处理 ==========


class TestParamsField:
    def test_params_not_leaked_into_metadata(self):
        record = {"q": "hi", "params": {"temperature": 0.2}, "biz_id": 7}
        _, meta = _convert(record)
        assert "params" not in meta
        assert meta == {"biz_id": 7}

    def test_build_gen_params_strips_construction_keys(self):
        params_list = [
            {"system": "s", "user_template": "t", "temperature": 0.9, "stop": ["\n"]},
            None,
            {"system": "only-system"},
        ]
        gen = build_gen_params_list(params_list)
        assert gen == [{"temperature": 0.9, "stop": ["\n"]}, None, None]

    def test_build_gen_params_none(self):
        assert build_gen_params_list(None) is None


# ========== 第二步：缓存键随 per-record 参数区分 ==========


class TestCacheKeyPerRecord:
    def test_same_messages_different_params_miss(self, tmp_path):
        cache = ResponseCache(ResponseCacheConfig(enabled=True, cache_dir=str(tmp_path / "c")))
        msgs = [[{"role": "user", "content": "hi"}]]
        # 以 temperature=0.2 存入
        cache.set(msgs[0], {"content": "A"}, model="m", temperature=0.2)
        # 不同 temperature 查询 → 不命中
        cached, uncached = cache.get_batch(msgs, model="m", params_list=[{"temperature": 0.9}])
        assert cached == [None]
        assert uncached == [0]
        # 相同 temperature 查询 → 命中
        cached2, _ = cache.get_batch(msgs, model="m", params_list=[{"temperature": 0.2}])
        assert cached2[0] == {"content": "A"}
        cache.close()
