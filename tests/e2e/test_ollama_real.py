"""真实 OpenAI 兼容端点(本地 ollama)回归测试。

目的：mock 测试只验证流程逻辑，不验证第三方 API 兼容性。本文件对真实的
ollama /v1 端点发起真实请求，覆盖本轮修复中 mock 无法验证的传输层兼容性：
请求体构造、真实 SSE 流解析、真实 usage、批量顺序、fallback、并发丢任务、
以及图片编码端到端（调色板模式）。

ollama 未运行时整个模块自动跳过（不阻塞 CI）。本地运行：
    pytest tests/e2e/test_ollama_real.py -v

需要本地 ollama（http://localhost:11434），并至少拉取过 qwen2.5（文本用例）/
qwen2.5vl（视觉用例）。缺失的模型对应用例会各自跳过。
"""

import asyncio
import base64
import io

import pytest
from PIL import Image

from flexllm import EndpointConfig, LLMClient, LLMClientPool, MllmClient

OLLAMA_URL = "http://localhost:11434/v1"


def _ollama_models() -> set[str]:
    """返回 ollama 已安装模型名集合；连不上返回空集。"""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            import json

            data = json.load(r)
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


_MODELS = _ollama_models()

pytestmark = pytest.mark.skipif(not _MODELS, reason="本地 ollama(11434) 不可用，跳过真实端点测试")


def _pick(*candidates: str) -> str | None:
    """从已安装模型里挑第一个匹配的名字。

    按 base name（冒号前部分）精确匹配而非 startswith：候选 "qwen3" 不应
    撞上 "qwen3-vl"。sorted 保证同一 base 多个 tag 时结果确定（set 迭代
    顺序受哈希随机化影响）。
    """
    for c in candidates:
        base = c.split(":")[0]
        for m in sorted(_MODELS):
            if m == c or m.split(":")[0] == base:
                return m
    return None


TEXT_MODEL = _pick("qwen2.5:latest", "qwen2:latest", "phi4:latest", "gemma3:latest")
VISION_MODEL = _pick("qwen2.5vl:latest", "qwen3-vl:8b", "minicpm-v:latest")

# ollama 在线但没装任何文本候选模型时，用例应 skip 而非以 model=None 全线报错
_requires_text = pytest.mark.skipif(
    TEXT_MODEL is None, reason="ollama 已运行但无可用文本模型（qwen2.5/qwen2/phi4/gemma3）"
)


def _client(model: str) -> LLMClient:
    return LLMClient(base_url=OLLAMA_URL, model=model, api_key="ollama")


# ---------- 基础传输层 ----------


@_requires_text
class TestRealTransport:
    def test_sync_chat(self):
        """真实同步调用返回非空 str。"""
        r = _client(TEXT_MODEL).chat_completions_sync(
            [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=10
        )
        assert isinstance(r, str) and r.strip()

    async def test_async_chat(self):
        async with _client(TEXT_MODEL) as c:
            r = await c.chat_completions(
                [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=10
            )
        assert isinstance(r, str) and r.strip()

    async def test_streaming_real_sse(self):
        """真实 SSE 流：多个 chunk 且拼接非空（验证流式解析对真实分片的兼容）。"""
        chunks = []
        async with _client(TEXT_MODEL) as c:
            async for ch in c.chat_completions_stream(
                [{"role": "user", "content": "Count from 1 to 5."}], max_tokens=40
            ):
                chunks.append(ch)
        assert len(chunks) >= 2, "真实流应产生多个增量 chunk"
        assert "".join(chunks).strip()

    async def test_return_usage_real_tokens(self):
        """真实 usage：prompt_tokens 与 completion_tokens 均非零（回归 #9 归零类问题）。"""
        async with _client(TEXT_MODEL) as c:
            r = await c.chat_completions(
                [{"role": "user", "content": "hello"}], return_usage=True, max_tokens=10
            )
        assert r.usage["prompt_tokens"] > 0
        assert r.usage["completion_tokens"] > 0
        assert r.usage["total_tokens"] >= r.usage["prompt_tokens"]


# ---------- 批量 / return_raw ----------


@_requires_text
class TestRealBatch:
    async def test_batch_count_and_order_no_loss(self):
        """真实批量：结果数与输入一致、无 None（回归 worker/断点续传丢结果类问题）。"""
        msgs = [[{"role": "user", "content": f"Reply with only the number {i}"}] for i in range(8)]
        async with _client(TEXT_MODEL) as c:
            res = await c.chat_completions_batch(msgs, show_progress=False, max_tokens=10)
        assert len(res) == 8
        assert all(r is not None for r in res), "真实批量不应丢任何结果"
        assert all(isinstance(r, str) for r in res)

    async def test_batch_return_raw_is_dict(self):
        """return_raw=True 返回后端原始 dict 列表（回归 #10：此前是死参数返回 str）。"""
        msgs = [[{"role": "user", "content": "hi"}] for _ in range(3)]
        async with _client(TEXT_MODEL) as c:
            res = await c.chat_completions_batch(
                msgs, show_progress=False, return_raw=True, max_tokens=10
            )
        assert len(res) == 3
        for r in res:
            assert isinstance(r, dict) and "choices" in r


# ---------- 成本追踪（真实 usage）----------


@_requires_text
class TestRealCostTracking:
    async def test_unknown_model_zero_cost_no_budget_trip(self):
        """未知(自建)模型真实调用：成本为 0，配了极小预算也不熔断（回归 #3 假计费）。"""
        from flexllm import CostTrackerConfig

        # 极小预算：若按 gpt-4o-mini 假计费，多次调用会触发 BudgetExceededError
        client = LLMClient(
            base_url=OLLAMA_URL,
            model=TEXT_MODEL,
            api_key="ollama",
            cost_tracker=CostTrackerConfig.with_budget(0.001),
        )
        async with client as c:
            msgs = [[{"role": "user", "content": f"count {i}"}] for i in range(5)]
            res = await c.chat_completions_batch(
                msgs, show_progress=False, track_cost=True, max_tokens=20
            )
        assert all(r is not None for r in res), "自建模型不应因假账单被熔断"


# ---------- 多 endpoint fallback ----------


@_requires_text
class TestRealFallback:
    async def test_dead_then_live_endpoint_all_complete(self):
        """死端点在前 + 活 ollama 在后：fallback 后全部完成、无丢失（真实验证 #8 路径）。"""
        eps = [
            EndpointConfig(base_url="http://localhost:9/v1", api_key="x", model="dead"),
            EndpointConfig(base_url=OLLAMA_URL, api_key="ollama", model=TEXT_MODEL),
        ]
        async with LLMClientPool(endpoints=eps, fallback=True) as pool:
            msgs = [[{"role": "user", "content": f"say {i}"}] for i in range(10)]
            res = await pool.chat_completions_batch(msgs, show_progress=False, max_tokens=10)
        assert len(res) == 10
        assert all(r is not None for r in res), "fallback 后不应丢任务"


# ---------- 并发压力（worker 竞态 #8）----------


@_requires_text
class TestRealConcurrencyStress:
    async def test_high_concurrency_fallback_no_lost_tasks(self):
        """高并发 + 死端点在前触发大量 fallback：验证 worker 原子 claim 不丢任务。

        这是 #8 竞态最直接的真实压力：末个任务被取走但未计数的窗口若存在，
        fallback 重入队会丢；这里重复多轮，任一结果为 None 即判失败。
        """
        eps = [
            EndpointConfig(base_url="http://localhost:9/v1", api_key="x", model="dead"),
            EndpointConfig(base_url=OLLAMA_URL, api_key="ollama", model=TEXT_MODEL),
        ]
        for _ in range(3):
            async with LLMClientPool(endpoints=eps, fallback=True) as pool:
                msgs = [[{"role": "user", "content": f"n={i}"}] for i in range(20)]
                res = await pool.chat_completions_batch(msgs, show_progress=False, max_tokens=8)
            assert len(res) == 20
            assert all(r is not None for r in res), "高并发 fallback 出现丢任务"

    async def test_concurrent_mixed_clients_independent(self):
        """并发跑两个独立 client（一个高并发批量、一个流式）互不干扰。"""

        async def batch_job():
            async with _client(TEXT_MODEL) as c:
                msgs = [[{"role": "user", "content": f"x{i}"}] for i in range(10)]
                return await c.chat_completions_batch(msgs, show_progress=False, max_tokens=8)

        async def stream_job():
            out = []
            async with _client(TEXT_MODEL) as c:
                async for ch in c.chat_completions_stream(
                    [{"role": "user", "content": "hi"}], max_tokens=15
                ):
                    out.append(ch)
            return out

        batch_res, stream_res = await asyncio.gather(batch_job(), stream_job())
        assert len(batch_res) == 10 and all(r is not None for r in batch_res)
        assert "".join(stream_res).strip()


# ---------- 图片处理端到端（调色板修复 D2）----------


class TestRealVision:
    @pytest.mark.skipif(not VISION_MODEL, reason="无可用视觉模型")
    async def test_palette_image_encoded_correctly(self):
        """调色板(P)模式红图端到端发给视觉模型，能被识别为红色。

        若 D2 调色板 bug 存在，np.array(P图) 得到的是索引而非 RGB，编码后颜色
        损坏，模型无法答出红色。这是对图片编码正确性的端到端真实验证。
        """
        img = Image.new("P", (64, 64))
        palette = [0, 0, 0] * 256
        palette[0], palette[1], palette[2] = 255, 0, 0  # 索引 0 = 纯红
        img.putpalette(palette)  # 整图默认填索引 0
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color dominates this image? Answer in one word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ]
        ]
        async with MllmClient(base_url=OLLAMA_URL, model=VISION_MODEL, api_key="ollama") as c:
            res = await c.call_llm(msgs, max_tokens=20, show_progress=False)
        assert res[0] is not None
        assert "red" in res[0].lower(), f"调色板图颜色损坏，模型未识别为红色：{res[0]!r}"
