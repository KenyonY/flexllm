"""pricing/updater.py 回归测试：原子写 与 最低价合并（含 output）。

这两处修复此前零测试覆盖：
- #A4 save_pricing_data 用临时文件 + os.replace 原子写，避免多进程写坏文件
- #10 同名模型多版本保留最低价时，input 相等要继续比 output（老代码只比 input）
"""

import json

from flexllm.pricing import updater
from flexllm.pricing.updater import collect_pricing, save_pricing_data


class TestSavePricingAtomic:
    def test_writes_valid_json_and_no_tmp_leftover(self, tmp_path):
        target = tmp_path / "pricing_data.json"
        ok = save_pricing_data({"m1": {"input": 1.0, "output": 2.0}}, target)
        assert ok is True
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["models"]["m1"] == {"input": 1.0, "output": 2.0}
        # 原子写不应留下 .tmp 中间文件
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == [], f"残留临时文件: {leftovers}"

    def test_overwrite_is_atomic_replace(self, tmp_path):
        target = tmp_path / "pricing_data.json"
        save_pricing_data({"a": {"input": 1.0, "output": 1.0}}, target)
        # 覆盖写：结果应是完整的新内容，不是交错/截断
        save_pricing_data({"b": {"input": 3.0, "output": 4.0}}, target)
        data = json.loads(target.read_text(encoding="utf-8"))
        assert "a" not in data["models"]
        assert data["models"]["b"] == {"input": 3.0, "output": 4.0}


class TestCollectPricingLowest:
    def _model(self, mid, inp, out):
        # OpenRouter 风格：pricing 为 $/token 字符串，parse_pricing 会 *1e6
        return {"id": mid, "pricing": {"prompt": str(inp / 1e6), "completion": str(out / 1e6)}}

    def test_same_input_keeps_lower_output(self, monkeypatch):
        """input 相同、output 不同：应保留 output 更低者（回归 #10，老代码只比 input 会漏）。"""
        models = [
            self._model("vendor-a/gpt-x", 1.0, 5.0),  # 先出现，output 高
            self._model("vendor-b/gpt-x", 1.0, 2.0),  # 同 input，output 低 → 应胜出
        ]
        monkeypatch.setattr(updater, "fetch_models", lambda: models)
        pricing = collect_pricing()
        # normalize_model_id 去掉 vendor 前缀，两者归一为同名 gpt-x
        assert "gpt-x" in pricing
        assert pricing["gpt-x"]["output"] == 2.0, "input 相等时应保留更低的 output"

    def test_lower_input_wins(self, monkeypatch):
        models = [
            self._model("a/m", 3.0, 1.0),
            self._model("b/m", 1.0, 9.0),  # input 更低 → 胜出，即便 output 更高
        ]
        monkeypatch.setattr(updater, "fetch_models", lambda: models)
        pricing = collect_pricing()
        assert pricing["m"]["input"] == 1.0
