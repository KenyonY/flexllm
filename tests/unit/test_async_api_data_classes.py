"""
async_api 模块数据类测试

测试 interface.py 中的数据类：
- RequestResult
"""

from flexllm.async_api.interface import RequestResult


class TestRequestResult:
    """RequestResult 数据类测试"""

    def test_creation_with_required_fields(self):
        """测试必填字段创建"""
        result = RequestResult(
            request_id=1,
            data={"message": "hello"},
            status="success",
            latency=0.5,
        )

        assert result.request_id == 1
        assert result.data == {"message": "hello"}
        assert result.status == "success"
        assert result.latency == 0.5
        assert result.meta is None  # 默认值

    def test_creation_with_meta(self):
        """测试带 meta 字段创建"""
        result = RequestResult(
            request_id=2,
            data="response",
            status="error",
            latency=1.2,
            meta={"retry": 1, "source": "api"},
        )

        assert result.request_id == 2
        assert result.meta == {"retry": 1, "source": "api"}

    def test_data_can_be_any_type(self):
        """测试 data 字段可以是任意类型"""
        # dict
        r1 = RequestResult(request_id=1, data={"key": "value"}, status="success", latency=0.1)
        assert r1.data == {"key": "value"}

        # list
        r2 = RequestResult(request_id=2, data=[1, 2, 3], status="success", latency=0.1)
        assert r2.data == [1, 2, 3]

        # None
        r3 = RequestResult(request_id=3, data=None, status="error", latency=0.1)
        assert r3.data is None

        # string
        r4 = RequestResult(request_id=4, data="plain text", status="success", latency=0.1)
        assert r4.data == "plain text"
