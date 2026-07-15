"""正向代理（issue #11）测试

仅支持 http(s):// —— aiohttp 不支持 SOCKS，且不校验 scheme：给它 socks5:// 它会
照样往该端口发 HTTP CONNECT，在 SOCKS 服务端上表现为难以定位的连接错误。因此在
构造时就拒绝，而不是留到运行时。
"""

import pytest

from flexllm import LLMClient
from flexllm.async_api.core import ConcurrentRequester, validate_proxy


class TestValidateProxy:
    def test_accepts_http_and_https(self):
        assert validate_proxy("http://gw:8080") == "http://gw:8080"
        assert validate_proxy("https://gw:8443") == "https://gw:8443"

    def test_accepts_credentials_in_url(self):
        url = "http://user:pass@gw:8080"
        assert validate_proxy(url) == url

    def test_none_passes_through(self):
        assert validate_proxy(None) is None

    @pytest.mark.parametrize("bad", ["socks5://gw:1080", "socks4://gw:1080", "socks5h://gw:1080"])
    def test_rejects_socks(self, bad):
        with pytest.raises(ValueError, match="不支持的代理 scheme"):
            validate_proxy(bad)

    def test_rejects_missing_scheme(self):
        with pytest.raises(ValueError, match="不支持的代理 scheme"):
            validate_proxy("gw:8080")


class TestProxyInjection:
    """proxy 注入到 aiohttp 请求"""

    class _FakeResponse:
        status = 200

        def raise_for_status(self):
            pass

        async def json(self):
            return {"ok": True}

    class _FakeCtx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        def __init__(self, outer):
            self.calls = []
            self._outer = outer

        def request(self, method, url, **kwargs):
            self.calls.append(kwargs)
            return TestProxyInjection._FakeCtx(TestProxyInjection._FakeResponse())

    async def test_client_proxy_injected(self):
        req = ConcurrentRequester(concurrency_limit=1, proxy="http://gw:8080", retry_times=1)
        session = self._FakeSession(self)
        await req.make_requests(session, "POST", "http://target/v1")

        assert session.calls[0]["proxy"] == "http://gw:8080"

    async def test_per_request_proxy_wins(self):
        # setdefault 语义：显式传入的 proxy 优先于客户端级配置
        req = ConcurrentRequester(concurrency_limit=1, proxy="http://client-gw:8080", retry_times=1)
        session = self._FakeSession(self)
        await req.make_requests(session, "POST", "http://target/v1", proxy="http://req-gw:9090")

        assert session.calls[0]["proxy"] == "http://req-gw:9090"

    async def test_no_proxy_means_no_kwarg(self):
        # 不传 proxy 时不得注入，否则会覆盖 trust_env 的环境变量行为
        req = ConcurrentRequester(concurrency_limit=1, retry_times=1)
        session = self._FakeSession(self)
        await req.make_requests(session, "POST", "http://target/v1")

        assert "proxy" not in session.calls[0]


class TestProxyPlumbing:
    def test_single_endpoint(self):
        c = LLMClient(base_url="http://x/v1", api_key="k", model="m", proxy="http://gw:8080")
        assert c._single_client._proxy == "http://gw:8080"
        assert c._single_client._client._proxy == "http://gw:8080"

    def test_endpoint_proxy_overrides_pool_default(self):
        c = LLMClient(
            endpoints=[
                {"base_url": "http://inherits/v1"},
                {"base_url": "http://overrides/v1", "proxy": "http://gw2:9090"},
            ],
            proxy="http://gw:8080",
        )
        got = {ep.base_url: cl._client._proxy for ep, cl in zip(c._endpoints, c._clients)}
        assert got == {
            "http://inherits/v1": "http://gw:8080",
            "http://overrides/v1": "http://gw2:9090",
        }

    def test_mixed_direct_and_proxied_endpoints(self):
        # 无顶层 proxy：未配置的 endpoint 直连，这是环境变量做不到的
        c = LLMClient(
            endpoints=[
                {"base_url": "http://direct/v1"},
                {"base_url": "http://vpn-only/v1", "proxy": "http://gw:8080"},
            ]
        )
        got = {ep.base_url: cl._client._proxy for ep, cl in zip(c._endpoints, c._clients)}
        assert got == {"http://direct/v1": None, "http://vpn-only/v1": "http://gw:8080"}

    def test_invalid_proxy_fails_at_construction(self):
        with pytest.raises(ValueError, match="不支持的代理 scheme"):
            LLMClient(base_url="http://x/v1", api_key="k", proxy="socks5://gw:1080")


class TestProxyFromConfig:
    """config.yaml 里的 per-model proxy

    CLI 走 from_config：没有这条，命令行用户就只能靠进程级环境变量，
    而"仅此 endpoint 需经网关"正是环境变量表达不了的。
    """

    @pytest.fixture
    def config_file(self, tmp_path, monkeypatch):
        # 环境变量会覆盖 config，测试里必须清掉
        for var in ("FLEXLLM_BASE_URL", "OPENAI_BASE_URL", "FLEXLLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        path = tmp_path / "config.yaml"
        path.write_text(
            "default: viavpn\n"
            "models:\n"
            "  - name: viavpn\n"
            "    id: m-vpn\n"
            "    base_url: http://vpn-only:8000/v1\n"
            "    api_key: sk-x\n"
            "    proxy: http://gateway:8080\n"
            "  - name: direct\n"
            "    id: m-direct\n"
            "    base_url: http://local:8000/v1\n"
            "    api_key: sk-y\n",
            encoding="utf-8",
        )
        return str(path)

    def test_proxy_from_config(self, config_file):
        c = LLMClient.from_config(config_file, model="viavpn")
        assert c._single_client._client._proxy == "http://gateway:8080"

    def test_model_without_proxy_goes_direct(self, config_file):
        c = LLMClient.from_config(config_file, model="direct")
        assert c._single_client._client._proxy is None

    def test_override_beats_config(self, config_file):
        c = LLMClient.from_config(config_file, model="viavpn", proxy="http://other:1234")
        assert c._single_client._client._proxy == "http://other:1234"
