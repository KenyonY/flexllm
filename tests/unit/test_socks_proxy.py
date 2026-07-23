"""SOCKS 代理测试

SOCKS 与 HTTP 代理的实现路径不同：aiohttp 原生的 `proxy=` 参数只会发 HTTP
CONNECT，SOCKS 必须在 connector 层建隧道（aiohttp-socks 的 ProxyConnector）。
因此这里用一个真实的 SOCKS5 服务端做端到端验证，确认请求确实经代理转发，
而不是只断言参数被存下来了。
"""

import asyncio
import json
import socket

import pytest
from aiohttp import web

from flexllm import LLMClient
from flexllm.async_api.core import ConcurrentRequester, is_socks_proxy, validate_proxy


class Socks5Server:
    """最小 SOCKS5 服务端（RFC 1928 CONNECT + RFC 1929 用户名密码认证）

    记录每个连接请求的目标地址，用于断言流量确实走了代理。
    """

    def __init__(self, username: str | None = None, password: str | None = None):
        self.username = username
        self.password = password
        self.targets: list[tuple[str, int]] = []  # 客户端要求连接的目标
        self.auth_attempts: list[tuple[str, str]] = []
        self._server = None
        self.port = None

    async def _handle(self, reader, writer):
        try:
            await self._negotiate(reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if not writer.is_closing():
                writer.close()

    async def _negotiate(self, reader, writer):
        # 1) 方法协商
        _ver, nmethods = await reader.readexactly(2)
        methods = await reader.readexactly(nmethods)
        if self.username is not None:
            if 0x02 not in methods:
                writer.write(b"\x05\xff")
                await writer.drain()
                return
            writer.write(b"\x05\x02")
            await writer.drain()
            # 2) 用户名密码认证（RFC 1929）
            await reader.readexactly(1)  # 子协商版本
            ulen = (await reader.readexactly(1))[0]
            uname = (await reader.readexactly(ulen)).decode()
            plen = (await reader.readexactly(1))[0]
            passwd = (await reader.readexactly(plen)).decode()
            self.auth_attempts.append((uname, passwd))
            ok = uname == self.username and passwd == self.password
            writer.write(b"\x01" + (b"\x00" if ok else b"\x01"))
            await writer.drain()
            if not ok:
                return
        else:
            writer.write(b"\x05\x00")
            await writer.drain()

        # 3) CONNECT 请求
        _ver, cmd, _rsv, atyp = await reader.readexactly(4)
        if atyp == 0x01:
            host = socket.inet_ntoa(await reader.readexactly(4))
        elif atyp == 0x03:
            n = (await reader.readexactly(1))[0]
            host = (await reader.readexactly(n)).decode()
        else:
            host = socket.inet_ntop(socket.AF_INET6, await reader.readexactly(16))
        port = int.from_bytes(await reader.readexactly(2), "big")
        self.targets.append((host, port))

        if cmd != 0x01:
            writer.write(b"\x05\x07\x00\x01" + b"\x00" * 6)
            await writer.drain()
            return
        try:
            remote_r, remote_w = await asyncio.open_connection(host, port)
        except OSError:
            writer.write(b"\x05\x01\x00\x01" + b"\x00" * 6)
            await writer.drain()
            return
        writer.write(b"\x05\x00\x00\x01" + b"\x00" * 6)
        await writer.drain()

        # 4) 双向透传
        await asyncio.gather(
            self._pipe(reader, remote_w),
            self._pipe(remote_r, writer),
            return_exceptions=True,
        )

    @staticmethod
    async def _pipe(reader, writer):
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            if not writer.is_closing():
                writer.close()

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        self._server.close()
        await self._server.wait_closed()

    def url(self, scheme: str = "socks5") -> str:
        if self.username is not None:
            return f"{scheme}://{self.username}:{self.password}@127.0.0.1:{self.port}"
        return f"{scheme}://127.0.0.1:{self.port}"


class EchoServer:
    """返回固定 JSON 的目标服务器"""

    def __init__(self):
        self._runner = None
        self.port = None
        self.hit_count = 0

    async def _handler(self, request):
        self.hit_count += 1
        return web.json_response({"result": "ok"})

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/test", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()

    def url(self, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}/test"


class SseServer:
    """返回 OpenAI 风格 SSE 流的目标服务器"""

    def __init__(self):
        self._runner = None
        self.port = None

    async def _handler(self, request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for piece in ("Hello", " world"):
            payload = json.dumps({"choices": [{"delta": {"content": piece}}]})
            await resp.write(f"data: {payload}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()


async def _request_via(proxy: str, target_url: str):
    requester = ConcurrentRequester(concurrency_limit=2, retry_times=1, proxy=proxy, timeout=10)
    try:
        results, _ = await requester.process_requests(
            request_params=[{"json": {"q": 1}}],
            url=target_url,
            show_progress=False,
        )
        return results[0]
    finally:
        await requester.aclose()


class TestValidateSocksProxy:
    def test_socks_schemes_accepted(self):
        assert validate_proxy("socks5://gw:1080") == "socks5://gw:1080"
        assert validate_proxy("socks4://gw:1080") == "socks4://gw:1080"

    def test_socks5h_normalized_to_socks5(self):
        """socks5h 是 curl 的写法，python_socks 不认；SOCKS5 默认 rdns=True 语义等价"""
        assert validate_proxy("socks5h://gw:1080") == "socks5://gw:1080"
        assert validate_proxy("socks5h://u:p@gw:1080") == "socks5://u:p@gw:1080"

    def test_unknown_scheme_still_rejected(self):
        for bad in ("socks6://gw:1080", "ftp://gw:21", "gw:1080"):
            with pytest.raises(ValueError, match="代理 scheme"):
                validate_proxy(bad)

    def test_is_socks_proxy(self):
        assert is_socks_proxy("socks5://gw:1080")
        assert is_socks_proxy("socks5h://gw:1080")
        assert not is_socks_proxy("http://gw:8080")
        assert not is_socks_proxy(None)

    def test_missing_port_rejected_at_construction(self):
        """无端口的 SOCKS URL 应构造时报错并带代理上下文

        回归：python_socks 要求显式端口，此前 "socks5://gateway" 能通过校验，
        直到首个请求建 session 时才抛不含 'proxy' 字样的裸 ValueError，
        且从 process_requests 抛出中断整批任务。
        """
        with pytest.raises(ValueError, match="SOCKS 代理"):
            validate_proxy("socks5://gateway")

    def test_missing_dependency_reports_at_construction(self, monkeypatch):
        """未装 aiohttp-socks 时应在构造时报错并给出安装指引，而非发请求时才炸"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "aiohttp_socks":
                raise ImportError("mocked missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ValueError, match=r"flexllm\[socks\]"):
            validate_proxy("socks5://gw:1080")


class TestSocks5EndToEnd:
    async def test_request_routed_through_socks5(self):
        """回归：SOCKS 代理曾被 validate_proxy 直接拒绝"""
        async with EchoServer() as target, Socks5Server() as proxy:
            result = await _request_via(proxy.url(), target.url())

        assert result.status == "success", result.data
        assert result.data == {"result": "ok"}
        assert target.hit_count == 1
        # 目标端口出现在 SOCKS 服务端的连接记录里 → 流量确实经过代理
        assert [t[1] for t in proxy.targets] == [target.port]

    async def test_socks5_with_authentication(self):
        async with EchoServer() as target, Socks5Server(username="u", password="p") as proxy:
            result = await _request_via(proxy.url(), target.url())

        assert result.status == "success", result.data
        assert proxy.auth_attempts == [("u", "p")]

    async def test_socks5h_url_works_end_to_end(self):
        async with EchoServer() as target, Socks5Server() as proxy:
            result = await _request_via(proxy.url(scheme="socks5h"), target.url())

        assert result.status == "success", result.data
        assert len(proxy.targets) == 1

    async def test_hostname_resolved_by_proxy(self):
        """rdns：域名原样交给代理解析（socks5h 语义），而非本地解析成 IP

        VPN 网关场景的关键——目标域名往往只有网关那侧能解析。
        """
        async with EchoServer() as target, Socks5Server() as proxy:
            result = await _request_via(proxy.url(), target.url(host="localhost"))

        assert result.status == "success", result.data
        assert proxy.targets[0][0] == "localhost"

    async def test_streaming_routed_through_socks5(self):
        """流式路径不走 ConcurrentRequester，各客户端自建 session

        回归：三处流式路径（base/gemini/claude）原本固定用 TCPConnector 并传
        proxy= kwarg，SOCKS 下 aiohttp 会对着 SOCKS 端口发 HTTP CONNECT，
        表现为代理侧收到一条 ASCII 被当成 IPv6 的垃圾目标地址。
        """
        async with SseServer() as target, Socks5Server() as proxy:
            client = LLMClient(
                base_url=f"http://127.0.0.1:{target.port}/v1",
                api_key="k",
                model="m",
                proxy=proxy.url(),
            )
            chunks = [
                c
                async for c in client.chat_completions_stream(
                    messages=[{"role": "user", "content": "hi"}]
                )
            ]

        assert "".join(chunks) == "Hello world"
        assert [t[1] for t in proxy.targets] == [target.port]

    async def test_env_proxy_does_not_hijack_socks(self, monkeypatch):
        """回归：环境变量 HTTP_PROXY 曾劫持显式配置的 SOCKS 代理

        SOCKS 路径不传 per-request proxy= 参数，此前 session 仍 trust_env=True，
        aiohttp 会按 HTTP_PROXY 把连接目标改成环境代理地址，再经 SOCKS 隧道
        去连它（在对端不存在）——显式配置反被环境变量覆盖。
        """
        monkeypatch.setenv("HTTP_PROXY", "http://198.51.100.1:9999")
        monkeypatch.setenv("http_proxy", "http://198.51.100.1:9999")
        async with EchoServer() as target, Socks5Server() as proxy:
            result = await _request_via(proxy.url(), target.url())

        assert result.status == "success", result.data
        # SOCKS 端收到的 CONNECT 目标必须是真实目标，而非环境代理地址
        assert [t[1] for t in proxy.targets] == [target.port]

    async def test_env_proxy_does_not_hijack_socks_streaming(self, monkeypatch):
        """流式路径（create_proxied_session）同样不受环境代理劫持"""
        monkeypatch.setenv("HTTP_PROXY", "http://198.51.100.1:9999")
        monkeypatch.setenv("http_proxy", "http://198.51.100.1:9999")
        async with SseServer() as target, Socks5Server() as proxy:
            client = LLMClient(
                base_url=f"http://127.0.0.1:{target.port}/v1",
                api_key="k",
                model="m",
                proxy=proxy.url(),
            )
            chunks = [
                c
                async for c in client.chat_completions_stream(
                    messages=[{"role": "user", "content": "hi"}]
                )
            ]

        assert "".join(chunks) == "Hello world"
        assert [t[1] for t in proxy.targets] == [target.port]

    async def test_failure_surfaces_when_proxy_refuses(self):
        """代理不可达时应作为错误结果返回，而不是静默成功"""
        async with EchoServer() as target:
            # 未监听的端口
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                dead_port = s.getsockname()[1]
            result = await _request_via(f"socks5://127.0.0.1:{dead_port}", target.url())

        assert result.status == "error"
        assert target.hit_count == 0
