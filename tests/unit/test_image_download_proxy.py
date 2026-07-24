"""图片/媒体下载经正向代理（issue #11 延伸）

图片预处理链的下载 session 由 unified_messages_preprocess 在入口建立并贯穿
多层传给各下载点。本测试用真实 SOCKS5 / HTTP 正向代理 + 真实图片服务器，
验证 `LLMClient(proxy=...)` 预处理带 URL 图片的消息时，下载确实经代理转发，
而不是只断言参数被存下来。

SOCKS 走 connector 层隧道（下载点零改动即生效）；HTTP 代理是 per-request，
靠挂在 session 上的 proxy_kwargs 在每个 session.get 处注入。两条路径都要覆盖。
"""

import base64
from io import BytesIO

from aiohttp import ClientSession, web
from PIL import Image

from flexllm import LLMClient
from flexllm.msg_processors.unified_processor import unified_messages_preprocess

from .test_socks_proxy import Socks5Server


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class PngServer:
    """返回真实 PNG 的图片服务器，记录命中次数"""

    def __init__(self):
        self._runner = None
        self.port = None
        self.hit_count = 0
        self.payload = _png_bytes()

    async def _handler(self, request):
        self.hit_count += 1
        return web.Response(body=self.payload, content_type="image/png")

    async def __aenter__(self):
        app = web.Application()
        app.router.add_get("/img.png", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()

    def url(self, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}/img.png"


class HttpForwardProxy:
    """最小 HTTP 正向代理：转发 absolute-URI 请求，记录经过的 URL

    aiohttp 的 proxy= 会以 absolute-form（GET http://host/path）发到代理，
    这里据此判定客户端确实把请求交给了代理而非直连。
    """

    def __init__(self):
        self._runner = None
        self.port = None
        self.forwarded: list[str] = []

    async def _handler(self, request):
        target = str(request.url)  # 代理侧看到的是完整 absolute URL
        self.forwarded.append(target)
        async with ClientSession() as s:
            async with s.get(target) as resp:
                body = await resp.read()
                return web.Response(body=body, content_type=resp.headers.get("Content-Type"))

    async def __aenter__(self):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _image_message(url: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]


def _extract_data_uri(processed: list[dict]) -> str:
    part = processed[0]["content"][0]
    return part["image_url"]["url"]


class TestUnifiedPreprocessProxy:
    async def test_download_via_socks5(self):
        """回归：图片下载 session 曾固定 TCPConnector，SOCKS 下无法建隧道"""
        async with PngServer() as img, Socks5Server() as proxy:
            processed = await unified_messages_preprocess(
                _image_message(img.url()), proxy=proxy.url()
            )

        assert img.hit_count == 1
        assert _extract_data_uri(processed).startswith("data:image/")
        # 目标端口出现在 SOCKS 连接记录里 → 下载确实经代理
        assert [t[1] for t in proxy.targets] == [img.port]

    async def test_download_via_http_proxy(self):
        async with PngServer() as img, HttpForwardProxy() as proxy:
            processed = await unified_messages_preprocess(
                _image_message(img.url()), proxy=proxy.url()
            )

        assert img.hit_count == 1
        assert _extract_data_uri(processed).startswith("data:image/")
        assert any(f"127.0.0.1:{img.port}" in u for u in proxy.forwarded)

    async def test_no_proxy_direct_download(self):
        """不传 proxy 时正常直连下载（向后兼容）"""
        async with PngServer() as img:
            processed = await unified_messages_preprocess(_image_message(img.url()))

        assert img.hit_count == 1
        assert _extract_data_uri(processed).startswith("data:image/")

    async def test_hostname_resolved_by_socks_proxy(self):
        """SOCKS5 默认 rdns：域名交给代理解析，VPN 场景关键"""
        async with PngServer() as img, Socks5Server() as proxy:
            await unified_messages_preprocess(
                _image_message(img.url(host="localhost")), proxy=proxy.url()
            )

        assert proxy.targets[0][0] == "localhost"


class TestLLMClientPlumbsProxyToImageDownload:
    async def test_client_proxy_reaches_image_download(self):
        """端到端：LLMClient 的 proxy 一路透传到图片下载 session"""
        async with PngServer() as img, Socks5Server() as proxy:
            client = LLMClient(
                base_url="http://unused/v1", api_key="k", model="m", proxy=proxy.url()
            )
            processed = await client._preprocess_messages(
                _image_message(img.url()), preprocess_msg=True
            )

        assert _extract_data_uri(processed).startswith("data:image/")
        assert [t[1] for t in proxy.targets] == [img.port]

    async def test_batch_path_plumbs_proxy(self):
        """批量预处理路径同样透传（proxy 随 kwargs 流到 unified_messages_preprocess）"""
        async with PngServer() as img, Socks5Server() as proxy:
            client = LLMClient(
                base_url="http://unused/v1", api_key="k", model="m", proxy=proxy.url()
            )
            processed = await client._preprocess_messages_batch(
                [_image_message(img.url())], preprocess_msg=True
            )

        assert _extract_data_uri(processed[0]).startswith("data:image/")
        assert img.port in [t[1] for t in proxy.targets]


def test_png_fixture_is_valid():
    """确保测试图片本身是合法 PNG，排除断言假阳性"""
    img = Image.open(BytesIO(_png_bytes()))
    assert img.size == (2, 2)
    uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    assert uri.startswith("data:image/png;base64,")
