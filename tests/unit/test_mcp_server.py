"""Serve 模块单元测试"""


class TestServeConfig:
    def test_default_config(self):
        from flexllm.serve import ServeConfig

        config = ServeConfig()
        assert config.port == 8000
        assert config.host == "0.0.0.0"

    def test_create_app_routes(self):
        from flexllm.serve import ServeConfig, ServeServer

        config = ServeConfig()
        server = ServeServer(config)
        app = server._create_app()

        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]
        assert "/api/generate" in routes
        assert "/health" in routes
        assert "/api/config" in routes
