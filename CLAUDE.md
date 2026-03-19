# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

flexllm 是一个高性能 LLM 客户端库，支持批量处理、响应缓存和断点续传。支持 OpenAI 兼容 API（vLLM、Ollama、DeepSeek 等）、Google Gemini API 和 Anthropic Claude API。

## 常用命令

```bash
# 安装依赖
pip install -e ".[all]"      # 安装所有功能
pip install -e ".[dev]"      # 开发环境

# 运行测试
pytest                        # 运行所有测试
pytest tests/unit/           # 只运行单元测试（pre-push hook 执行的）
pytest tests/test_xxx.py     # 运行单个测试文件
pytest -k "test_name"        # 按名称匹配运行测试
pytest -m "not slow"         # 跳过慢测试

# 代码格式化（使用 ruff）
ruff check --fix flexllm tests   # lint + 自动修复
ruff format flexllm tests        # 格式化

# CLI 使用（别名 xllm）
flexllm ask "问题"                    # 快速问答
flexllm chat                          # 交互式聊天
flexllm batch in.jsonl -o out.jsonl   # 批量处理（支持断点续传）
flexllm list                          # 列出配置模型
flexllm set-model gpt-4               # 设置默认模型
flexllm test                          # 测试连接
flexllm init                          # 初始化配置文件
flexllm mock                          # 启动 Mock LLM 服务器（测试用）
flexllm mock --qa qa.jsonl            # Mock 服务使用 QA 数据集确定性回复
flexllm pricing gpt-4                 # 查询模型定价
flexllm credits                       # 查询 API Key 余额

# HTTP API 服务器（微调模型部署）
flexllm serve -m model-name -s "System prompt" -p 8000
```

## 核心架构

```
LLMClient = LLMClientPool (统一入口，单/多 endpoint 均使用此类)
    │
    ├── 单 endpoint 模式：自动创建底层客户端
    │   ├── OpenAIClient (OpenAI 兼容 API: vLLM/Ollama/DeepSeek 等)
    │   ├── GeminiClient (Google Gemini API)
    │   └── ClaudeClient (Anthropic Claude API)
    │
    ├── 多 endpoint 模式：负载均衡
    │   └── ProviderRouter (round_robin)
    │
    └── 所有客户端继承自 LLMClientBase (抽象基类)
            ├── ConcurrentRequester (async_api/ 异步并发引擎)
            ├── ResponseCache (cache/ 响应缓存，使用 flaxkv2 LMDB)
            ├── CostTracker (pricing/ 成本追踪)
            └── ImageProcessor (msg_processors/ 图片处理)

高级客户端：
    ├── MllmClient (多模态 LLM，支持图片/视频输入)
    ├── ChainOfThoughtClient (思维链推理)
    └── batch_tools/ (MllmTableProcessor, MllmFolderProcessor)
```

**注意**：`LLMClient` 现在是 `LLMClientPool` 的别名（`clients/llm.py`），单 endpoint 时行为完全一致，零额外开销。

### 关键设计模式

1. **客户端抽象**：`LLMClientBase` 定义 4 个核心抽象方法，子类只需实现差异化逻辑：
   - `_get_url()` - 构造请求 URL
   - `_get_headers()` - 构造请求头
   - `_build_request_body()` - 构造请求体
   - `_extract_content()` - 提取响应内容

2. **断点续传**：`chat_completions_batch()` 通过 `output_jsonl` 参数支持 JSONL 增量写入，中断后自动恢复

3. **响应缓存**：通过 `ResponseCacheConfig` 配置，支持 TTL 和多进程并发读写（LMDB 原生支持）

4. **思考模式**：统一的 `thinking` 参数支持 DeepSeek-R1、Qwen3、Claude、Gemini 等推理模型

## 测试结构

```
tests/
├── unit/           # 单元测试（pre-push 执行）
├── integration/    # 集成测试
├── e2e/            # 端到端测试（需要 API Key）
└── conftest.py     # pytest fixtures
```

测试需要环境变量：
- `GEMINI_API_KEY` - Gemini 测试
- `SILICONFLOW_API_KEY` - SiliconFlow 测试

pytest 配置已启用 `asyncio_mode = auto`，异步测试函数会自动运行。

## CLI 配置

配置文件位置：`~/.flexllm/config.yaml`

环境变量（优先级高于配置文件）：
- `FLEXLLM_BASE_URL` / `OPENAI_BASE_URL`
- `FLEXLLM_API_KEY` / `OPENAI_API_KEY`
- `FLEXLLM_MODEL` / `OPENAI_MODEL`

## 发版流程

当用户说"更新发版"时，执行以下步骤：

```bash
# 1. 确保 dev 分支已 push
git push origin dev

# 2. 切换到 main 分支并 rebase dev
git checkout main
git pull origin main
git rebase dev

# 3. 更新版本号（flexllm/__init__.py 中的 __version__）
# 根据改动类型决定版本号：patch(x.x.+1) / minor(x.+1.0) / major(+1.0.0)

# 4. 提交版本更新
git add flexllm/__init__.py
git commit -m "chore: bump version to x.x.x"

# 5. 生成 CHANGELOG 并创建 tag
git-cliff --tag vX.X.X -o CHANGELOG.md
git add CHANGELOG.md
git commit -m "chore(release): vX.X.X"
git tag vX.X.X

# 6. 推送（会触发 pre-push 测试和 GitHub Action 创建 Release）
git push origin main
git push origin vX.X.X
```

或使用发版脚本（需先手动更新版本号）：
```bash
./scripts/release.sh vX.X.X
git push origin main && git push origin vX.X.X
```

## 代码规范

- **ruff** 配置：`line-length=100`，`target-version="py310"`，lint 仅启用 isort（`select=["I"]`）
- ruff 不做风格检查（无 E/W/F 规则），主要职责是 import 排序和代码格式化

## Git Hooks（pre-commit）

项目配置了 pre-commit hooks：
- **pre-commit**: ruff（lint + 格式化）、trailing-whitespace、end-of-file-fixer 等
- **pre-push**: pytest 单元测试（tests/unit/）

```bash
# 手动运行所有检查
pre-commit run --all-files

# 跳过检查（紧急情况）
git commit --no-verify -m "message"
git push --no-verify
```

## CI/CD

- **GitHub Actions 测试**（`test.yml`）：push/PR 到 main 时运行，矩阵测试 Python 3.10 + 3.12，执行 `pytest tests/unit/`
- **GitHub Actions 发版**（`gh-release.yml`）：推送 `v*` tag 时触发，使用 git-cliff 生成 changelog 并创建 GitHub Release
