# Changelog

## [0.8.5] - 2026-03-10

### Bug Fixes

- **cache**: 修复 pool batch 缓存三个问题

### Documentation

- **mock**: CLI help 补充 QA 匹配规则说明

### Features

- **mock**: QA 匹配支持子串包含，多个匹配取最长
- **mock**: 每个请求无条件在终端打印输入输出摘要，--log 可选写入文件

## [0.8.4] - 2026-03-10

### Features

- **mock**: 支持请求日志记录，base64 图片自动替换为占位符

## [0.8.3] - 2026-03-10

### Bug Fixes

- **mock**: 将 mock server 监听地址改为 0.0.0.0 以支持远程访问

### Features

- **mock**: 支持通过 QA 数据集实现确定性回复

## [0.8.2] - 2026-03-08

### Bug Fixes

- **cli**: 修复 help 文本格式混乱，pricing 支持每日自动更新
- **ci**: 补充 test 依赖 pytest-aiohttp 和 pyyaml

### Miscellaneous

- Bump version to 0.8.2

### Refactor

- **cache**: 移除 IPC 模式，切换到 LMDB 原生多进程支持

## [0.8.1] - 2026-03-07

### Features

- **claude**: 支持 OAuth token 认证和 models 命令

## [0.8.0] - 2026-03-07

### Bug Fixes

- **client**: Chat_completions 请求失败时打印 warning 日志

### Features

- **agent**: 流式输出、上下文压缩、编辑增强、项目指令、Skills、MCP 配置
- **cli**: Credits 命令支持直接通过 -k/--key 查询 API Key 余额

## [0.7.3] - 2026-03-03

### Features

- **client**: From_config 第一个参数改为配置文件路径，model 改为 keyword-only

### Miscellaneous

- Bump version to 0.7.3

## [0.7.2] - 2026-03-03

### Features

- **client**: From_config 支持字符串输入、user_template 和 .env

### Miscellaneous

- Bump version to 0.7.2

## [0.7.1] - 2026-03-03

### Features

- **client**: Add LLMClient.from_config() factory method

### Miscellaneous

- Bump version to 0.7.1

### Refactor

- 移除 MCP Server 模块（LLM 套 LLM 价值有限）

## [0.7.0] - 2026-02-25

### Features

- **processor**: 音频预处理和视频帧提取，fix aiohttp session 泄漏

### Miscellaneous

- Bump version to 0.7.0

## [0.6.2] - 2026-02-24

### Bug Fixes

- OpenAI client 自动将 audio_url 转换为 input_audio 格式

### Documentation

- 清理冗余过时文档

### Features

- **processor**: 支持 video_url/audio_url/input_audio 预处理
- **agent**: Subagent 支持、输出管理重构、mock tool call

### Miscellaneous

- Bump version to 0.6.2
- **pricing**: 更新模型定价数据

### Refactor

- **processor**: 统一默认预处理路径到 unified_processor

### Testing

- **e2e**: 添加视频/音频/图片媒体预处理端到端测试

## [0.6.1] - 2026-02-07

### Bug Fixes

- **ci**: Test extra 添加 mcp 依赖，CI 环境可运行 MCP 测试
- **test**: Mcp SDK 未安装时跳过 MCP server 创建测试

### Performance

- **cache**: IPC 缓存批量查询优化，25000 条从 ~6s 降至 ~0.26s

### Refactor

- **deps**: Mcp 从 agent extra 移入基础依赖

## [0.6.0] - 2026-02-06

### Features

- **agent**: 新增 Agent 基础设施，从 LLM 客户端升级为 Agent 平台

### Miscellaneous

- Bump version to 0.6.0

## [0.5.8] - 2026-02-03

### Bug Fixes

- **ci**: 添加 ruff 到 test 依赖
- **cli**: 修复 credits 命令因 API 返回 None 导致格式化失败

### Features

- **agent**: 新增代码验证闭环，支持修改后自动检查和修复
- **progress**: Batch 进度条适配非 TTY 环境
- **agent**: 新增细粒度文件工具和并行执行支持

### Miscellaneous

- Bump version to 0.5.8

### Performance

- **test**: 将 mock 相关测试标记为 slow，pre-push 从 40s 降到 0.7s
- **test**: 优化单元测试速度，pre-push 跳过慢测试

### Refactor

- **clients**: 提取批量处理公共逻辑到 batch_helpers 模块

### Testing

- **async_api**: 新增 async_api 模块核心功能单元测试

## [0.5.7] - 2026-02-02

### Features

- **agent**: 注册式内置工具扩展，支持多工具组合(shell/dtflow/maque/flexllm)
- 新增 AgentClient（tool-use 循环）+ CLI 模块拆分
- 支持模型调用参数(model_params)自动透传，修复 -m 参数取 id 逻辑

### Miscellaneous

- Bump version to 0.5.7, 更新模型定价数据

## [0.5.6] - 2026-02-01

### Bug Fixes

- **chat-web**: 重试时遵循上下文开关状态

### Documentation

- 统一对外接口为 LLMClient，移除 LLMClientPool 引用
- 更新 SKILL.md 和 README.md，添加 serve 命令文档

### Features

- **chat-web**: 支持 --title 参数自定义页面 Logo 文本

## [0.5.5] - 2026-02-01

### Bug Fixes

- **chat-web**: 修复多轮对话时输入框被挤出视口

### Features

- **serve**: 新增 flexllm serve 命令，将 LLM 包装为 HTTP API 服务
- **chat-web**: 添加重试和上下文发送按钮
- 添加 Web 聊天界面和流式思考内容支持

### Refactor

- **chat-web**: 上下文按钮改为状态切换，默认不携带

### Styling

- **chat-web**: 按钮移入输入框内部，对齐主流聊天界面
- **chat-web**: 停止按钮改为黑白配色
- **chat-web**: 对齐 nanochat/claude.ai 视觉风格

## [0.5.4] - 2026-01-31

### Bug Fixes

- **progress**: 无定价信息时显示 token 数量而非成本

### Miscellaneous

- Bump version to 0.5.4

## [0.5.3] - 2026-01-31

### Bug Fixes

- **cli**: 修复 --user-field 参数导致格式检测错误的问题

### Miscellaneous

- Bump version to 0.5.3

## [0.5.2] - 2026-01-31

### Documentation

- 添加配置文件示例
- **readme**: 补充 system 字段配置说明

### Features

- **config**: 支持 user_template 配置
- **cli**: Batch 命令的 --output 参数改为可选

### Miscellaneous

- Bump version to 0.5.2

## [0.5.1] - 2026-01-31

### Features

- **config**: 添加系统提示词配置支持

### Miscellaneous

- Bump version to 0.5.1

## [0.5.0] - 2026-01-30

### Documentation

- 移除中文 README
- README 补充 batch -n 和 -uf/-sf 示例
- 精简 SKILL.md 并补充 batch 命令文档
- 修正 Claude Code 集成说明
- 完善 Claude Code 集成说明
- 在安装部分添加 Claude Code 集成说明
- 添加 roadmap 规划文档
- 重构文档，突出核心功能定位

### Features

- **mock**: 支持 Claude/Gemini API 格式和思考内容返回
- **cli**: Batch 命令添加 -n/--limit 参数，只处理前 N 条记录
- **cli**: Batch 命令添加 --user-field/-uf 和 --system-field/-sf 参数
- **cli**: Alpaca 格式支持记录级 system 字段
- **cli**: Batch 输入格式支持 input/user 字段
- **cli**: Batch 命令添加 --save-input 参数
- **batch**: 添加 save_input 参数控制输出 JSONL 的 input 保存策略
- **cli**: 添加 install-skill 命令
- **pool**: 健康检查默认阈值设为无限大
- **progress**: 优化进度条显示重试和错误信息

### Miscellaneous

- Bump version to 0.5.0 & 修复 test_mock_e2e 中残留的 load_balance 参数
- **skill**: 移除多余的安装说明
- 补充版权信息并升级版本至 0.4.6

### Performance

- **progress**: 添加进度条刷新节流控制

### Refactor

- **router**: 移除 load_balance 参数，简化为 round_robin 策略
- **clients**: 统一 LLMClient/LLMClientPool 架构 & 修复 async event loop 绑定问题

## [0.4.5] - 2026-01-23

### Features

- **cache**: 优化响应缓存结构，支持存储 usage 信息

## [0.4.4] - 2026-01-22

### Bug Fixes

- **pool**: 修复 aiohttp session 未关闭警告
- **pool**: 为 LLMClientPool.chat_completions_batch 添加 metadata_list 参数支持

### Features

- **cli**: 添加 credits 命令支持查询 API Key 余额

## [0.4.3] - 2026-01-21

### Features

- **pool**: 优化 fallback 重试机制和连接复用

## [0.4.2] - 2026-01-21

### Bug Fixes

- 双行进度条在无定价时也能显示模型名和token统计
- 改进模型定价匹配逻辑，支持 provider/model 格式

### Features

- **pool**: 统一 LLMClientPool 进度条显示，修复 output_jsonl 功能
- 进度条支持双行显示成本信息

### Miscellaneous

- Bump version to 0.4.2

## [0.4.1] - 2026-01-20

### Bug Fixes

- 修复并发滑动窗口未正常工作的问题

## [0.4.0] - 2026-01-19

### Documentation

- 添加发版流程和 Git Hooks 说明

### Features

- 重构目录结构，添加成本追踪功能

## [0.3.4] - 2026-01-18

### Miscellaneous

- Bump version to 0.3.4
- 添加 git-cliff 配置和发版脚本
- 添加 pre-commit 配置，使用 ruff 替代 black+isort

### Styling

- 使用 ruff 格式化代码

## [0.3.3] - 2026-01-18

### Refactor

- 重构模块结构，优化代码组织

### V0.3.2

- 重构定价模块，支持从 OpenRouter API 自动更新

## [0.3.1] - 2026-01-17

### Features

- 最低python版本切换为python3.10

### V0.3.1

- 优化依赖结构，扩展 batch 命令配置文件支持

## [0.3.0] - 2026-01-11
