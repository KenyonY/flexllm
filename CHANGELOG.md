# Changelog

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
