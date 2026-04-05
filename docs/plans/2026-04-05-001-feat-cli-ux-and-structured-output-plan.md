---
title: "feat: CLI 体验增强 — 结构化输出、代码提取、文件输入"
type: feat
status: completed
date: 2026-04-05
---

# CLI 体验增强 — 结构化输出、代码提取、文件输入

## Overview

借鉴 simonw/llm 的 CLI 设计，增强 flexllm 的 CLI 交互体验。核心后端能力（结构化输出、tool calling）已就绪，本计划聚焦于 CLI 层暴露和实用 UX 功能。

## Problem Frame

flexllm 的核心优势在批量处理和高并发，但日常 CLI 交互体验有改进空间：
1. 结构化输出后端已支持（三个 client 均实现），但 CLI 无法使用
2. 缺少常用 CLI 便捷功能：代码提取、文件输入

## Requirements Trace

- R1. CLI 暴露结构化输出能力（`--schema`），覆盖 ask/chat/batch
- R2. 支持代码块提取：`-x` 从回复中提取 fenced code block
- R3. 支持文件输入：`-f file` 将文件内容附加到 prompt

## Scope Boundaries

- 不引入插件系统（ROI 不足）
- 不引入默认命令（`flexllm ask` 已足够清晰）
- 不引入对话持久化/续接（flexllm 核心场景是批量/API，非 CLI 聊天）
- 不实现 CLI tool execution loop（执行循环留给 Python API）
- 不引入 SQLite 日志系统

## Context & Research

### Relevant Code and Patterns

- `flexllm/cli/commands.py` — 所有 CLI 命令注册
- `flexllm/cli/chat_helpers.py` — interactive_chat/single_chat 实现
- `flexllm/clients/base.py:304` — `chat_completions` 已通过 `**kwargs` 传递 `response_format`
- `flexllm/clients/claude.py:121` — Claude 已显式处理 `response_format`（转为 prompt 注入）
- `flexllm/clients/gemini.py:143` — Gemini 已显式处理 `response_format`（转为原生格式）
- `flexllm/clients/openai.py:179` — OpenAI 通过 `body.update(kwargs)` 透传

### simonw/llm 的相关设计

- 代码提取：`-x` 提取第一个 fenced code block，`--xl` 提取最后一个
- 结构化输出：`--schema` 接受 JSON Schema 或简洁 DSL
- 文件输入：`-f` / `--fragment` 附加文件/URL 内容

## Key Technical Decisions

- **结构化输出参数设计**：`--schema json` 简写为 json_object 模式；`--schema '{"type":"object",...}'` 接受 JSON Schema 字符串；`--schema @file.json` 从文件读取。不引入 DSL 语法（KISS）。
- **代码提取**：纯输出后处理，用正则提取 markdown fenced code block，不改变 API 调用逻辑。
- **文件输入**：`-f` 读取文件内容，拼接到 prompt 前面（类似 stdin 管道的行为），支持多次指定。

## Open Questions

### Resolved During Planning

- **`--schema` 的 JSON 解析可能因 shell 转义复杂**：支持 `@file.json` 从文件读取作为 fallback。

### Deferred to Implementation

- `--schema @file.json` 的文件路径解析细节（相对路径 vs 绝对路径）

## Implementation Units

- [x] **Unit 1: CLI 结构化输出支持**

**Goal:** 在 ask/chat/batch 命令中添加 `--schema` 参数，透传到已有的 `response_format` 后端

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `flexllm/cli/commands.py`
- Modify: `flexllm/cli/utils.py`（添加 schema 解析辅助函数）
- Test: `tests/unit/test_cli_schema.py`

**Approach:**
- 添加 `--schema` Option 到 ask/chat/batch 三个命令
- 解析逻辑：`json` → `{"type": "json_object"}`（简写），`@file.json` → 读取文件，JSON 字符串 → 解析为 dict
- 构造 `response_format` dict 传入 `model_params`（已有 kwargs 透传链路）
- batch 命令的 `--schema` 对所有记录统一生效

**Patterns to follow:**
- 参考 `--thinking` 参数的处理模式（`parse_thinking` 解析 → 写入 `model_params`）

**Test scenarios:**
- Happy path: `--schema '{"type":"object","properties":{"name":{"type":"string"}}}'` 正确解析为 response_format dict
- Happy path: `--schema json` 简写解析为 `{"type": "json_object"}`
- Happy path: `--schema @schema.json` 从文件读取 schema
- Edge case: 无效 JSON 字符串 → 明确错误提示
- Edge case: `@` 引用的文件不存在 → 明确错误提示

**Verification:**
- `flexllm ask "列出3种编程语言" --schema json` 返回 JSON
- `flexllm batch input.jsonl -o out.jsonl --schema @schema.json` 批量输出结构化数据

---

- [x] **Unit 2: 代码块提取 `-x`**

**Goal:** 从 LLM 回复中提取 fenced code block 并只输出代码内容

**Requirements:** R2

**Dependencies:** None

**Files:**
- Modify: `flexllm/cli/commands.py`（ask/chat 添加 `-x` flag）
- Modify: `flexllm/cli/utils.py`（添加 `extract_code_block` 函数）
- Test: `tests/unit/test_extract_code.py`

**Approach:**
- 正则匹配 ` ```...``` ` 格式的 fenced code block
- `-x` 提取第一个 code block
- 在 print 输出前做后处理，不改变 API 调用
- 如果没有 code block，原样输出并在 stderr 提示

**Patterns to follow:**
- simonw/llm 的提取逻辑（简单正则 + 去除 fence 行）

**Test scenarios:**
- Happy path: 包含单个 code block 的回复 → 正确提取代码内容（不含 ``` 标记）
- Happy path: 包含多个 code block → `-x` 返回第一个
- Edge case: 无 code block → 原样输出 + stderr 提示
- Edge case: 嵌套或不完整的 fence → 提取第一个完整的

**Verification:**
- `flexllm ask "写一个Python hello world" -x` 只输出纯代码

---

- [x] **Unit 3: 文件输入 `-f`**

**Goal:** 支持 `-f file` 将文件内容附加到 prompt，方便对文件内容提问

**Requirements:** R3

**Dependencies:** None

**Files:**
- Modify: `flexllm/cli/commands.py`（ask/chat 添加 `-f` 选项）
- Test: `tests/unit/test_cli_file_input.py`

**Approach:**
- `-f path` 读取文件内容，拼接到 prompt 前面（格式：`文件内容\n\n用户问题`）
- 支持多次指定（`-f a.py -f b.py`），按顺序拼接
- 与 stdin 管道兼容：stdin + `-f` + prompt 三者均可组合
- 大文件保护：超过阈值（如 100KB）时 stderr 警告但不阻断

**Patterns to follow:**
- 参考 ask 命令已有的 stdin 拼接逻辑（`commands.py:61-63`）

**Test scenarios:**
- Happy path: `flexllm ask -f code.py "解释这段代码"` → 文件内容 + 问题正确拼接
- Happy path: 多文件 `-f a.py -f b.py "对比"` → 按序拼接
- Edge case: 文件不存在 → 明确错误提示
- Edge case: stdin + `-f` 组合 → 三者正确合并
- Integration: 与 `--user-template` 组合使用时，模板应用于最终拼接结果

**Verification:**
- `flexllm ask -f README.md "总结这个项目"` 正确读取文件并返回摘要

## System-Wide Impact

- **CLI 参数兼容性**：所有新参数使用长选项 + 短选项格式，不与现有参数冲突。
- **kwargs 传递链路**：`response_format` 已有完整的 CLI → chat_completions → _build_request_body 链路，无需修改后端。
- **输出后处理**：`-x` 代码提取在 print 前做后处理，不影响 API 层和缓存层。
- **Unchanged invariants**：Python API（LLMClient, chat_completions, chat_completions_batch）行为不变；缓存机制不变；batch 的断点续传逻辑不变。

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `--schema` 的 JSON 解析可能因 shell 转义复杂 | 支持 `@file.json` 从文件读取作为 fallback |

## Sources & References

- simonw/llm: https://github.com/simonw/llm （CLI 设计参考）
- 现有 kwargs 透传链路：`flexllm/clients/base.py:304`
