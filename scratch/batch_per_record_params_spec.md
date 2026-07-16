# 需求规格：batch 支持 per-record 参数 + 多轮 system/模板语义统一

> 状态：待实现（交接给实现者）
> 涉及命令：`flexllm batch`

## 0. 总目标

让 `flexllm batch` 的输入文件 `input.jsonl` 每行能单独设置参数
（`system`、`user_template`、`stop`、`temperature` 等），用于**批量构造不同参数下的回复（参数扫描 / parameter sweep）**。

典型场景：同一个 prompt 配不同 `temperature` / `stop` / 模板，一次跑出来对比效果。

**贯穿原则：行内 > 配置。** 配置 / CLI 提供默认值；行内（messages 自带的 system、记录里的 `system` 字段、`params`）优先；配置只兜底，不抢戏。

分两步实现，第一步是第二步的地基。

---

## 第一步：统一全局 / CLI 的 system + user_template 应用语义

### 改动位置

`flexllm/cli/utils.py` 的 `convert_to_messages()`（当前约 122–205 行）。

签名不变：

```python
convert_to_messages(record, format_type, message_fields,
                    global_system=None, user_template=None) -> (messages, metadata)
```

### 目标行为（对所有格式统一）

1. **system 兜底**：先按各格式构造好 `messages`（保留行内 system 原样），最后统一处理：
   - 若 `messages` 中**不存在** `role == "system"` 的消息，且 `global_system` 非空，
     则把 `{"role": "system", "content": global_system}` 插到 `messages[0]`；
   - 若已有行内 system，**保持不动**。

2. **user_template**：只套用到 `messages` 中**最后一条** `role == "user"` 且 `content` 为 `str` 的消息
   （从后往前找到第一条符合的即停），其余 user 消息不动；多模态消息（`content` 为 list）跳过。

### 需要删除 / 改写的旧逻辑

- **约 192–194 行**：`if global_system and format_type != "openai_chat": 删所有 system 再插 global_system`
  —— 整段替换为上面的"统一兜底"。
- **simple 分支（约 169 行）**：`system = global_system or record.get("system")`
  —— 改成只读行内 `record["system"]`；`global_system` 交给统一兜底。
- **openai_chat 分支（约 133–142 行）**："对每条 user 套模板"
  —— 删除，改由统一的"只套最后一条"处理。
- **alpaca / simple / custom 分支里各自的 `apply_user_template(...)` 调用**
  —— 删除，统一处理。

### 必须保留

- `used_fields` 收集逻辑（决定哪些字段进 metadata），含 alpaca 的 `"output"`。
- `prefix` 处理（约 196–202 行）保持在最后（prefix 是 assistant 消息，不受模板影响）。

### 行为变更清单（breaking，需在 CHANGELOG / commit 说明）

1. **openai_chat + 配 system**：之前永远忽略 → 现在行内无 system 时补上。（修复 bug）
2. **openai_chat 多轮 + 配 user_template**：之前套每条 user → 现在只套最后一条。
3. **simple / alpaca / custom + 全局 system + 行内也有 system**：之前全局覆盖行内 → 现在行内优先。

> 不配 system / template 时，所有格式行为不变（回归基线）。

---

## 第二步：per-record `params`

### 输入格式

每行记录可选带 `params`（dict），例：

```json
{"messages": [{"role":"user","content":"解释量子纠缠"}], "params": {"temperature": 0.2, "stop": ["\n\n"]}}
{"q": "解释量子纠缠", "params": {"system": "你是物理老师", "user_template": "用初中生能懂的话：{content}", "temperature": 0.9}}
```

字段名固定用 **`params`**（嵌套，不平铺——与业务 metadata 干净区分）。

### params 内两类键，消费位置不同

- **消息构造类：`system`、`user_template`**
  —— 在 `convert_to_messages` 阶段消费，作为该行的有效 system / 模板，
  仍遵守第一步语义（system 行内优先兜底、模板只套最后一条 user）。
- **生成参数类：其余所有键**（`temperature` / `stop` / `max_tokens` / `top_p` / `response_format` …）
  —— 作为该行 per-record kwargs，在请求体构造时覆盖全局 kwargs。

### 优先级

- **system 有效值（兜底用）**：`messages 内显式 system` > `params.system` > `CLI -s` > `配置 models.system`。
  即 messages 里写了 system 就最高；否则用 `params.system` 补；再否则用 CLI / 配置补。
- **user_template**：`params.user_template` > `CLI --user-template` > `配置 user_template`。
- **生成参数**：`params.<key>` > `CLI 对应参数` > `配置 models 节`。

### 引擎层改动

- `chat_completions_batch`（`flexllm/clients/base.py:421`）与 pool 版（`flexllm/clients/pool.py:724`）
  新增 `params_list: list[dict] | None`（与 `messages_list` 等长，元素为该行生成参数 dict 或 None）。
- 请求体构造：每条有效参数 = `{**全局kwargs, **(params_list[i] or {})}`。
- **缓存**：`get_batch` / `_make_key`（`flexllm/cache/response_cache.py:213`）当前用统一 `**kwargs` 算 key；
  改为每条用 `{**kwargs, **(params_list[i] or {})}` 算各自 key，保证不同参数不互相命中。
- **断点续传**：`save_input` 校验只看 messages，不感知 `params` 变化
  （与现有"改 kwargs 不影响续传判定"一致，属预期行为，需在文档注明）。

### CLI / 命令层改动

`flexllm/cli/commands.py` 的 `batch`（约 1035–1141 行）：

- 逐条解析 `params`，分离出 (per-record system, per-record user_template) 与 per-record 生成 kwargs。
- 调 `convert_to_messages` 时传入按优先级解析后的有效 system / user_template。
- 组装 `params_list` 传给 `chat_completions_batch`。
- `convert_to_messages` 把 `"params"` 加入 `used_fields`，**不让它原样透传进 metadata**。
- `--dry-run` 至少展示第一条解析后的实际参数，便于确认。

---

## 输出回显（参数扫描的关键）

**默认开启**，但只对带 `params` 的行生效：

- **写什么**：把该行的 `params` **原样回显**到输出（放一个明确字段，如 `params`）。
  由于优先级 `params > CLI > 配置`，params 里写的就是该行实际生效的值，原样回显即"这条到底用了什么"。
- **写多少**：只回显 per-record 的 `params`（扫描的自变量）；
  **不**把全局默认参数塞进每行（对每行都一样，纯冗余；全局值在命令 / 配置里有据可查）。
- **默认开 / 关**：默认开。无 `params` 的行不会多出该字段，普通 batch 行为完全不变（零回归）；
  有 `params` 的行正是要分析的，写了刚好。因此无需额外开关。

理由：参数扫描的价值在于把"回复"和"产生它的参数"对应起来。若输出只有 messages + 回复，
一旦有断点续传 / 部分失败 / 并发乱序，靠行号回查输入文件去拼参数会很脆弱。

---

## 验收要点

> 建议用 `flexllm mock` 起真实服务做端到端验证，不只 mock 单测；
> 对第三方请求体的关键字段（stop / temperature）需抓真实请求确认。

**第一步**

- openai_chat 行内无 system + 配 system → system 插到开头。
- openai_chat 行内有 system + 配 system → 保留行内。
- openai_chat 多轮 + 配 user_template → 仅最后一条 user 被套，历史轮原样。
- simple 行内有 system + 配全局 system → 行内优先。
- simple 无行内 system + 配全局 → 补全局。
- 不配 system / template → 四种格式行为不变（回归）。

**第二步**

- 每行不同 temperature / stop → 实际请求体各自不同（mock server 抓请求验证）。
- `params.system` / `params.user_template` 覆盖全局，且遵守第一步语义。
- `params` 不原样出现在输出 metadata 里。
- 相同 messages、不同 `params` → 缓存互不命中。
- 带 `params` 的行 → 输出含回显的 `params` 字段；不带的行 → 无该字段。
- 无 `params` 字段 → 行为等同第一步（回归）。

---

## 待拍板点（实现前可再确认）

1. **messages 内显式 system 是否应优先于 `params.system`**——本规格按"最明确者优先"定为是。
2. **输出回显是否只回显 per-record `params`**——本规格定为是（不回显全局默认）。
   若需"每行自包含的完整生效参数快照"，则改为回显合并后的最终参数（代价：每行更啰嗦）。
