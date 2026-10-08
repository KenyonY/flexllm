# Reasoning：统一接口，按端点和模型声明能力

`Reasoning` 表达调用意图，`ReasoningCapabilities` 描述某个端点上的某个模型，
`reasoning_adapter` 负责协议转换。能力查询不访问网络，也不根据模型名字猜测档位。

## Python 调用

```python
from flexllm import LLMClient, Reasoning

client = LLMClient.from_config(model="my-model")

result = await client.chat(messages)  # 继承配置的 reasoning
result = await client.chat(messages, reasoning=Reasoning(effort="high"))
result = await client.chat(messages, reasoning=Reasoning(enabled=False))
result = await client.chat(messages, reasoning=Reasoning(budget_tokens=4096))
result = await client.chat(messages, reasoning=None)  # 明确使用服务端默认
```

后面三种控制方式并不保证每个模型都支持；已声明不支持时，在缓存查询和网络请求之前报
`ValueError`。`Reasoning()`、`{}`、显式 `None` 都表示不发送思考控制字段。
省略请求参数才继承客户端默认。一次请求的 reasoning **整体替换**默认策略，不逐字段合并。
`from_config(..., reasoning=...)` 可替换配置默认。

- `enabled`：明确开启或关闭。开启不等同于 `medium`；有些协议没有独立的开启开关。
- `effort`：服务端的原生档位名，原样发送。不把 `xhigh`、`max`、`ultra` 相互映射。
- `budget_tokens`：正整数预算，不把强度档位换算成 token 数。其计量方式由服务端定义。
- `enabled=False` 不能同时指定 effort 或预算。effort 与预算仅在能力中显式声明允许组合时可并用。

`chat_stream`、`chat_batch`、`chat_batch_iter` 及同步入口使用同样的参数。
批处理每行的 `params.reasoning` 整体覆盖请求级策略。非法参数属于用法错误；批量参数应在
发起请求前修正。缓存使用实际转换后的生成参数；思考策略不同不会共用同一缓存键。
断点续传仍然按消息匹配已完成记录，修改策略后要使用新的输出文件才能重新执行。

正文在 `result.content`，可用的思考文本在 `result.reasoning_content`，用量在 `result.usage`。
流式思考使用 `thinking` 事件。没有返回思考文本不代表模型没有推理。

## 能力查询和配置

```python
caps = client.capabilities.reasoning
print(caps.effort_levels)  # 例如 ("high", "max")
print(caps.can_disable)
print(caps.budget_tokens)
```

| 字段 | 值的含义 |
| --- | --- |
| `can_enable` / `can_disable` | `True` 可用，`False` 不可用，`None` 未知 |
| `effort_levels` | 元组为已声明的档位；空元组是不支持；`None` 未知 |
| `budget_tokens` | `TokenBudget(min, max)` 为范围；`False` 不支持；`None` 未知 |
| `supports_effort_and_budget` | 默认 `False`，不擅自组合两种控制方式 |

未知能力允许显式尝试，由后端校验，但协议适配器无法表达的控制会在本地报错。
单独的 `enabled=True` 在 OpenAI effort 协议下没有直接表示方式，需明确选择 effort
（允许 `enabled=True, effort="high"`）；`can_enable=False` 表示不支持独立开启开关。Gemini 需指定
其支持的 effort 或预算。Claude adapter 的 `enabled=True` 使用 adaptive thinking；旧款手动
思考模型应显式给预算。

下面是一个 SiliconFlow 配置示例；模型范围及档位要根据实际部署确认：

```yaml
models:
  - name: my-model
    id: deepseek-ai/DeepSeek-V4-Flash
    provider: openai
    base_url: https://api.siliconflow.cn/v1
    api_key: YOUR_API_KEY
    reasoning:
      effort: high
    reasoning_adapter: siliconflow
    reasoning_capabilities:
      can_enable: true
      can_disable: true
      effort_levels: [high, max]
      budget_tokens: {min: 128, max: 32768}
      supports_effort_and_budget: true
```

这三个配置字段由本地解析。`reasoning_capabilities`、`reasoning_adapter` 不会进入 HTTP 请求体；
`reasoning` 转换成目标协议字段（OpenRouter 恰好也叫 `reasoning`）。Python 构造函数接受同名参数；能力也可使用 `ReasoningCapabilities` 和 `TokenBudget` 对象。

声明绑定到配置的模型。请求时改用另一个 `model`，能力回到未知；CLI 或 `from_config`
显式替换 base URL、模型或端点列表时，不继承原目标的能力声明和 adapter。
单次请求的 `url=` 若不同于客户端生成的请求地址，不能同时使用非空 reasoning 策略
（包括客户端默认）；请为目标 `base_url` 建立客户端并声明对应能力，避免误用原端点协议。
多端点可以在每个 endpoint 中分别配置这三个字段。用 `client.endpoint_capabilities` 查询各自
能力；只有所有端点声明一致时才允许 `client.capabilities`。路由中的每个端点分别验证请求，
故障转移不改变调用方指定的档位。CLI 预检会检查目标池的全部端点。

当前能力来源是显式配置。`/models` 通常只列模型 ID 和协议，不能据此承诺某档位可用。
OpenRouter 的模型目录还可能提供 `reasoning.supported_efforts`、`mandatory` 等能力元数据，
可据此填写声明，但当前客户端不会自动拉取。模型出现在目录中不保证当前账号或地区可调用。
不内置推测性的型号表，也不自动执行付费探测。客户端构造时读取配置快照，修改文件后需重建客户端。

## 协议适配

| adapter | 开关 | effort | 预算 |
| --- | --- | --- | --- |
| `openai` | 关闭发 `reasoning_effort: none`；不支持单独开启 | `reasoning_effort` | 不支持 |
| `openrouter` | `reasoning.enabled` | `reasoning.effort` | `reasoning.max_tokens` |
| `siliconflow` | `enable_thinking` | `reasoning_effort`，同时开启思考 | `thinking_budget`，同时开启思考 |
| `deepseek` | `thinking.type: enabled/disabled` | `reasoning_effort`，同时开启思考 | 不支持 |
| `vllm` | `chat_template_kwargs.enable_thinking` | `reasoning_effort`，同时开启思考 | 不支持 |
| `claude` | `thinking.type: adaptive/disabled` | `output_config.effort`，同时启用 adaptive | `thinking.budget_tokens`，使用 enabled 模式 |
| `gemini` | 关闭发 `thinkingBudget: 0` | `thinkingConfig.thinkingLevel` | `thinkingConfig.thinkingBudget` |

`provider=claude/gemini` 自动使用相应 adapter。OpenAI 协议对 SiliconFlow、DeepSeek 的官方
域名选择对应 adapter，`openrouter.ai` 自动使用 `openrouter`，其他地址默认 `openai`；自建 vLLM、中转的 DeepSeek 路由需显式配置。
自动选择格式不等于确认模型能力。Gemini 2.x 应声明只支持预算，不接受 effort；本接口不会
套用旧版的等级转预算逻辑，也不允许同时指定 effort 和预算。Claude 的预算还受服务端 `max_tokens` 约束，现有客户端会在必要时
提高输出上限以满足协议。

OpenRouter 使用 `provider: openai` 和 `base_url: https://openrouter.ai/api/v1`，即使目标是
Gemini，也无需切换成原生 Gemini 协议。可显式设置 `reasoning_adapter: openrouter`。
`Reasoning(budget_tokens=128)` 会发送 `reasoning: {max_tokens: 128}`，与输出上限
`max_tokens` 区分；不允许同时指定 effort 和预算。FlexLLM 不转换档位，但 OpenRouter 或其
上游可能自行映射档位和预算，因此 Gemini 3 的预算参数不能理解为精确的 token 上限。
新接口仍只接受 `enabled`、`effort`、`budget_tokens`，不能直接传 OpenRouter 原生
`reasoning.max_tokens` 或 `reasoning.exclude`；省略策略或传 `None` 的语义与其他 adapter 一致。

原生参数仍可独立使用，但不能和新 `reasoning`（包括配置默认）混用。旧 `thinking` 入口暂时
保持旧行为；新代码使用此处的契约。不要同时配置 `thinking` 和 `reasoning`。

## CLI

```bash
flexllm capabilities -m my-model --json
flexllm ask -m my-model --reasoning-effort high "只回复 OK"
flexllm ask -m my-model --no-reasoning-enabled "只回复 OK"
flexllm ask -m my-model --reasoning-budget 4096 "只回复 OK"
flexllm ask -m my-model --reasoning-default "只回复 OK"
flexllm ask -m my-model --reasoning-effort high "只回复 OK" --dry-run
```

`ask`、`chat`、`batch`、`serve`、`chat-web` 支持以上 reasoning 参数。
`capabilities` 始终输出 JSON，未知值为 `null`，不输出密钥。非法配置和不支持的档位退出码为 2；
有效 dry-run 仍为 10。`--reasoning-default` 不能与其他思考控制参数组合。
Web/Serve 使用启动时传入的策略和能力配置；当前网页没有动态的思考档位选择控件。

## 厂商协议参考

- [OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning)
- [Claude effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [OpenRouter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [DeepSeek thinking](https://api-docs.deepseek.com/guides/thinking_mode/)
- [SiliconFlow Chat Completions](https://docs.siliconflow.cn/docs/api/chat-completions-post)

厂商和中转可更新协议。HTTP 200 只能证明请求被接受，不能单独证明强度未被服务端忽略或映射。
