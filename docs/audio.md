# 语音能力

flexllm 支持 OpenAI 兼容的两个语音端点，以及对话接口里的音频输入。

| 能力 | 端点 | Python API | CLI |
|---|---|---|---|
| 语音转录（音频→文字） | `/audio/transcriptions` | `transcribe` / `transcribe_batch` | `flexllm transcribe` |
| 语音合成（文字→音频） | `/audio/speech` | `speech` / `speech_batch` | `flexllm speak` |
| 音频输入对话 | `/chat/completions` | `chat_completions` | — |

三者都复用客户端的并发、QPS 限流、重试和代理配置。

---

## 语音转录

```python
from flexllm import LLMClient

client = LLMClient(base_url="https://open.bigmodel.cn/api/paas/v4", api_key="...")

# 最简：返回文本
text = client.transcribe_sync("a.wav", model="glm-asr")

# 拿到分段/语言/时长，并渲染字幕
result = client.transcribe_sync("a.wav", model="glm-asr", return_details=True)
print(result.text, result.language, result.duration)
print(result.to_srt())     # SRT 字幕
print(result.to_vtt())     # WebVTT 字幕

# 批量并发（并发数由 concurrency_limit 控制）
texts = client.transcribe_batch_sync(["a.wav", "b.wav"], model="glm-asr")
```

异步版本去掉 `_sync` 后缀：`await client.transcribe(...)`。

**参数**

| 参数 | 说明 |
|---|---|
| `audio` | 音频文件路径，或原始 bytes（此时须配合 `filename`） |
| `model` | 转录模型（`whisper-1` / `glm-asr` / `glm-asr-2512`）。省略时用客户端默认模型 |
| `language` | 音频语言（ISO-639-1，如 `zh`），给出可提升准确率 |
| `prompt` | 引导提示词，如专有名词表 |
| `response_format` | 仅 `json` / `verbose_json`，字幕在本地渲染（见下） |
| `timestamp_granularities` | 时间戳粒度，如 `["segment"]`、`["word"]` |
| `return_details` | 返回 `TranscriptionResult` 而非纯文本 |
| `return_raw` | 返回 `RequestResult` 原始响应 |

未识别的关键字参数原样透传为表单字段，例如智谱 `glm-asr-2512` 的 `stream=False`。

### 字幕为什么在本地渲染

`srt` / `vtt` 不透传给服务端，而是由 `segments` 本地渲染，原因有两条：

1. **各家支持不一致**。智谱 `glm-asr` 直接忽略 `response_format`，无论传什么都返回 JSON；传 `srt` 只会得到一份 JSON。
2. **纯文本响应会被判为错误**。OpenAI 在 `response_format=srt` 时返回的是纯文本，而 `ConcurrentRequester` 对 2xx 非 JSON 响应按错误处理。

`segments` 是结构化的，本地渲染跨 provider 结果一致。传入 `srt`/`vtt`/`text` 会直接报错并提示改用 `to_srt()`。

若模型没有返回 `segments`（如 `glm-asr-2512`），会用整段文本兜底成单条 segment，字幕仍可生成，但时间戳为 0。

### CLI

```bash
flexllm transcribe a.wav -m glm-asr                 # 转录文本到 stdout
flexllm transcribe a.wav -m whisper-1 -l zh         # 指定语言
flexllm transcribe a.wav -m glm-asr -f json         # 含 segments/language/duration
flexllm transcribe a.wav -m glm-asr -f srt          # SRT 字幕
flexllm transcribe *.wav -m glm-asr -c 10           # 批量并发，输出 JSONL
flexllm transcribe *.wav -m glm-asr -f srt          # 批量：各自写同名 .srt
flexllm transcribe *.wav -m glm-asr -o out.jsonl    # 写入文件
```

输出约定：

- 单文件默认打印纯文本，便于管道；多文件打印 JSONL（含 `file` 字段保留对应关系）
- `-f srt|vtt` 且多文件时写到各音频的同名字幕文件；此时不能再用 `-o` 指向单个文件——
  多份字幕拼进一个文件序号会从 1 重来、时间轴回退，VTT 还会多出一个 `WEBVTT` 头，
  不构成合法字幕，因此该组合会直接报错
- 进度条与写入路径提示走 stderr，stdout 只有数据
- 有文件转录失败时以非零退出码结束，失败清单在错误 JSON 的 `context.failed` 中
- 同名字幕文件已存在时直接覆盖

---

## 语音合成

```python
# 返回音频字节
audio: bytes = client.speech_sync("你好", model="glm-tts", voice="tongtong")

# 直接写文件，返回 Path
path = client.speech_sync("你好", model="glm-tts", voice="tongtong", output="hello.wav")

# 批量并发
paths = client.speech_batch_sync(
    ["第一句", "第二句"], model="glm-tts", voice="tongtong",
    outputs=["1.wav", "2.wav"],
)
```

**参数**：`text`（对应服务端 `input`）、`model`、`voice`、`response_format`（默认 `wav`）、`speed`、`output`。其余关键字参数透传给服务端。

`/audio/speech` 的响应是音频字节而非 JSON，因此请求走 `response_type="bytes"`，跳过 JSON 解析。请求失败时服务端返回的仍是 JSON 错误体，会照常解析并以 `RequestResult` 返回。

### CLI

```bash
flexllm speak "你好，今天天气怎么样" -m glm-tts -v tongtong   # 默认写 speech.wav
flexllm speak "你好" -m glm-tts -o hello.wav                   # 指定输出路径
echo "长文本" | flexllm speak -m glm-tts                       # stdin 输入
flexllm speak "你好" -m glm-tts -o - | ffplay -                # 管道播放
```

未指定 `-o` 时写入当前目录的 `speech.<format>`，避免把二进制喷到终端；`-o -` 显式写 stdout。

---

## 对话中的音频输入

多模态模型（如 `glm-4-voice`、`gpt-4o-audio-preview`）通过 `/chat/completions` 直接接收音频。

```python
messages = [{"role": "user", "content": [
    {"type": "text", "text": "复述音频内容"},
    {"type": "audio_url", "audio_url": {"url": "/path/to/a.wav"}},   # 本地路径
]}]

resp = client.chat_completions_sync(messages, preprocess_msg=True)
```

**`preprocess_msg=True` 必须显式开启**，否则本地路径不会被读取转成 base64。

支持两种写法：

- `audio_url`：flexllm 便捷格式，值可以是本地路径、http URL 或 data URI；对 OpenAI 兼容端点会自动转成标准 `input_audio`
- `input_audio`：OpenAI 标准格式，`data` 字段同样支持直接写本地路径

MIME 子类型会规范化成 OpenAI 规范要求的 `format` 值。这一步是必需的：Linux 上 `mimetypes` 对 `.wav` 猜出的是 `audio/x-wav`，直接把子类型当 `format` 发出去会被服务端拒绝（智谱返回 `error code 1214`）。

### 音频预处理

安装 `pip install "flexllm[audio]"`（soundfile + scipy）后可在预处理阶段重采样、转单声道、转格式、截断：

```python
resp = await client.chat_completions(
    messages, preprocess_msg=True,
    target_sample_rate=16000, target_channels=1,
    target_audio_format="wav", max_duration_seconds=30,
)
```

> 注意：这些参数目前只在 `MllmClient` 路径生效，`LLMClient` 不会透传它们（音频仍会正常 base64 化，只是不做重采样/截断）。

### 模型约束

部分模型对 `content` 形态有额外要求。例如 `glm-4-voice` 只接受列表形式的 content，传字符串会返回 `error code 1210`：

```python
# ✗ 报错
{"role": "user", "content": "你好"}
# ✓
{"role": "user", "content": [{"type": "text", "text": "你好"}]}
```

这意味着 `flexllm ask` 等默认传字符串的入口无法直接用于这类模型。
