# faster-qwen3-tts OpenAI API 文档

本文档对应当前仓库中的 [openai_server.py](/home/catdog/faster-qwen3-tts/examples/openai_server.py)，描述其对外暴露的 OpenAI 兼容 TTS 接口，以及流式音频调用方式。

关于请求级 `seed`、CUDA graph 内外采样、可复现性和常见调参问题的详细说明，见 [request-seed-cuda-graph-faq.md](./request-seed-cuda-graph-faq.md)。

## 概览

- 协议：HTTP
- 接口风格：OpenAI-compatible TTS
- 默认端口：`8000`
- 主要接口：
  - `GET /health`
  - `GET /v1/audio/voices`
  - `POST /v1/audio/speech`
  - `POST /v1/audio/voice-clone/pt`

适用场景：

- OpenWebUI
- 自定义 OpenAI-compatible client
- 直接通过 `curl` / Python / JavaScript 调用

## Base URL

本地容器直连示例：

```text
http://127.0.0.1:8000
```

算力舱部署后：

```text
https://qwen3-tts-ai.<你的应用域名>
```

## 1. 健康检查

### `GET /health`

用于判断服务是否已完成模型加载和 startup warmup。

请求：

```bash
curl http://127.0.0.1:8000/health
```

响应示例：

```json
{
  "status": "ok",
  "ready": true,
  "model_loaded": true,
  "mode": "custom",
  "voices": ["aiden", "dylan", "vivian"],
  "startup_warmup_enabled": true,
  "startup_warmup_completed": true,
  "startup_warmup_seconds": 15.11
}
```

字段说明：

- `status`: `ok` 表示服务已 ready，`starting` 表示仍在启动中
- `ready`: 是否可接收正式推理请求
- `model_loaded`: 模型是否已加载
- `mode`: 当前运行模式，`clone` 或 `custom`
- `voices`: 当前可用音色列表
- `startup_warmup_*`: 启动预热状态和耗时

判定 ready 的建议：

- 以 `ready == true` 作为服务可用标准

## 2. 查询可用音色

### `GET /v1/audio/voices`

返回当前服务可用 voice 列表。

请求：

```bash
curl http://127.0.0.1:8000/v1/audio/voices
```

响应示例：

```json
{
  "voices": [
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian"
  ],
  "default_voice": "vivian",
  "mode": "custom"
}
```

说明：

- `voice` 请求参数必须使用这里返回的名称
- 若请求里传入不存在的 `voice`，服务会尝试回退到 `default_voice`
- 若没有可回退的默认音色，则返回 `400`

## 3. 文本转语音

### `POST /v1/audio/speech`

这是核心的 OpenAI-compatible TTS 接口。

请求头：

```http
Content-Type: application/json
Accept: audio/wav
```

请求体：

```json
{
  "model": "tts-1",
  "input": "今天的风比昨天轻一点，适合慢慢说话。",
  "voice": "vivian",
  "response_format": "wav",
  "speed": 1.0,
  "instruct": "用比较轻柔、放松的语气来读。"
}
```

字段说明：

- `model`: 兼容字段，当前服务接受但不用于切换模型，建议固定传 `tts-1`
- `input`: 要合成的文本，不能为空
- `voice`: 音色名称，来自 `/v1/audio/voices`
- `response_format`: 支持 `wav`、`pcm`、`mp3`、`opus`
- `speed`: 当前版本仅接受该字段，但实际未生效
- `instruct`: 可选的自定义指令文本，用于控制语气、风格、节奏或口音倾向

`instruct` 的行为：

- 当前服务运行在 `custom` 或 `clone` 模式时可用
- 请求里传入 `instruct` 时，优先级高于服务端静态 voice 配置中的默认 `instruct`
- 请求里不传 `instruct` 时，沿用当前服务默认行为
- 对 Base voice cloning，`instruct` 可用，但在 `xvec_only=True` 的底层模式中仍应视为实验特性

返回行为：

- `wav`: 流式返回，`Content-Type: audio/wav`
- `pcm`: 流式返回，`Content-Type: audio/pcm`
- `mp3`: 非流式返回，`Content-Type: audio/mpeg`
- `opus`: 非流式返回，`Content-Type: audio/ogg`

### 3.1 使用 multipart 上传 `pt` 文件做动态声音克隆

在 `clone` 模式下，`/v1/audio/speech` 额外支持 `multipart/form-data`。

调用方可以在请求里上传一个 `voice_clone_pt` 文件，作为这一次合成请求的 speaker embedding。
这个 `pt` 文件由 `/v1/audio/voice-clone/pt` 生成，服务端不会替调用方持久化。

请求字段：

- 普通字段仍然沿用 `model`、`input`、`voice`、`response_format`、`instruct`、`language`、`seed`
- 新增文件字段：`voice_clone_pt`

行为说明：

- `voice_clone_pt` 只在服务运行于 `clone` 模式时可用
- 请求中携带 `voice_clone_pt` 时，优先级高于静态 voice 配置里的 `ref_audio` 或 `speaker_pt`
- 该模式底层走的是 `x-vector-only` speaker embedding 复用路径，适合调用方自己管理音色资产

示例：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -F "model=tts-1" \
  -F "input=今天我们直接复用上传的 speaker pt 来合成语音。" \
  -F "voice=alloy" \
  -F "response_format=wav" \
  -F "voice_clone_pt=@speaker.pt;type=application/octet-stream" \
  --output speech.wav
```

### 3.2 非流式调用

适合直接下载完整文件。

#### WAV 示例

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world.","voice":"vivian","response_format":"wav"}' \
  --output speech.wav
```

#### WAV + instruct 示例

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"今天我们试试更柔和一点的表达。","voice":"vivian","response_format":"wav","instruct":"请用温和、娓娓道来的语气朗读。"}' \
  --output speech.wav
```

#### MP3 示例

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world.","voice":"vivian","response_format":"mp3"}' \
  --output speech.mp3
```

#### Opus 示例

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world.","voice":"vivian","response_format":"opus"}' \
  --output speech.opus
```

说明：

- `mp3` 返回前会先生成完整音频，再进行编码
- `opus` 同样会先生成完整音频，再编码成 Ogg Opus
- 因此 `mp3` 和 `opus` 都不适合低延迟首包场景

## 4. 生成 speaker pt 文件

### `POST /v1/audio/voice-clone/pt`

该接口用于从参考音频中提取 speaker embedding，并直接返回一个 `.pt` 文件给调用方保存。

请求格式：

- `multipart/form-data`
- 必填文件字段：`ref_audio`
- 可选文本字段：`filename`

返回格式：

- `Content-Type: application/octet-stream`
- 响应体内容就是可复用的 `.pt` 文件

示例：

```bash
curl http://127.0.0.1:8000/v1/audio/voice-clone/pt \
  -F "ref_audio=@ref_audio.wav" \
  -F "filename=speaker.pt" \
  --output speaker.pt
```

说明：

- 当前接口只在服务运行于 `clone` 模式时可用
- 返回的是 x-vector speaker embedding，对应仓库里 `speaker.pt` 的复用方式
- 服务端不会帮调用方登记、命名或回收这些 `pt` 文件，调用方自行管理即可

## 5. 流式调用

### 流式语义

当前服务的“流式”不是 OpenAI Chat Completions 那种 SSE 事件流，而是：

- 直接返回音频字节流
- 使用 HTTP chunked transfer 逐块下发
- 只在 `response_format=wav` 或 `response_format=pcm` 时生效

具体行为：

- `wav`: 先返回一个流式 WAV 头，再持续返回 PCM16 音频块
- `pcm`: 直接持续返回裸 PCM16 音频块
- `mp3`: 不支持流式下发

### 5.1 curl 流式保存

```bash
curl -N http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"今天我们来测试流式语音返回。","voice":"vivian","response_format":"wav"}' \
  --output stream.wav
```

说明：

- `-N` 可以关闭 curl 的输出缓冲，更容易观察流式过程
- 输出文件会随着流返回逐步增长

### 5.2 Python 流式读取

```python
import requests

url = "http://127.0.0.1:8000/v1/audio/speech"
payload = {
    "model": "tts-1",
    "input": "今天我们来测试流式语音返回。",
    "voice": "vivian",
    "response_format": "wav",
    "instruct": "请用偏轻声、自然停顿的方式朗读。",
}

with requests.post(url, json=payload, stream=True, timeout=600) as resp:
    resp.raise_for_status()
    with open("stream.wav", "wb") as f:
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
                f.flush()
```

如果你要做边收边播：

- `wav` 模式下先解析开头 WAV header
- `pcm` 模式下按 `24kHz / mono / 16-bit little-endian` 直接播放

### 5.3 JavaScript 流式读取

```javascript
const resp = await fetch("http://127.0.0.1:8000/v1/audio/speech", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "tts-1",
    input: "今天我们来测试流式语音返回。",
    voice: "vivian",
    response_format: "wav",
    instruct: "请用平稳、自然的旁白语气朗读。"
  })
});

if (!resp.ok) {
  throw new Error(`HTTP ${resp.status}`);
}

const reader = resp.body.getReader();
const chunks = [];

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  chunks.push(value);
}

const blob = new Blob(chunks, { type: "audio/wav" });
```

## 6. 错误返回

常见错误：

### 400 参数错误

示例：

```json
{
  "detail": "response_format 'aac' not supported. Use: wav, pcm, mp3, opus"
}
```

或：

```json
{
  "detail": "'input' text is empty"
}
```

### 400 音色不存在

```json
{
  "detail": "Voice 'unknown' is not configured. Available voices: ['aiden', 'vivian']"
}
```

### 503 服务未就绪

```json
{
  "detail": "Model not loaded"
}
```

建议客户端处理：

- 启动后先轮询 `/health`
- 仅在 `ready=true` 后再调用 `/v1/audio/speech`

## 7. 与 OpenAI 官方接口的差异

当前实现是 OpenAI-compatible subset，不是完整的 OpenAI Audio API 实现。

已兼容的核心点：

- `POST /v1/audio/speech`
- `model / input / voice / response_format / speed` 这些常见字段
- 额外支持一个扩展字段：`instruct`
- 额外支持扩展接口：`POST /v1/audio/voice-clone/pt`
- 额外支持 `multipart/form-data` 上传 `voice_clone_pt`

当前差异：

- 不支持 `stream=true` 这种显式开关参数
- 流式返回是原始音频字节流，不是 SSE
- `speed` 字段当前未生效
- 多音色能力通过 `/v1/audio/voices` 暴露，不是 OpenAI 官方标准接口

## 8. 推荐调用方式

低延迟场景：

- 使用 `response_format=wav`
- 客户端按流式 chunk 读取

边播边收场景：

- 使用 `response_format=pcm`
- 客户端直接按 PCM16 播放

下载成品文件场景：

- 使用 `response_format=mp3`、`opus` 或 `wav`

## 9. 最小可用示例

### 先检查 ready

```bash
curl http://127.0.0.1:8000/health
```

### 再发起 TTS

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"欢迎使用 faster-qwen3-tts。","voice":"vivian","response_format":"wav","instruct":"请用清晰、亲切的语气介绍。"}' \
  --output out.wav
```
