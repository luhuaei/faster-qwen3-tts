# Request-Level Seed, CUDA Graph, And Reproducibility FAQ

本文档记录本仓库当前 Orin 部署实现中，请求级 `seed`、CUDA graph 内外采样、可复现性、EOS 行为与常见调参问题的完整说明。

适用范围：

- 仓库分支：`lzc-aipod-v2`
- 已验证提交：`8140f2f`
- 已验证镜像：`registry.lazycat.cloud/x/faster-qwen3-tts:0.6b-custom-openai-orin-v11`
- 主要对应代码：
  - [examples/openai_server.py](../examples/openai_server.py)
  - [faster_qwen3_tts/predictor_graph.py](../faster_qwen3_tts/predictor_graph.py)
  - [faster_qwen3_tts/model.py](../faster_qwen3_tts/model.py)
  - [faster_qwen3_tts/sampling.py](../faster_qwen3_tts/sampling.py)
  - [tests/test_sampling.py](../tests/test_sampling.py)
  - [tests/test_openai_server.py](../tests/test_openai_server.py)

## 1. 结论摘要

当前最终实现采用的是“图内 RNG”和“图外 RNG”拆开的方案：

- CUDA graph 内部采样继续使用默认 CUDA RNG
- CUDA graph 外部采样使用独立的 `eager_generator`
- 每个请求开始前：
  - 用 `torch.cuda.manual_seed(seed)` 重置图内默认 CUDA RNG
  - 用 `eager_generator.manual_seed(seed + 1)` 重置图外 RNG
- `do_sample=False` 时不走随机采样，`seed` 不影响生成结果
- 服务端使用 `_model_lock` 串行化 GPU 推理，避免不同请求的 RNG 消耗互相穿插

这个设计的目标不是“让 CUDA graph 在 capture 时把结果固定死”，而是：

- 保留 CUDA graph 的性能收益
- 允许按请求动态传入 `seed`
- 让同一个请求在同一镜像版本、同一模型和同一音色配置下可复现
- 避免图内和图外采样互相污染 RNG 状态

## 2. 问题背景

这次实现要解决的核心问题有四类：

1. `do_sample=True` 时，多次请求同一句文本会出现明显差异，甚至音高、节奏、长度都不一致。
2. 在 Orin 上启用 CUDA graph 后，请求级采样参数和请求级随机种子很难直接作用到图内采样。
3. 早期尝试把请求级 CUDA `generator` 直接传进 graph 内部采样，运行时会报错：

```text
RuntimeError: Attempt to increase offset for a CUDA generator not in capture mode.
```

4. 用户需要一个真正的请求级 `seed`，并且要能通过 OpenAI API 直接传递。

## 3. 当前实现

### 3.1 OpenAI API 暴露 `seed`

当前 [examples/openai_server.py](../examples/openai_server.py) 中的 `SpeechRequest` 已暴露：

- `model`
- `input`
- `voice`
- `response_format`
- `speed`
- `instruct`
- `language`
- `seed`

其中：

- `speed` 当前仍然是兼容字段，但未实际生效
- `seed` 会一路传到 `generate_custom_voice()` / `generate_custom_voice_streaming()` / `generate_voice_clone()` / `generate_voice_clone_streaming()`

OpenAI 请求示例：

```json
{
  "model": "tts-1",
  "input": "欢迎使用 Qwen3TTS 服务。",
  "voice": "vivian",
  "response_format": "wav",
  "language": "Chinese",
  "instruct": "请用温和、清晰的语气朗读。",
  "seed": 1234
}
```

### 3.2 请求开始时如何配置 RNG

入口在 [faster_qwen3_tts/model.py](../faster_qwen3_tts/model.py) 的 `_configure_request_generator()`。

它的行为是：

1. 调用 `predictor_graph.set_request_seed(seed)`
2. 返回 `predictor_graph.eager_generator`
3. 后续图外的 `sample_logits(..., generator=request_generator)` 都共用这个 `eager_generator`

真正的 seed 设置逻辑在 [faster_qwen3_tts/predictor_graph.py](../faster_qwen3_tts/predictor_graph.py)：

```python
def set_request_seed(self, seed=None):
    if seed is None:
        with torch.cuda.device(self.device_index):
            torch.cuda.seed()
        self.eager_generator.seed()
    else:
        base_seed = _normalize_request_seed(seed)
        with torch.cuda.device(self.device_index):
            torch.cuda.manual_seed(base_seed)
        self.eager_generator.manual_seed((base_seed + 1) & ((1 << 63) - 1))
```

也就是说：

- 图内默认 CUDA RNG 用 `seed`
- 图外独立 `eager_generator` 用 `seed + 1`

补充一点：

- 请求传入的 seed 会先经过 `_normalize_request_seed()`
- 当前实现会把它约束到稳定的无符号 63-bit 范围
- 这是为了避免极端大整数、负数或跨环境整型差异带来的不稳定行为

### 3.3 图内采样和图外采样分别在哪里

这个区分非常关键。

图内采样：

- 发生在 [faster_qwen3_tts/predictor_graph.py](../faster_qwen3_tts/predictor_graph.py) 的 `_full_loop()`
- 这里的 `sample_logits()` 运行在 CUDA graph capture/replay 的那条路径里
- 这里没有再显式传 `generator=...`
- 因此它使用默认 CUDA RNG

图外采样：

- 发生在 [faster_qwen3_tts/generate.py](../faster_qwen3_tts/generate.py)
- 也发生在 [faster_qwen3_tts/streaming.py](../faster_qwen3_tts/streaming.py)
- 主要包括：
  - prefill 后第一枚 token 的采样
  - talker decode 每一步下一枚 token 的采样
- 这里显式使用 `sample_logits(..., generator=request_generator)`
- `request_generator` 就是 `_configure_request_generator()` 返回的 `eager_generator`

简单理解：

- predictor graph 内部 15 个 codebook token 的采样，属于图内 RNG
- talker 主循环上的 token 采样，属于图外 RNG

## 4. 为什么这么设计

### 4.1 为什么不把一个请求级 CUDA `generator` 直接塞进 graph 内

这是最早尝试过的方向，但在 Orin 当前这套 PyTorch/CUDA graph 组合上不稳，实际触发了：

```text
RuntimeError: Attempt to increase offset for a CUDA generator not in capture mode.
```

原因可以概括为：

- CUDA graph 会捕获一条固定的 GPU 执行路径
- `torch.multinomial(..., generator=some_cuda_generator)` 会推进该 generator 的 offset
- 如果这个 generator 的状态不是以 graph capture 所需的方式管理，replay 时推进 offset 会失败
- 在当前环境里，直接把请求级 CUDA generator 塞进图内采样，不是稳定方案

理论上，可以继续走更复杂的 graph-safe generator 状态管理路线，比如更细粒度地处理 capture/replay 前后的 graph-safe state；但在这次 Orin 镜像和运行时组合上，简单直接地这么做没有得到稳定可运行结果。

### 4.2 为什么图内改成默认 CUDA RNG

因为默认 CUDA RNG 天然和当前 CUDA graph replay 机制兼容得更好。

关键点：

- graph capture 固定的是执行图和内存关系
- 不是把某次采样出来的 token 结果固化在图里
- `torch.multinomial` 仍然会在每次 replay 时执行
- 它会在 replay 时从当前默认 CUDA RNG 状态里取随机数

因此：

- 只要在请求开始前把默认 CUDA RNG 重置到同一个状态
- 同一个输入 replay 时就会沿着同一条随机路径采样
- 从而得到同样的图内 token

这就是“图内 do_sample=True 也能按请求复现”的根本原因。

### 4.3 为什么图外还要单独用一个 `eager_generator`

因为如果图内和图外都共享同一个 RNG 流，会有两个问题：

1. RNG 消耗强耦合
2. 维护困难

更具体地说：

- 图内 predictor 每次会消耗一批随机数
- 图外 talker 主循环每步也会消耗随机数
- 只要任意一侧的调用次数、提前结束时机、异常中断路径有变化，另一侧的 RNG 序列也会跟着偏移

后果是：

- 你看起来只改了图外逻辑，图内结果也会变
- 或者图内提前结束了几步，图外后续采样序列就全变了

所以最终实现选择把它们拆成两条独立 RNG 流：

- 图内：默认 CUDA RNG
- 图外：`eager_generator`

这样做的好处：

- 图内 replay 的随机序列只受图内自身消耗影响
- 图外采样的随机序列只受图外自身消耗影响
- 两边可以用同一个请求种子稳定初始化，但彼此不串扰

### 4.4 为什么图外用 `seed + 1`

这是为了从同一个请求种子稳定派生出第二条 RNG 流。

如果图内和图外都直接用完全相同的 seed 初始化，也不是绝对错误，但会引入不必要的相关性。使用 `seed` 和 `seed + 1` 的好处是：

- 每个请求只需要传一个 `seed`
- 图内、图外都能从它稳定初始化
- 两条随机流彼此独立
- 同一个请求再次执行时，仍然会得到同样的图内和图外序列

### 4.5 为什么不为每个请求重新 capture 一次 graph

因为这会直接失去 CUDA graph 的主要价值。

重新 capture 的问题：

- 开销大
- 启动和首请求延迟显著变高
- 逻辑复杂
- 没必要

当前方案只在服务启动或首次 warmup 时 capture 一次，后续每个请求只重置 RNG 状态，这样既保留了性能，也保留了请求级复现能力。

## 5. 基本原理

### 5.1 CUDA graph 里采样是在 capture 时就固定了吗

不是。

要区分“图结构固定”和“随机结果固定”：

- capture 时固定的是 GPU kernel 执行顺序、静态内存、张量形状和调用路径
- 不是把那次 `torch.multinomial` 抽到的 token 值永久写死
- replay 时仍然会重新执行采样
- 采样结果取决于 replay 时的 RNG 状态和输入 logits

所以正确说法是：

- graph 内的采样行为会受到 CUDA graph 影响
- 但它不是在预编译时就决定结果
- 它仍然是 replay-time 动态生效的

### 5.2 为什么“把 graph 重置成默认 RNG，再按请求 seed 重置”基本可以

因为 replay 真正依赖的是“当前默认 CUDA RNG 状态”，而不是“capture 时的临时随机结果”。

只要下面几个条件同时满足：

- 输入文本一致
- voice 一致
- language 一致
- instruct 一致
- 模型权重一致
- 镜像版本一致
- 请求前把默认 CUDA RNG 重置到相同状态

那么 graph 内 replay 就会沿着同样的随机路径执行。

### 5.3 为什么 `do_sample=False` 时 seed 不重要

因为 [faster_qwen3_tts/sampling.py](../faster_qwen3_tts/sampling.py) 里：

```python
if not do_sample:
    return torch.argmax(logits, dim=-1)
```

这条路径不调用 `torch.multinomial`，因此：

- 不消费 RNG
- 不依赖 seed
- 只要输入一样，就应该稳定输出一样的 token

这也是为什么你观察到 `do_sample=False` 时生成出来的音频 md5 一样。

## 6. Orin 上的实际验证

### 6.1 直接模型调用验证

在 Orin 上对同一个输入重复执行：

- 相同文本
- 相同 voice
- 相同参数
- 相同 `seed=1234`

得到的结果为：

```text
0 74880 4b83d57b8430a9a38fbd289c49c91982
1 74880 4b83d57b8430a9a38fbd289c49c91982
2 74880 4b83d57b8430a9a38fbd289c49c91982
```

可见：

- 音频长度一致
- md5 一致

### 6.2 OpenAI HTTP 接口验证

对 `/v1/audio/speech` 重复请求同一个 body，并固定 `seed=1234`：

```text
/tmp/seed1.wav 78b75e46c38b755eb660506e597e21c5 149804
/tmp/seed2.wav 78b75e46c38b755eb660506e597e21c5 149804
/tmp/seed3.wav 78b75e46c38b755eb660506e597e21c5 149804
```

可见：

- 音频字节长度一致
- md5 一致

### 6.3 单元测试覆盖

当前仓库里也加入了对应测试：

- [tests/test_sampling.py](../tests/test_sampling.py)
  - `test_sample_logits_respects_explicit_generator_seed`
  - `test_predictor_graph_request_seed_splits_graph_and_eager_generators`
  - `test_min_new_tokens_suppresses_early_eos`
- [tests/test_openai_server.py](../tests/test_openai_server.py)
  - 验证 `seed` 已从 OpenAI 请求正确传递到生成函数

## 7. 如何判断 seed 是否真的生效

不要只听主观感觉，也不要只看一次波形。建议同时看三类信号。

### 7.1 先做最小闭环测试

固定以下字段不变：

- `input`
- `voice`
- `language`
- `instruct`
- `response_format`
- `seed`

然后重复请求 3 次以上，比较：

- 文件字节长度
- `md5sum`
- 主观听感

如果 seed 生效，在同一镜像版本上，`wav` 结果通常应该：

- 长度一致
- md5 一致
- 听起来一致

### 7.2 如果听感不一样，优先排查什么

优先排查下面几项：

1. 你是不是实际打到了旧镜像，而不是 `v11`
2. 请求 body 有没有任何字段变化
3. `language` 或 `instruct` 是否一边传了、一边没传
4. 音色配置是否变过
5. 服务是否并发执行、没有串行锁
6. 是不是有一边根本没把 `seed` 传到服务端

### 7.3 为什么有时“看起来 seed 没生效”，其实是别的变量变了

TTS 对早期 token 极其敏感。

只要下面任一项变化：

- 文本里多一个空格
- `instruct` 语气描述不同
- `language` 从 `Auto` 改成 `Chinese`
- voice 配置切到了另一个默认值
- 模型权重或镜像版本升级

后续整条声学 token 路径都可能变化，表现为：

- 音高不同
- 节奏不同
- 句长不同
- 拖长和停顿位置不同

这不是 seed 失效，而是输入条件已经变了。

## 8. FAQ

### Q1. 如果 `do_sample=True`，`temperature=1.0`，`top_k=0`，`top_p=1.0`，会发生什么

这是最“完全采样”的状态：

- `temperature=1.0` 不额外压缩或放大 logits
- `top_k=0` 表示不做 top-k 截断
- `top_p=1.0` 表示不做 nucleus 截断
- `do_sample=True` 表示最终用 `torch.multinomial` 从完整 softmax 分布里抽样

结果就是：

- 随机性最大
- 结果差异最大
- 更容易出现同一句文本多次生成时声调、节奏、时长都变化

### Q2. 为什么 `do_sample=False` 也可能一直输出，拿不到 EOS

因为 `do_sample=False` 只是把“随机抽样”改成了“每一步取 argmax”。

如果模型当前在某些状态下：

- 持续把某个非 EOS token 打成最高分
- 或者重复结构比分支到 EOS 更占优

那么 greedy decode 会一直沿着这条路径走下去，直到：

- 终于遇到 EOS
- 或者触发 `max_new_tokens`
- 或者命中别的停止条件

所以：

- `do_sample=False` 不等于“一定很快收敛到 EOS”
- 它只是去掉随机性，不保证一定正确结束

### Q3. 在温和采样下，多次请求为什么会出现音高一会高、一会低

因为 TTS 是强自回归系统，前面少量 codec token 的变化会持续级联到后面。

一旦最早几步 token 不同：

- 后续音高轮廓会变
- 韵律会变
- 停顿点会变
- 句子时长会变

所以“温和采样”只是在统计上减弱波动，不等于每次都会接近同一个结果。

如果你要稳定复现：

- 用固定 `seed`
- 或者直接 `do_sample=False`

### Q4. `min_new_tokens>10`、`do_sample=False`、`repetition_penalty=1.1` 能不能解决问题

这组参数有帮助，但不是万能解。

它们分别解决的是不同问题：

- `min_new_tokens>10`：防止太早输出 EOS
- `do_sample=False`：去掉随机性
- `repetition_penalty=1.1`：抑制重复 token

适合的场景：

- 你想优先求稳定
- 某些音色会过早结束
- 某些音色会卡住重复

不能保证的部分：

- 不保证一定拿到 EOS
- 不保证一定没有拖长
- 不保证所有音色都最佳

它更像是“偏保守的稳定化配置”，不是所有问题的一键修复。

### Q5. 能不能做成请求级固定 seed

可以，当前实现已经支持。

OpenAI 请求体直接传：

```json
{
  "model": "tts-1",
  "input": "你好",
  "voice": "vivian",
  "response_format": "wav",
  "seed": 1234
}
```

### Q6. 如果 graph 内恢复成默认 RNG，然后图外使用请求级 seed，能生成同样的声音吗

要分情况。

如果 graph 内 `do_sample=True`，但你没有在请求开始前把默认 CUDA RNG 重置到同一个 seed，那么：

- 图内 predictor 仍然会随机
- 整体结果不能稳定复现

如果 graph 内 `do_sample=False`，图内不再消耗随机数，那么只控制图外 RNG 就基本可以复现图外那部分行为。

最终完整可复现的正确做法是：

- 图内默认 CUDA RNG 也按请求 seed 重置
- 图外 `eager_generator` 也按请求 seed 派生重置

### Q7. 如果 graph 内 `do_sample=False`，图外使用请求级 seed，能生成同样的声音吗

基本可以。

原因是：

- graph 内 argmax 不消耗随机数
- 剩下只有图外 talker 采样依赖 RNG
- 这时只要图外 generator 固定，同一请求就能复现

但前提仍然是：

- 输入一致
- 模型一致
- 服务版本一致

### Q8. “解析 1 中的基本可以的原理，CUDA graph 的采样不影响吗”

会影响，前提是 graph 内还在采样。

更准确地说：

- graph 本身不会把采样结果预先固定死
- 但 graph 内如果执行了 `torch.multinomial`
- 它当然会消费 RNG，并决定 codebook token

所以：

- graph 内 `do_sample=True` 时，图内 RNG 一定会影响结果
- graph 内 `do_sample=False` 时，图内 RNG 就不再影响结果

### Q9. 如果改回之前 graph 内 `do_sample=True` 的方案，采样会不会在 capture 时就确定

不会。

即使 graph 内 `do_sample=True`：

- capture 固定的是执行路径
- replay 时仍然会实际执行采样
- 结果由 replay 时的 RNG 状态决定

所以 graph 内采样是“动态生效”的，不是“编译时固定”的。

### Q10. graph 内 `do_sample=True` 时，正确接法是不是“graph 持有固定 CUDA generator，然后请求前 graphsafe_set_state”

这是一个理论上可行、但工程上更复杂的方向。

这次最终没有采用它作为交付方案，原因是：

- 在当前 Orin 环境上，直接把请求级 CUDA generator 接进 graph 内，实际跑出了 capture 相关报错
- 即使继续沿 graph-safe generator 状态路线深入，也会让实现明显更复杂、更脆弱

当前交付方案更简单稳妥：

- graph 内改用默认 CUDA RNG
- 请求开始前用 `torch.cuda.manual_seed(seed)` 重置图内 RNG
- 图外单独使用 `eager_generator`

这个方案在当前环境下已经验证可运行、可复现。

### Q11. 为什么之前会报 `Attempt to increase offset for a CUDA generator not in capture mode`

因为你试图在 CUDA graph 路径里推进一个不在正确 capture 语义下管理的 CUDA generator。

核心问题不是“不能有 generator”，而是：

- graph replay 要求 RNG 状态推进方式和 capture 时一致
- 显式 generator 的 offset 管理必须满足 graph 约束
- 当前实现直接这么接时，没有满足这个条件

所以 replay 推进 offset 时就失败了。

### Q12. 如何在 OpenAI API 上传固定 seed

直接在 `POST /v1/audio/speech` 的 JSON body 里加入 `seed`：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"你好","voice":"vivian","response_format":"wav","seed":1234}' \
  --output out.wav
```

### Q13. 为什么我固定了 seed，生成的文件还是不一样

先区分两种情况。

第一种，`do_sample=False`：

- 这时本来就不依赖 seed
- 只要输入一样，通常结果应该一致

第二种，`do_sample=True`：

如果结果还不一样，通常优先排查：

1. 是否真的跑在新版本实现上
2. 请求体是否完全一致
3. `language` 和 `instruct` 是否一致
4. 是否串行执行，还是有并发干扰
5. 是否实际把 `seed` 传到服务端
6. 是否比较的是不同格式或不同后处理结果

在当前 `v11` 和 `8140f2f` 上，同一请求固定 seed 已实测可复现。

### Q14. 我请求的是 WAV，听到的声音还是不一样，时长也不一样。是不是 seed 没生效，怎么确定

先不要只靠耳朵判断，建议这样确认：

1. 同一个 `curl` body 连续请求 3 次
2. 都保存成独立文件
3. 比较 `wc -c`
4. 比较 `md5sum`

示例：

```bash
md5sum seed1.wav seed2.wav seed3.wav
wc -c seed1.wav seed2.wav seed3.wav
```

如果：

- 字节数相同
- md5 相同

那就说明 seed 已经生效，而且生成结果完全一致。

如果：

- 长度不同
- 听感不同

那就说明至少有一个前提没满足，常见是：

- 不是同一镜像版本
- 请求参数并不完全一致
- 服务端不是这套实现

### Q15. 为什么 `do_sample=False` 时，生成出来的声音 md5 是一样的

因为它走的是 greedy argmax 路径，不消费随机数。

所以：

- seed 不参与
- RNG 不参与
- 只要 logits 一样，token 就一样
- token 一样，解码出来的音频自然也一样

### Q16. graph 内采样是在 CUDA graph 预编译时就确定，还是可以动态更改也能生效

结论很明确：

- 不是在预编译时就确定
- 是 replay 时根据当前 RNG 状态动态生效

这正是请求级 seed 能和 CUDA graph 共存的前提。

### Q17. 有了请求级 seed 以后，预构建好的 CUDA graph 还生效吗

仍然生效。

当前方案并没有因为请求级 seed 而放弃 CUDA graph。

实际运行方式是：

- 服务启动时先 warmup 并 capture graph
- 后续请求直接 replay 已捕获的 graph
- 每个请求只在 replay 前重置 RNG 状态
- 不需要为每个请求重新 capture

所以：

- CUDA graph 的性能收益还在
- 请求级 seed 也能生效
- 两者不是互斥关系

## 9. 设计取舍与限制

### 9.1 当前设计的优点

- 能保留 CUDA graph 性能收益
- 能支持请求级 `seed`
- 能在当前 Orin 环境下稳定运行
- 图内和图外 RNG 解耦，行为更可解释
- `do_sample=False` 与 `do_sample=True` 两条路径都清晰

### 9.2 当前设计的前提

- 同一镜像版本
- 同一模型权重
- 同一音色配置
- 同一请求参数
- 服务端串行执行推理

### 9.3 当前设计没有解决什么

- 它不让不同文本得到同样的音频
- 它不保证所有音色都自然
- 它不自动修复模型本身的 EOS 偏好问题
- 它不让旧版本镜像自动具备请求级 seed 复现能力

## 10. 维护建议

如果后续继续调整采样链路，建议坚持这几个原则：

1. 不要轻易把图内和图外重新绑到同一个 RNG 流上。
2. 如果要重新尝试 graph-safe explicit CUDA generator，先在 Orin 目标环境上验证 capture/replay 兼容性。
3. 每次改动后都做三类回归：
   - 直接模型调用复现测试
   - OpenAI HTTP 接口复现测试
   - `do_sample=False` 的 greedy 一致性测试
4. 文档和接口必须保持一致，尤其是 `seed` 是否真实对外暴露。

## 11. 一句话总结

当前交付方案的本质是：

- graph 内采样不在 capture 时固定结果
- 它在 replay 时读取当前默认 CUDA RNG
- 图外采样读取独立 `eager_generator`
- 每个请求用同一个 `seed` 同时重置两条 RNG 流
- 因而在保留 CUDA graph 的前提下，实现了请求级可复现
