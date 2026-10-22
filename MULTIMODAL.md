# EmbeddingGemma 2 多模态模式（v3.11.10）

本次按讨论后的范围实现 **LiteLLM → LiteRT-LM → EmbeddingGemma 2 440M**。SiliconFlow、火山方舟留待有实际服务时接入，不提供尚未实现的配置选项。legacy 仍为默认模式。

## 安装与配置

1. 在 AstrBot 停用插件，备份插件目录、配置及实际数据目录。数据目录由原有 StarTools / 兼容迁移逻辑决定，勿删除历史文件。
2. 使用本次 ZIP 更新插件，必须一并安装新增的 memory_embedding 目录。依赖沿用 requirements.txt，插件不加载模型、不引入 PyTorch。
3. 在插件配置填写下表，保存后重载。

| 配置 | 当前部署填写值 |
|---|---|
| embedding_backend | litert_lm |
| embedding_api_url | http://192.168.50.5:4000/v1 |
| embedding_api_key | 有 embeddinggemma-2-440m 权限的 LiteLLM Key |
| embedding_model | embeddinggemma-2-440m |
| embedding_dimensions | 768 |
| embedding_model_revision | local-v1 |
| embedding_max_concurrency | 1 或 2，默认 2 |
| multimodal_fusion_enabled | true |
| multimodal_max_images | 4 |
| multimodal_sticker_filter_enabled | false |
| multimodal_vision_fallback_enabled | true |

地址供 192.168.50.x 局域网客户端使用；ZeroTier 客户端可改用 http://10.147.17.5:4000/v1，本次实测走后者。可配置能直达的 LiteRT-LM Base URL，模型名须与入口实际名称一致。Key 仅配置面板掩码显示，不等于加密存储；源码、历史消息和测试结果不保存 Key。embedding_provider_id 仅用于旧后端。

4. 管理员发送 /嵌入测试，确认返回 768 维。此命令仅测文本，不代表图片通路已验收。
5. 在测试群发送文字、单图、多图加正文，再用 /搜图 公园新闻，使用 /记忆状态 查看当前后端、维度、空间向量数量和错误。
6. 提醒仍须满足不同发送者人数及历史时间条件。连续发送几条消息不能绕过原有时间要求；实际 QQ 引用、图片来源需安装后验收。

本地图片默认允许系统临时目录及 AstrBot 工作目录下 data/temp，其他适配器缓存目录添加到 multimodal_local_image_roots。私网图片服务须将实际主机名/IP 添加到 multimodal_allowed_image_hosts。API 地址不受此限制。图片下载不跟随重定向。

## 行为与兼容

- 文本查重使用 sentence similarity 前缀；每张图片直接编码。搜图查询使用 search query 前缀，复用同一份图片向量。
- 图文、多图可补充一个原生融合向量。已实测 2、4 张图输入，并验证最后一张图影响结果。最多融合 4 张；逐图上限可设为 1—8，超过 4 张时跳过融合。
- 多图要求数量相同、全部图片一一匹配，正文也需匹配。融合分数辅助排序，不能绕过正文或图片覆盖要求。
- 图片缺失、下载失败、数量超限时，不判定整条重复；可用图片仍能搜索，并保留原消息中的位置。
- 第一阶段由文本、逐图、融合、跨模态向量及精确指纹/dHash召回 Top-K；第二阶段验证全部图片和正文，以通过验证的最高分记录为事件锚点。人数遍历全部匹配记录，并要求同时匹配当前消息及锚点；Top-K 不是 ANN 索引，不截断人数统计；K 过小可能让未通过完整验证的高分候选挤掉有效候选。dHash 只召回、不单独确认。
- 默认 3 名不同发送者，同发送者排除，以及“现在减去命中历史消息时间至少 3600 秒”保持原逻辑，没有改成最近提醒时间的冷却。
- enable_llm_judge、judge_provider_id 沿用原配置。上下文判断模型不会被告知它看过未提供的历史图片。
- 按本次讨论，semantic_auto_remind 默认 true：关闭 LLM 后，符合完整内容匹配、人数及时间条件的语义近似也可提醒。改为 false 后，无 LLM 时仅精确重复提醒。
- 普通图片默认调用一次视觉 LLM 生成并保存描述（multimodal_vision_fallback_enabled=true），提醒消息会显示历史图片描述，便于人工判断是否误判。该调用需要 vision_provider_id，可能产生费用；关闭后普通图片不调用视觉 LLM。开启表情包过滤时也会逐图调用；识别失败不当作表情包，也不丢弃图片向量。
- URL、视频、合并转发沿用原精确指纹/视频路径，占位文本不送入新后端。
- 新特征写入 multimodal/schema=1；旧 embedding 不覆盖，新记录的旧 embedding 为 null。保留文件 SHA-256、像素 SHA-256、dHash；新模式不以 dHash 单独决定重复。
- 不持久保存原图/Base64。缺少原图的历史图片不强行补建向量；搜图保留旧描述关键词/字符匹配兜底。
- 模型、配置修订、端点、预处理、维度共同隔离语义空间，旧文本向量不与新图片向量混用。精确文件/像素匹配允许跨空间。
- local-v1 是配置修订，不是自动识别的权重版本。替换网关模型权重后必须更新 embedding_model_revision；修改端点地址也会隔离空间。
- 日分片 JSON/Gzip 及保留期清理沿用原机制。切回 legacy 并重载即可回到旧处理链，但新记录不会因此自动获得旧文本向量。
- 完整回滚时停用插件，恢复版本与配置备份。原始 Git 基线 ad9d446，实施前检查点 524844f；没有推送远端、清空数据或部署到运行中的 AstrBot。

## 初始阈值及实测边界

| 用途 | 默认余弦阈值 |
|---|---:|
| 文本重复 | 0.90 |
| 图片重复 | 0.94 |
| 图片描述语义重复 | 0.80 |
| 文字与单图跨模态重复 | 0.94 |
| 文字搜图 | 0.50 |

这些是待真实群聊标注集校准的初始值。语义图片匹配还会比较双方图片描述的文本向量；描述相似度低于 0.80 时拒绝匹配。旧记录没有描述向量时退回 LLM judge 比较描述。正文中可见数字不一致仍会拒绝，默认没有独立 OCR，不能可靠检测图片中的数字差异。

| 真实服务合成截图样本 | 图片余弦相似度 |
|---|---:|
| 同图缩小、JPEG 重压缩 | 0.9873 |
| 同版式新闻只改金额、日期 | 0.9839 |
| 相同新闻版式、不同主题 | 0.8242 |

0.94 阈值会把前两者均判为相似，第二项是已知误报。三个标注对结果 TP=1、FP=1、TN=1、FN=0，仅用于暴露边界，不代表实际准确率。搜图查询对目标图得分 0.8053，目标排名第一；查重文本前缀对该图约 0.7918，未通过默认跨模态查重阈值。

文本编码默认预算 900 UTF-8 字节，约 300 汉字；超限保留全文与精确匹配，不静默截断。embedding_max_text_bytes 可调，但当前服务总输入预算 1024 tokens，字节数不等于 token 数，增大预算可能导致拒绝请求。

单图默认 10 MiB、2000 万像素，最长边缩至 1024，EXIF 校正、RGB、JPEG90，动图仅第一帧。请求有并发、超时、响应大小限制；网络错误/429/部分5xx有限重试，401不重试。验证一维数值向量、维度、有限值及非零范数，统一 L2 归一化。内存向量缓存最多256项、600秒。

## 验证与复现

离线测试 104 项通过，其中原有回归 40 项。覆盖协议、错误、重试/缓存、空间隔离、多图匹配、缺图/超限、无描述搜图、图片访问限制、并发、群范围/人数/时间、压缩存储往返、旧精确路径及命令过滤。

在仓库目录运行：

~~~powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
~~~

真实后端测试：

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:EMBEDDING_API_BASE = "http://192.168.50.5:4000/v1"
$env:EMBEDDING_API_KEY = "<你的 LiteLLM Key>"
.venv\Scripts\python tests/live_multimodal.py --output tests/live_multimodal_result.json
Remove-Item Env:EMBEDDING_API_KEY
~~~

结果保存在 tests/live_multimodal_result.json。单图提取约0.80—0.99秒，仅此次少量串行样本耗时，不是长期负载/并发吞吐结论。

可自行创建以下 JSON 标注集，计算 TP/FP/TN/FN、precision、recall、误报率及编码中位耗时：

~~~json
{
  "messages": [
    {"id": "a", "text": "", "images": ["fixtures/original.png"]},
    {"id": "b", "text": "", "images": ["fixtures/resized.jpg"]}
  ],
  "pairs": [{"a": "a", "b": "b", "duplicate": true}],
  "config": {"multimodal_image_threshold": 0.94}
}
~~~

~~~powershell
.venv\Scripts\python tests/evaluate_multimodal.py dataset.json --online --output result.json
~~~

--online 使用上述环境变量，图片相对数据集目录解析。不加 --online 时每条消息需含已编码的 multimodal 字段，可离线调阈值。输出只含ID、分数、判断及指标，不含正文/凭据；评测仅覆盖内容匹配，不模拟人数及 LLM 提醒全过程。

未验证：真实 AstrBot/QQ 加载、消息引用、图片路径、真实群聊准确率、长期负载。SiliconFlow 与火山方舟本期未实现、未测试。

## 文件清单

| 文件 | 用途 |
|---|---|
| main.py | 普通多模态消息、搜图、连接测试、状态入口 |
| _conf_schema.json、metadata.yaml | 配置及版本 |
| memory_embedding/provider.py | HTTP 协议、任务前缀、向量校验、缓存 |
| memory_embedding/images.py | 加载、预处理、指纹 |
| memory_embedding/manager.py | 逐图/融合编码、生命周期 |
| memory_embedding/matching.py | 匹配、空间隔离、搜图 |
| memory_embedding/plugin.py、__init__.py | AstrBot 集成、包入口 |
| tests/test_multimodal.py、test_multimodal_safety.py | 新增测试 |
| tests/live_multimodal.py、live_multimodal_result.json | 真实测试及结果 |
| tests/evaluate_multimodal.py | 标注集评测 |
| README.md、MULTIMODAL.md | 配置、验证、升级回滚说明 |


## 多模态调试日志

在插件配置开启 multimodal_debug_log（默认 false），保存并重载插件。日志以 INFO 级别输出，在 AstrBot 日志中搜索 [MultimodalDebug]，无需切换全局 DEBUG；若全局日志过滤高于 INFO，则需要允许显示 INFO。

每条处理链使用随机 trace 串联：start → image_load/text_embedding/image_embedding/fusion_embedding → features → candidate → matching → gates → llm_judge → decision。搜图使用 search，记录结果数、前10项分数、是否退回关键词搜索及接口错误。逐图 image_index 从0开始。编码耗时包括缓存命中时的调用耗时，不等于纯模型推理耗时。

vision 阶段新增 classified_sticker（模型是否判为表情包）、raw_is_sticker（模型原始字段值，最多20字符）、json_ok（是否成功解析 JSON）、sticker_source（is_sticker/type/parse_failed/empty_response/error 等）、content_chars、response_chars，以及 response_preview（原始返回截断500字符）和 content_preview（解析后的 content 截断300字符）。预览会遮盖已配置的 API Key、URL/Base64 资源，但可能包含图片描述、OCR 文字或聊天内容，仅排查时开启。

candidate 记录召回分数、确认分数、文本/图片配对分数和拒绝原因；只记录召回 Top-K，不输出全部历史。常见拒绝原因：numbers_conflict（正文数字冲突）、image_coverage_or_threshold（图片不能完整匹配或未达阈值）、description_below_threshold（图片描述语义不一致）、caption_presence_mismatch（正文有无不同）、caption_below_threshold、different_vector_space。语义图片匹配会输出 description_score；旧记录缺少描述向量时，在 confirmed 后按需计算并输出 description_gate（candidates/checked/rejected/scores）。matching 的 recalled=0 表示没有进入 Top-K 的候选，可能是保留期、空间、完整性或召回阈值条件不符，不等于服务故障。

gates 展示不同发送者人数、同发送者条件、历史年龄及冷却秒数；decision 展示最终是否计划提醒。该日志不证明消息已成功送达。/记忆状态 同时显示调试开关状态。

日志不输出完整向量、API Key、图片路径或完整原始回复。只保存到 AstrBot 日志，诊断字段在存储消息前移除。vision 的 preview 字段是截断后的图片描述/模型返回，可能转述聊天或图片文字；排查结束后关闭开关，避免大量输出。

### LLM 判断理由（v3.11.2）

开启同一个 multimodal_debug_log 开关后，llm_judge 日志新增 status 和 reason。正常 JSON 回复显示模型的 reason，最多500字符，并遮盖已配置的 Embedding Key、常见 Key/Bearer 及 URL/Base64 资源。理由可能转述聊天内容，因此仅排查时开启。不记录完整提示词或完整模型回复。

status 可区分 json_response、judge_provider_not_configured、judge_provider_not_found、empty_response、json_parse_failed_text_fallback 和 judge_request_failed；非正常 JSON 时不编造模型理由。不改变原有提醒判定。

### 多模态上下文判断修复（v3.11.3）

近期精确复读统计改用完整正文及全部图片指纹；不同图片不能因为同为 [图片×1] 就计为复读。语义近似不计入精确复读；旧记录缺少特征时不能靠占位符认定精确重复。旧模式文本统计保持原样。

LLM 请求在模板展开后追加匹配类型、完整消息匹配分数、图片数量及覆盖、正文情况和语义阈值。自定义 judge_prompt 也会收到这段证据；分数不是事实相同的概率。文字与单图跨模态会明确标注，不能误称两图逐张一致。仍不发送原图，只让 LLM 依据匹配证据及聊天上下文判断是否适合提醒。

MultimodalDebug 新增 judge_evidence 阶段，记录匹配类型/分数、图片数量和精确复读统计，不输出提示词正文。人数、冷却、同发送者排除及最终判断开关均不改变。已验证实际构造的请求内容（Mock），尚未用运行中的群聊重新验证模型判断效果。

### SnowLuma 视频兼容（v3.11.4）

SnowLuma 的 get_file 只解析图片/语音缓存，不能当作视频取回接口：[官方文档](https://snowluma.github.io/zh/docs/api/extended/get_file)。使用该后端时，将新配置“视频文件取回后端” video_file_backend 设置为 snowluma，保存并重载。插件跳过 get_file，直接使用消息中的本地可读路径或 URL，沿用大小校验、SHA-256及五帧处理。该配置对 legacy 和多模态模式的视频均有效。

默认 auto 保持原流程，但收到明确的 get_file only resolves cached image/voice ids 限制后，会按连接对象及账号缓存10分钟，期间跳过该视频取回接口，首次识别只输出 INFO。一般缓存未命中、retcode=100 或临时错误不会导致禁用。napcat 强制保持原 get_file 路径。缓存不写磁盘，重载清空。

同时支持原始视频段的可读 path，并在组件 file 为不透明标识时优先使用原始 file 中的 HTTP URL。只有文件名/资源ID而没有可读路径或可下载URL时仍标记 unavailable，不把普通视频ID猜成群文件ID。真实下载或来源错误仍会保留诊断，未掩盖失败。video_debug_log 可查看 get_file 的 skipped/unsupported 状态及 download 结果。

本次完成 Mock/本地文件回归；未连接用户运行中的 SnowLuma，实际消息URL及视频下载仍需安装后验证。前述 LLM / 多模态匹配规则未改。

### 图片多来源回退（v3.11.5）

当前 litert_lm 多模态路径按每张图片保留组件 path/url/file，随后追加同序原始 image 段的 path/url/file，最多6个并去重；仅当原始图片数与组件数相同才按顺序补充，避免错绑。路径缺失、解码失败、HTTP错误或权限限制后尝试同图下一个来源，成功即停止，只编码一次，不跨图片借用来源。全部失败标记该图缺失，其他图片仍可搜图。

候选共享 embedding_timeout 总时间预算，保留原图片大小/像素/私网地址/目录限制，不因回退绕过这些限制。MultimodalDebug 的 image_source_attempt 展示候选编号、类型、耗时、结果，不输出路径/URL。旧 legacy 视觉转述路径本次未调整，也未增加 get_msg 刷新过期URL。本版安装包同时补齐原有提醒图片 1000101866.jpg。

### 统一 legacy 图片加载（v3.11.6）

普通图片的 legacy 与 litert_lm 路径现在共用来源收集及 ImageLoader，覆盖同图 path/url/file 和原始消息段回退。legacy 将成功加载并预处理的图片送入视觉模型，使用同次加载的 dHash；图片数据只驻留本次处理内存，不存入历史记录。识图返回表情包或失败时不改用同图其他地址重复识图。legacy 不创建 embedding provider，也无需 embedding API 配置。

两种模式均遵循现有 multimodal_allowed_image_hosts、multimodal_local_image_roots、图片大小/像素和超时设置。legacy 的新图片哈希经过 EXIF 方向校正；旧历史哈希不改写，带旋转标记的旧图片可能与新哈希不同。未新增 get_msg 刷新URL，未改变视频和合并转发处理。

### 提醒展示图片描述（v3.11.7）

多模态模式默认开启 `multimodal_vision_fallback_enabled`：每张成功加载的图片会额外调用一次 `vision_provider_id`，生成描述并随消息保存。描述写入逐图字段 `multimodal.images[].description`，聚合值仍写入 `image_description`，旧版内嵌 `[图片内容: ...]` 继续可读。

重复提醒在历史消息正文后追加 `🖼️ 图片内容：...`；多图按 `图1：`、`图2：` 标注，单图描述最多120字符、总体最多400字符，超出截断。这样无需打开原图也能判断匹配是否误报。`/搜图` 命中多图记录时优先显示被命中那张图的描述。

关闭 `multimodal_vision_fallback_enabled` 后新图片不再调用视觉模型；没有描述的历史记录仍只显示 `[图片×N]`。升级前已保存的多模态记录没有原图，无法补生成描述，提醒会标注“未保存（旧记录或识图未成功）”；新记录不受影响。视觉调用失败不影响图片向量、查重和搜图，只缺少该图描述。

### 表情包提示词调整（v3.11.8）

默认 `vision_prompt` 改为按图片主要用途判断：新闻、截图、数据、公告、教程、具体事件叙述及普通实拍照片记录；网络表情包、反应图、单格梗图即使带文字也跳过。旧版“不确定就判为内容图”的偏向已移除，避免带字表情包被反复放过。

如果配置里自定义过 `vision_prompt`，该字段会覆盖内置默认值；要使用新版默认提示词，请清空该配置后重载。提示词只能降低视觉模型在边界样本上的摇摆，不能保证同一张图每次返回完全一致；如需对重复表情包做确定性过滤，仍需额外的指纹缓存。

### 视觉判定日志与回调兼容（v3.11.9）

`vision` 阶段现在输出模型原始判定：`classified_sticker`、`raw_is_sticker`、`json_ok`、`sticker_source` 和 `content_chars`。若日志中完全没有 `vision` 阶段，优先检查是否只替换了 `plugin.py` 或 `manager.py` 其中一个文件，以及是否有 `_multimodal_vision() takes ...` 之类的 TypeError。

`manager.py` 同时兼容只接收 `uri` 的旧版 vision 回调。只替换部分文件时不会再因参数数量不匹配而中断，但只有 `plugin.py` 与 `manager.py` 同时更新才会出现新增的判定字段。

### 视觉返回预览（v3.11.10）

`vision` 阶段新增 `response_preview`（模型原始返回，最多500字符）、`content_preview`（解析后的 content，最多300字符）和 `response_chars`。预览会遮盖已配置的 API Key、URL/Base64 资源，但可能包含图片描述、OCR 文字或聊天内容，因此只在排查时开启 `multimodal_debug_log`，排查后立即关闭。

通过预览可以直接区分：模型明确回了 `is_sticker: false`、模型返回了无法解析的 JSON、或模型没有返回 `is_sticker` 字段。若确认是模型明确判 false，应调整 `vision_prompt` 或更换视觉模型，而不是继续改解析逻辑。
