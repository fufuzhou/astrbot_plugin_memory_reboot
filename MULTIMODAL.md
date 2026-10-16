# EmbeddingGemma 2 多模态模式（v3.11.0）

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
| multimodal_vision_fallback_enabled | false |

地址供 192.168.50.x 局域网客户端使用；ZeroTier 客户端可改用 http://10.147.17.5:4000/v1，本次实测走后者。可配置能直达的 LiteRT-LM Base URL，模型名须与入口实际名称一致。Key 仅配置面板掩码显示，不等于加密存储；源码、历史消息和测试结果不保存 Key。embedding_provider_id 仅用于旧后端。

4. 管理员发送 /嵌入测试，确认返回 768 维。此命令仅测文本，不代表图片通路已验收。
5. 在测试群发送文字、单图、多图加正文，再用 /搜图 公园新闻，使用 /记忆状态 查看当前后端、维度、空间向量数量和错误。
6. 提醒仍须满足不同发送者人数及历史时间条件。连续发送几条消息不能绕过原有时间要求；实际 QQ 引用、图片来源需安装后验收。

本地图片默认允许系统临时目录及 AstrBot 工作目录下 data/temp，其他适配器缓存目录添加到 multimodal_local_image_roots。私网图片服务（如 NapCat）须将实际主机名/IP 添加到 multimodal_allowed_image_hosts。API 地址不受此限制。图片下载不跟随重定向。

## 行为与兼容

- 文本查重使用 sentence similarity 前缀；每张图片直接编码。搜图查询使用 search query 前缀，复用同一份图片向量。
- 图文、多图可补充一个原生融合向量。已实测 2、4 张图输入，并验证最后一张图影响结果。最多融合 4 张；逐图上限可设为 1—8，超过 4 张时跳过融合。
- 多图要求数量相同、全部图片一一匹配，正文也需匹配。融合分数辅助排序，不能绕过正文或图片覆盖要求。
- 图片缺失、下载失败、数量超限时，不判定整条重复；可用图片仍能搜索，并保留原消息中的位置。
- 第一阶段由文本、逐图、融合、跨模态向量及精确指纹/dHash召回 Top-K；第二阶段验证全部图片和正文，以通过验证的最高分记录为事件锚点。人数遍历全部匹配记录，并要求同时匹配当前消息及锚点；Top-K 不是 ANN 索引，不截断人数统计；K 过小可能让未通过完整验证的高分候选挤掉有效候选。dHash 只召回、不单独确认。
- 默认 3 名不同发送者，同发送者排除，以及“现在减去命中历史消息时间至少 3600 秒”保持原逻辑，没有改成最近提醒时间的冷却。
- enable_llm_judge、judge_provider_id 沿用原配置。上下文判断模型不会被告知它看过未提供的历史图片。
- 按本次讨论，semantic_auto_remind 默认 true：关闭 LLM 后，符合完整内容匹配、人数及时间条件的语义近似也可提醒。改为 false 后，无 LLM 时仅精确重复提醒。
- 普通图片不调用视觉 LLM。开启表情包过滤或图片描述后，才逐图额外调用 vision_provider_id；识别失败不当作表情包。额外调用可能产生费用。
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
| 文字与单图跨模态重复 | 0.94 |
| 文字搜图 | 0.50 |

这些是待真实群聊标注集校准的初始值。正文中可见数字不一致会拒绝匹配，但默认没有 OCR，不能检测图片中的数字差异。

| 真实服务合成截图样本 | 图片余弦相似度 |
|---|---:|
| 同图缩小、JPEG 重压缩 | 0.9873 |
| 同版式新闻只改金额、日期 | 0.9839 |
| 相同新闻版式、不同主题 | 0.8242 |

0.94 阈值会把前两者均判为相似，第二项是已知误报。三个标注对结果 TP=1、FP=1、TN=1、FN=0，仅用于暴露边界，不代表实际准确率。搜图查询对目标图得分 0.8053，目标排名第一；查重文本前缀对该图约 0.7918，未通过默认跨模态查重阈值。

文本编码默认预算 900 UTF-8 字节，约 300 汉字；超限保留全文与精确匹配，不静默截断。embedding_max_text_bytes 可调，但当前服务总输入预算 1024 tokens，字节数不等于 token 数，增大预算可能导致拒绝请求。

单图默认 10 MiB、2000 万像素，最长边缩至 1024，EXIF 校正、RGB、JPEG90，动图仅第一帧。请求有并发、超时、响应大小限制；网络错误/429/部分5xx有限重试，401不重试。验证一维数值向量、维度、有限值及非零范数，统一 L2 归一化。内存向量缓存最多256项、600秒。

## 验证与复现

离线测试 81 项通过，其中原有回归 40 项。覆盖协议、错误、重试/缓存、空间隔离、多图匹配、缺图/超限、无描述搜图、图片访问限制、并发、群范围/人数/时间、压缩存储往返、旧精确路径及命令过滤。

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
