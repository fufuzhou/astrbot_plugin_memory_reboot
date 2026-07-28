# ==============================================================================
# 第一部分：导入模块
# ==============================================================================

# ----- 1.1 Python 标准库 -----
import asyncio
import os
import re
import json
import time
import uuid
import hashlib
import datetime
import shutil
import gzip
import subprocess
import tempfile
from typing import Optional, List, Dict, Tuple
from urllib.parse import (
    parse_qs,
    parse_qsl,
    unquote,
    urlencode,
    urlparse,
    urlunparse,
)
from urllib.request import url2pathname

# ----- 1.2 第三方库（带依赖检查）-----
_missing_deps = []  # 记录缺失的依赖

try:
    import numpy as np  # 用于向量计算
except ImportError:
    _missing_deps.append("numpy")
    np = None

try:
    import aiohttp  # 用于异步HTTP请求
except ImportError:
    _missing_deps.append("aiohttp")
    aiohttp = None

HAS_PIL = False
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    # 虽然后备支持MD5，但强烈建议安装Pillow以获得dHash能力
    pass 

# ----- 1.3 AstrBot 框架 API -----
from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Plain, Image, Video, Reply, Forward

# ----- 1.4 插件命令过滤相关 -----
try:
    from astrbot.core.star.filter.command import CommandFilter
    from astrbot.core.star.filter.command_group import CommandGroupFilter
    from astrbot.core.star.star_handler import star_handlers_registry, StarHandlerMetadata
    HAS_COMMAND_FILTER = True
except ImportError:
    HAS_COMMAND_FILTER = False
    logger.warning("[Memory Reboot] 无法导入命令过滤模块，自动过滤插件命令功能不可用")

# 启动时报告缺失依赖
if _missing_deps:
    logger.error(
        f"[Memory Reboot] 缺少依赖: {', '.join(_missing_deps)}，"
        f"请运行: pip install {' '.join(_missing_deps)}"
    )


# ==============================================================================
# 第二部分：常量定义
# ==============================================================================

DEFAULT_DATA_RETENTION_DAYS = 7           # 数据保留天数
DEFAULT_SIMILARITY_THRESHOLD = 0.95       # 文本相似度阈值
DEFAULT_IMAGE_HASH_THRESHOLD = 0.90       # 图片哈希相似度阈值（dHash 256位）
DEFAULT_MIN_UNIQUE_SENDERS = 3            # 最少不同发送者数量
DEFAULT_COOLDOWN_SECONDS = 3600           # 冷却时间（秒）
DEFAULT_MIN_TEXT_LENGTH = 2               # 最小文本长度
DEFAULT_FORWARD_MAX_DEPTH = 3             # 合并转发最大展开层数
DEFAULT_FORWARD_MAX_FETCHES = 8           # 单条消息最多调用get_forward_msg次数
DEFAULT_FORWARD_FETCH_TIMEOUT = 10        # 单次拉取转发资源超时（秒）
DEFAULT_FORWARD_MAX_NODES = 100           # 单条消息最多处理转发节点数
DEFAULT_FORWARD_MAX_SEGMENTS = 500        # 单条消息最多处理消息段数
DEFAULT_FORWARD_MAX_CHARS = 12000         # 单条消息最多提取文本字符数
DEFAULT_FORWARD_MIN_VISIBLE_CHARS = 20    # 含不可读内层转发时最少可见正文字符数
DEFAULT_FORWARD_TEXT_MATCH_MIN_CHARS = 20 # 发送者+文本指纹最少字符数
DEFAULT_FORWARD_TEXT_MATCH_MIN_NODES = 2  # 发送者+文本指纹最少文本节点数
FORWARD_HASH_VERSION = 4                  # 转发内容指纹算法版本
DEFAULT_VIDEO_MAX_ANALYZE_SIZE_MB = 100   # 下载并抽帧的视频大小上限
DEFAULT_VIDEO_FRAME_HASH_THRESHOLD = 0.90 # 视频帧平均dHash相似度阈值
DEFAULT_VIDEO_MIN_FRAME_SIMILARITY = 0.85 # 单帧dHash最低相似度
DEFAULT_VIDEO_DOWNLOAD_TIMEOUT = 120      # 视频下载总超时（秒）
DEFAULT_VIDEO_PROCESS_TIMEOUT = 30        # 单次ffmpeg/ffprobe超时（秒）
VIDEO_FINGERPRINT_VERSION = 1             # 普通视频指纹算法版本
VIDEO_FRAME_POSITIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
MAX_VIDEO_COMPONENTS_PER_MESSAGE = 3
DEFAULT_URL_TRACKING_PARAMS = [
    "utm_*",
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "yclid",
    "_openstat",
    "spm",
    "spm_id_from",
    "vd_source",
]
REMINDER_IMAGE_FILENAME = "1000101866.jpg" # 提醒图片文件名
REMINDER_SUMMARY_MAX_CHARS = 80             # 提醒摘要正文最大长度


# ==============================================================================
# 第三部分：主插件类
# ==============================================================================

class MemoryRebootPlugin(Star):
    """
    Memory Reboot - 记忆重现插件主类
    
    继承自 AstrBot 的 Star 基类，实现群消息监听和处理逻辑。
    
    主要功能模块:
        1. 消息监听与内容提取
        2. 文本Embedding相似度计算
        3. 图片感知哈希相似度计算
        4. 视频文件与五帧感知指纹匹配
        5. LLM智能判断
        6. 提醒消息发送
    
    配置项说明（通过_conf_schema.json定义）:
        - blocked_groups: 黑名单群组列表
        - similarity_threshold: 文本相似度阈值 (0-1)
        - image_hash_threshold: 图片哈希相似度阈值 (0-1)
        - min_unique_senders: 触发提醒所需的最少不同发送者数量
        - cooldown_seconds: 同一话题的冷却时间（秒）
        - data_retention_days: 历史数据保留天数 (默认7天)
        - embedding_provider_id: Embedding模型提供商ID
        - vision_provider_id: 图片识别LLM的提供者ID
        - judge_provider_id: 判断LLM的提供者ID
        - forward_max_depth: 合并转发最大展开层数
        - forward_max_fetches: 单条消息最多调用get_forward_msg次数
        - forward_fetch_timeout: 单次调用get_forward_msg超时秒数
        - forward_max_nodes/segments/chars: 合并转发处理规模上限
        - forward_min_visible_chars: 内层不可读时生成哈希所需可见正文长度
        - forward_sender_text_match: 使用发送者ID和文本精确匹配转发
        - forward_debug_log: 临时输出完整的转发指纹诊断日志
        - url_only_exact_match: 纯URL消息只按规范化完整URL精确匹配
        - url_tracking_params: URL规范化时移除的跟踪参数
        - video_duplicate_enabled: 启用普通视频重复检测
        - video_max_analyze_size_mb: 下载并抽帧的视频大小上限
        - video_frame_hash_threshold: 五帧平均dHash相似度阈值
        - ffmpeg_path: FFmpeg可执行文件或命令名
        - video_debug_log: 临时输出视频处理后的特征指纹
    
    性能优化:
        - 内存缓存: 消息列表缓存在内存中，避免每次从磁盘加载
        - 增量写入: 新消息直接追加到缓存和磁盘，无需读-改-写
        - 懒加载: 仅在首次访问群组数据时从磁盘加载
    """
    
    # 内存缓存：{group_id: {"messages": [...], "last_load": timestamp, "dirty": bool}}
    _cache: Dict[str, Dict] = {}
    
    # ==========================================================================
    # 4.1 默认Prompt模板
    # ==========================================================================
    
    # 图片识别Prompt：用于判断图片类型并提取内容
    DEFAULT_VISION_PROMPT = """你是一个图像分类助手，你的任务是判断图片类型以及分析图片里的内容。
最后直接返回JSON，不要输出其他内容。

## 核心判断逻辑

**is_sticker: true (需要被跳过的图片)**
*   **意图**：仅用于表达情绪、态度，或属于无意义的打招呼/冒泡。
*   **特征**：
    *   经典的互联网表情包（熊猫头、各种猫猫、黄脸Emoji等）。
    *   单纯的人物/动漫图，配文仅为短语或一两句话（如"急了"、"不炒作永无出头之日"、"我是人工智能的对立面，我是天然愚蠢"）。
    *   没有文字内容或者文字内容缺乏具体指代，不包含专有名词、事件或独特观点。
    *   QQ/微信商店的贴纸。

**is_sticker: false (不需要跳过的图片)**
*   **意图**：用于分享信息、发起话题、展示证据或讲述一个具体的"梗"。
*   **特征**：
    *   **截图类**：新闻快讯、文章正文、社交媒体帖子（微博/推特/小红书）、聊天记录、软件界面。
    *   **复杂梗图**：多格漫画、有前后文对比的图、或配文包含具体事件/行业痛点/复杂观点的梗图。
    *   **其他**：海报、公告、数据图表。
*   **额外**：如果不确定（例如文字较多的模糊图片），请标记为 false 以免漏掉潜在话题。

## Content 提取规则 (当 is_sticker=false 时)
务必详细提取，因为这些内容将用于生成搜索向量。
1.  **全面OCR**：提取图片中所有可辨识的文字，留意关键的数字、日期、ID。
2.  **场景描述**：简述图片类型（如"屏幕截图"、"文字较多的新闻"）。
3.  **视觉细节**：如果是梗图，描述画面发生了什么（如"左边是...右边是..."）。

**一致性要求**：
- 优先输出OCR文字原文，减少主观描述
- 使用固定格式：【类型】+ 客观内容
- 避免使用"可能"、"似乎"、"大概"等不确定词汇
- 不要添加个人解读或情感评价

## JSON示例

示例1（纯情绪表达 -> True）：
{"is_sticker": true, "content": "动漫角色流泪，配文'怎么会这样'"}

示例2（新闻/资讯截图 -> False）：
{"is_sticker": false, "content": "【新闻截图】来源：闪电新闻。标题：'月之暗面Kimi官方账号喊话百度'。正文提到百度搜索Kimi官网前4条全是广告。画面显示主持人吴阳欣蔚。"}

示例3（复杂/具体内容的梗图 -> False）：
{"is_sticker": false, "content": "【对比梗图】主题：Nostalgia be like。上图：2016年波奇酱在房间里流泪。下图：2026年波奇酱依然在流泪，脑子里想着2016年的自己。寓意：十年过去了没有任何改变。"}

请分析本图："""

    # 判断Prompt：用于LLM判断是否需要提醒用户
    DEFAULT_JUDGE_PROMPT = """你是一个社群消息分析员。判断当前消息是否属于"旧闻重发"，决定是否需要提醒用户。

## 已确认的前提
1. 当前消息与历史消息**检测到相似内容**
2. 已有 **{unique_count}人** 发送过类似内容（阈值：{min_senders}人）
3. 当前发送者**不是**最早的发送者

## 数据

**历史参考**（{matched_time_ago}）：
发送者：{matched_sender}
内容：{matched_content}

**历史上下文**：
{history_str}

**当前上下文**（最近消息）：
{current_str}

**待判断消息**：
发送者：{sender_name}
内容：{content}

## 判断逻辑

**提醒（should_remind: true）的情况：**
- 用户显然不知道群里之前讨论过这个话题
- 直接转发/分享内容，没有附带评论或讨论
- 单纯发一张图片（无上下文引用）
- 与当前讨论话题无关，突然出现的旧内容（请仔细分析确认无关性）

**不提醒（should_remind: false）的情况：**
- 正在进行的讨论流中（最近几条消息都在讨论相关话题）。
- 已经开启了新话题，在新话题里并不是旧闻重发。
- 明确是对历史消息的回复、引用或补充
- 复读/接龙/玩梗行为
- 提供了新的信息增量（如后续进展、不同角度）
- **纯@消息**：消息仅包含@某用户名，无实质内容（如"@xxx"、"@xxx ?"），这类通常是社交互动而非信息分享
- **针对特定用户的短回复**：如"@xxx 收到"、"@xxx 好的"等日常交流

## 注意
- 很多用户只是单纯发图片或文字，没有"卧槽"之类的语气
- 判断重点是：**用户是否知道群里已经讨论过这件事**
- 如果当前上下文中没有相关讨论，且时间间隔较长，嫌疑比较高，但仍需谨慎判断。
- 同时也注意是否已经开启了新的话题，用户的话在新话题里是否能与上下文产生联系

## 特殊情况处理
- **纯@消息不应触发提醒**：如果当前消息的主体内容仅仅是@某个用户名（可能带有少量语气词如"?"、"！"、"啊"等），这不是信息分享行为，而是社交互动，不应提醒
- **@+短语+一两句话回复**：如"@xxx 收到"、"@xxx 确实"、"@xxx 你觉得怎么样"等通常属于日常社交互动，不应提醒

## 输出格式
简短分析后，输出JSON：

```json
{{
    "should_remind": true/false,
    "reason": "判断理由"
}}
```"""

    # ==========================================================================
    # 4.2 初始化方法
    # ==========================================================================
    
    def __init__(self, context: Context, config: dict):
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象，提供各种API访问能力
            config: 插件配置字典，来自_conf_schema.json的用户配置
        """
        super().__init__(context)
        self.config = config
        
        # 设置数据存储目录（支持旧版本数据自动迁移）
        self._setup_data_directory()
        
        # 插件所在目录（用于加载资源文件如提醒图片）
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化内存缓存
        self._cache = {}
        self._video_analysis_semaphore = asyncio.Semaphore(1)
        self._ffmpeg_path, self._ffprobe_path = self._resolve_ffmpeg_tools()

        logger.info("[Memory Reboot] 插件初始化完成（含内存缓存优化）")

    async def terminate(self):
        """插件卸载时调用"""
        for group_id in list(self._cache.keys()):
            self._flush_cache(group_id)
        self._cache.clear()
        logger.info("[Memory Reboot] 资源已清理")
    
    def _setup_data_directory(self):
        """
        设置数据存储目录，并处理旧版本数据迁移
        
        迁移逻辑：
        - 路径1 (v0.x): data/old_news_reminder
        - 路径2 (v1.0+ 硬编码): data/memory_reboot (cwd下)
        - 路径3 (v1.2+ 标准): StarTools.get_data_dir()
        
        将旧路径数据迁移到标准路径。
        """
        # 1. 获取标准路径 (pathlib.Path)
        standard_path = StarTools.get_data_dir()
        
        # 2. 定义旧路径
        cwd = os.getcwd()
        legacy_v0 = os.path.join(cwd, "data", "old_news_reminder")
        legacy_v1 = os.path.join(cwd, "data", "memory_reboot")
        
        # 3. 检查是否需要迁移
        # 如果标准目录不存在，且存在旧数据，则尝试迁移
        if not standard_path.exists():
            target_source = None
            if os.path.exists(legacy_v1):
                target_source = legacy_v1
                logger.info(f"[Memory Reboot] 检测到 v1.0 旧数据目录: {legacy_v1}")
            elif os.path.exists(legacy_v0):
                target_source = legacy_v0
                logger.info(f"[Memory Reboot] 检测到 v0.x 旧数据目录: {legacy_v0}")
            
            if target_source:
                try:
                    # 确保标准路径的父目录存在
                    standard_path.parent.mkdir(parents=True, exist_ok=True)
                    # 移动目录
                    shutil.move(target_source, str(standard_path))
                    logger.info(f"[Memory Reboot] 数据成功迁移至标准路径: {standard_path}")
                except Exception as e:
                    logger.error(f"[Memory Reboot] 数据迁移失败: {e}")
                    # 迁移失败，回退使用旧路径以防数据丢失
                    self.data_dir = target_source
                    return

        # 4. 设置最终路径并确保存在
        self.data_dir = str(standard_path)
        os.makedirs(self.data_dir, exist_ok=True)
    
    # ==========================================================================
    # 4.3 配置检查方法
    # ==========================================================================
    
    def _is_group_enabled(self, group_id: str) -> bool:
        """
        检查指定群组是否启用了插件功能
        
        通过检查群组ID是否在黑名单中来判断。
        
        Args:
            group_id: 群组ID
            
        Returns:
            True: 群组已启用（不在黑名单中）
            False: 群组已禁用（在黑名单中）
        """
        blocked = self.config.get("blocked_groups", [])
        # 统一转换为字符串进行比较，避免类型不一致问题
        return str(group_id) not in [str(g) for g in blocked]
    
    # ==========================================================================
    # 4.4 插件命令过滤方法
    # ==========================================================================
    
    def _get_all_plugin_commands(self) -> List[str]:
        """
        获取所有已注册插件的命令列表
        
        通过遍历AstrBot的star_handlers_registry获取所有已注册的命令和命令组。
        这些命令将用于自动过滤，避免将其他插件的指令误判为重复内容。
        
        Returns:
            命令列表，例如 ["签到", "帮助", "天气", ...]
        """
        if not HAS_COMMAND_FILTER:
            logger.debug("[Memory Reboot] 命令过滤模块不可用，跳过获取插件命令")
            return []
        
        commands = set()  # 使用set去重
        
        try:
            # 获取所有插件的元数据
            all_stars_metadata = self.context.get_all_stars()
            all_stars_metadata = [star for star in all_stars_metadata if star.activated]
            
            if not all_stars_metadata:
                logger.debug("[Memory Reboot] 没有找到任何激活的插件")
                return []
            
            # 构建模块路径集合，用于匹配handler
            valid_module_paths = set()
            for star in all_stars_metadata:
                module_path = getattr(star, "module_path", None)
                if module_path:
                    valid_module_paths.add(module_path)
            
            # 遍历所有注册的处理器
            for handler in star_handlers_registry:
                if not isinstance(handler, StarHandlerMetadata):
                    continue
                
                # 检查此处理器是否属于已激活的插件
                if handler.handler_module_path not in valid_module_paths:
                    continue
                
                # 遍历处理器的过滤器，查找命令或命令组
                for filter_ in handler.event_filters:
                    if isinstance(filter_, CommandFilter):
                        # 添加主命令
                        if filter_.command_name:
                            commands.add(filter_.command_name)
                        # 添加别名
                        if hasattr(filter_, 'alias') and filter_.alias:
                            for alias in filter_.alias:
                                commands.add(alias)
                    elif isinstance(filter_, CommandGroupFilter):
                        # 添加命令组名称
                        if filter_.group_name:
                            commands.add(filter_.group_name)
            
            logger.debug(f"[Memory Reboot] 获取到 {len(commands)} 个插件命令")
            
        except Exception as e:
            logger.error(f"[Memory Reboot] 获取插件命令失败: {e}")
        
        return list(commands)
    
    def _build_command_patterns(self) -> List[str]:
        """
        根据插件命令构建正则表达式模式列表
        
        为每个命令生成匹配模式，支持以下格式：
        - 直接匹配命令（如 "签到"）
        - 带前缀的命令（如 "/签到", "!签到"）
        - 命令后带参数（如 "签到 xxx", "/天气 北京"）
        
        Returns:
            正则表达式模式列表
        """
        commands = self._get_all_plugin_commands()
        patterns = []
        
        for cmd in commands:
            if not cmd:
                continue
            # 转义正则特殊字符
            escaped_cmd = re.escape(cmd)
            # 匹配：可选前缀 + 命令 + 可选参数
            # 前缀包括: / ! # . 等常见命令前缀
            pattern = f"^[/!#\\.。]?{escaped_cmd}($|\\s.*$)"
            patterns.append(pattern)
        
        return patterns
    
    # 自己的命令白名单，这些命令不应该被过滤（需要由自己的命令处理器处理）
    SELF_COMMANDS = ["记忆状态", "查看过滤命令", "擦除记忆"]
    
    def _is_plugin_command(self, content: str) -> bool:
        """
        检查消息内容是否为其他插件的命令
        
        注意：自己的命令（SELF_COMMANDS）不会被过滤，以确保命令处理器能够正常工作。
        
        Args:
            content: 消息文本内容
            
        Returns:
            True: 是其他插件命令，应该被过滤
            False: 不是插件命令，或是自己的命令
        """
        if not self.config.get("auto_filter_commands", True):
            return False
        
        if not content:
            return False
        
        content_stripped = content.strip()
        
        # 检查是否是自己的命令（白名单），如果是则不过滤
        for self_cmd in self.SELF_COMMANDS:
            # 匹配：可选前缀 + 命令
            self_pattern = f"^[/!#\\.。]?{re.escape(self_cmd)}($|\\s.*$)"
            try:
                if re.match(self_pattern, content_stripped, re.IGNORECASE):
                    logger.debug(f"[Memory Reboot] 检测到自己的命令，不过滤: {content_stripped[:50]}")
                    return False
            except re.error:
                continue
        
        # 获取命令模式并检查是否是其他插件的命令
        patterns = self._build_command_patterns()
        
        for pattern in patterns:
            try:
                if re.match(pattern, content_stripped, re.IGNORECASE):
                    logger.debug(f"[Memory Reboot] 检测到插件命令: {content_stripped[:50]}")
                    return True
            except re.error:
                continue
        
        return False
    
    # ==========================================================================
    # 4.4 数据持久化方法 (分片存储 + Gzip压缩优化)
    # ==========================================================================
    
    def _get_group_dir(self, group_id: str) -> str:
        """获取群组专属数据目录"""
        path = os.path.join(self.data_dir, str(group_id))
        os.makedirs(path, exist_ok=True)
        return path

    def _get_daily_file(self, group_id: str, timestamp: float) -> str:
        """获取指定时间戳对应的日存储文件路径（.json.gz）"""
        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        return os.path.join(self._get_group_dir(group_id), f"{date_str}.json.gz")

    def _migrate_legacy_data(self, group_id: str):
        """将旧版单文件数据迁移到分片压缩存储"""
        legacy_path = os.path.join(self.data_dir, f"{group_id}_messages.json")
        if not os.path.exists(legacy_path):
            return
        
        try:
            logger.info(f"[Memory Reboot] 检测到旧数据，正在迁移群 {group_id} 到压缩分片存储...")
            with open(legacy_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
            
            # 按日期分组
            grouped = {}
            for msg in messages:
                ts = msg.get("timestamp", 0)
                date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                if date_str not in grouped:
                    grouped[date_str] = []
                grouped[date_str].append(msg)
            
            # 写入压缩分片
            group_dir = self._get_group_dir(group_id)
            for date_str, msgs in grouped.items():
                path = os.path.join(group_dir, f"{date_str}.json.gz")
                tmp_path = path + ".tmp"
                with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                    json.dump(msgs, f, ensure_ascii=False)
                os.replace(tmp_path, path)
            
            # 备份原文件
            os.rename(legacy_path, legacy_path + ".bak")
            logger.info(f"[Memory Reboot] 数据迁移完成！旧文件已备份为 .bak")
            
        except Exception as e:
            logger.error(f"[Memory Reboot] 数据迁移失败: {e}")

    def _load_messages(self, group_id: str) -> List[Dict]:
        """
        加载最近N天的消息（带内存缓存优化）
        
        优化策略:
        - 首次加载从磁盘读取，后续直接返回缓存
        - 每天首次访问时检查是否需要清理过期数据
        """
        group_id = str(group_id)
        
        # 检查缓存是否存在且有效
        if group_id in self._cache:
            cache_entry = self._cache[group_id]
            cache_date = datetime.date.fromtimestamp(cache_entry.get("last_load", 0))
            today = datetime.date.today()
            
            # 如果是同一天的缓存，直接返回
            if cache_date == today:
                return cache_entry["messages"]
            
            # 跨天了，需要清理过期消息并更新缓存
            retention_days = self.config.get("data_retention_days", 7)
            cutoff = time.time() - retention_days * 86400
            cache_entry["messages"] = [m for m in cache_entry["messages"] if m.get("timestamp", 0) > cutoff]
            cache_entry["last_load"] = time.time()
            return cache_entry["messages"]
        
        # 缓存不存在，从磁盘加载
        return self._load_messages_from_disk(group_id)
    
    def _load_messages_from_disk(self, group_id: str) -> List[Dict]:
        """从磁盘加载消息并更新缓存"""
        # 1. 尝试迁移单一大文件
        self._migrate_legacy_data(group_id)
        
        group_dir = self._get_group_dir(group_id)
        retention_days = self.config.get("data_retention_days", 7)
        
        # 2. 计算需要加载的日期范围
        today = datetime.date.today()
        valid_dates = { (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                        for i in range(retention_days + 1) }
        
        all_messages = []
        if os.path.exists(group_dir):
            try:
                for filename in sorted(os.listdir(group_dir)):
                    filepath = os.path.join(group_dir, filename)
                    
                    # 自动迁移：如果发现未压缩的 .json，压缩为 .json.gz
                    if filename.endswith(".json") and not filename.endswith(".json.gz"):
                        try:
                            # 读取未压缩数据
                            with open(filepath, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            # 写入压缩数据
                            gz_path = filepath + ".gz"
                            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False)
                            # 删除旧文件并更新路径
                            os.remove(filepath)
                            filename = filename + ".gz"
                            filepath = gz_path
                            logger.debug(f"[Memory Reboot] 已自动压缩文件: {filename}")
                        except Exception as e:
                            logger.error(f"[Memory Reboot] 自动压缩失败: {e}")
                            continue

                    # 加载 .json.gz 文件
                    if filename.endswith(".json.gz"):
                        date_part = filename.replace(".json.gz", "")
                        if date_part in valid_dates:
                            try:
                                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                                    day_msgs = json.load(f)
                                    all_messages.extend(day_msgs)
                            except Exception as e:
                                logger.error(f"[Memory Reboot] 读取压缩文件失败 {filepath}: {e}")
            except Exception as e:
                logger.error(f"[Memory Reboot] 读取消息失败: {e}")
                
        # 确保按时间排序
        all_messages.sort(key=lambda x: x.get("timestamp", 0))
        
        # 更新缓存
        self._cache[group_id] = {
            "messages": all_messages,
            "last_load": time.time()
        }
        
        logger.debug(f"[Memory Reboot] 从磁盘加载 {len(all_messages)} 条消息到缓存")
        return all_messages
    def _append_message(self, group_id: str, message: Dict):
        """
        追加单条消息（内存缓存优化版）
        
        优化策略:
        - 先更新内存缓存（O(1)操作）
        - 异步/延迟写入磁盘
        - 使用批量写入减少I/O次数
        """
        group_id = str(group_id)
        
        # 1. 更新内存缓存
        if group_id not in self._cache:
            self._cache[group_id] = {
                "messages": [],
                "last_load": time.time()
            }
        
        self._cache[group_id]["messages"].append(message)
        
        # 2. 写入磁盘（优化：每N条消息或跨天时才写入）
        cache_entry = self._cache[group_id]
        messages = cache_entry["messages"]
        
        # 获取当天的消息用于写入
        today_str = datetime.datetime.fromtimestamp(message.get("timestamp", time.time())).strftime("%Y-%m-%d")
        today_messages = [m for m in messages if datetime.datetime.fromtimestamp(m.get("timestamp", 0)).strftime("%Y-%m-%d") == today_str]
        
        # 每10条消息写入一次，或者是当天第一条消息时写入
        should_write = len(today_messages) == 1 or len(today_messages) % 10 == 0
        
        if should_write:
            self._write_daily_messages(group_id, today_str, today_messages)
    
    def _write_daily_messages(self, group_id: str, date_str: str, messages: List[Dict]):
        """将消息写入指定日期的文件"""
        path = os.path.join(self._get_group_dir(group_id), f"{date_str}.json.gz")
        tmp_path = path + ".tmp"
        
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"[Memory Reboot] 写入失败: {e}")
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
    
    def _flush_cache(self, group_id: str):
        """强制将缓存写入磁盘（用于关闭前或手动保存）"""
        group_id = str(group_id)
        if group_id not in self._cache:
            return
        
        messages = self._cache[group_id]["messages"]
        if not messages:
            return
        
        # 按日期分组写入
        grouped = {}
        for msg in messages:
            date_str = datetime.datetime.fromtimestamp(msg.get("timestamp", 0)).strftime("%Y-%m-%d")
            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append(msg)
        
        for date_str, day_msgs in grouped.items():
            self._write_daily_messages(group_id, date_str, day_msgs)
        
        logger.debug(f"[Memory Reboot] 已将群 {group_id} 的 {len(messages)} 条消息写入磁盘")
    
    def _cleanup_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        清理过期消息
        
        注意: 使用内存缓存后，_load_messages已经在跨天时自动清理过期消息。
        此方法保留以兼容现有调用，但通常直接返回原列表。
        """
        # 缓存模式下，过期清理已在 _load_messages 中完成
        # 这里只做简单的时间戳校验（通常不需要）
        if not messages:
            return messages
        
        now = time.time()
        retention_days = self.config.get("data_retention_days", 7)
        cutoff = now - retention_days * 86400
        
        # 快速检查：如果最老的消息都没过期，直接返回
        if messages[0].get("timestamp", 0) > cutoff:
            return messages
        
        return [m for m in messages if m.get("timestamp", 0) > cutoff]
    
    def _cleanup_image_cache(self, group_id: str):
        """
        清理过期的图片缓存文件
        
        图片缓存用于计算感知哈希。过期的缓存文件会被删除以节省磁盘空间。
        文件名格式: 20260202_041900_123456.jpg (日期_时间_微秒.jpg)
        
        Args:
            group_id: 群组ID
        """
        cache_dir = os.path.join(self.plugin_dir, "image_cache", group_id)
        
        if not os.path.exists(cache_dir):
            return
        
        retention_days = self.config.get("data_retention_days", 7)
        cutoff = time.time() - retention_days * 86400
        
        try:
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                
                # 从文件名解析时间戳
                try:
                    date_part = filename.split("_")[0]  # 20260202
                    time_part = filename.split("_")[1]  # 041900
                    dt = datetime.datetime.strptime(
                        f"{date_part}_{time_part}", 
                        "%Y%m%d_%H%M%S"
                    )
                    file_timestamp = dt.timestamp()
                    
                    # 删除过期文件
                    if file_timestamp < cutoff:
                        os.remove(filepath)
                        logger.debug(f"[Memory Reboot] 清理过期图片: {filename}")
                        
                except (ValueError, IndexError):
                    # 文件名格式不符合预期，跳过
                    pass
                    
        except Exception as e:
            logger.error(f"[Memory Reboot] 清理图片缓存失败: {e}")
    
    # ==========================================================================
    # 4.5 Embedding相关方法
    # ==========================================================================

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取文本的Embedding向量

        通过AstrBot provider调用embedding服务。
        优先使用 get_all_embedding_providers() 获取嵌入提供商。

        Args:
            text: 要编码的文本

        Returns:
            文本的向量表示（浮点数列表），失败返回None
        """
        provider_id = self.config.get("embedding_provider_id", "")
        if not provider_id:
            logger.debug("[Memory Reboot] Embedding: 未配置provider_id")
            return None

        try:
            provider = None

            # 优先从嵌入提供商列表中查找
            if hasattr(self.context, 'get_all_embedding_providers'):
                all_providers = self.context.get_all_embedding_providers()
                # 先尝试ID精确匹配
                for p in all_providers:
                    if hasattr(p, 'id') and p.id == provider_id:
                        provider = p
                        break
                # 如果ID匹配失败，尝试名称匹配
                if not provider:
                    for p in all_providers:
                        if hasattr(p, 'meta') and hasattr(p.meta, 'name'):
                            if p.meta.name == provider_id:
                                provider = p
                                break

            # 回退到通用方法
            if not provider:
                provider = self.context.get_provider_by_id(provider_id)

            if not provider:
                logger.debug(f"[Memory Reboot] 未找到嵌入提供商: {provider_id}")
                return None

            # 尝试多种嵌入方法
            methods = ['get_embeddings', 'embeddings', 'embedding', 'get_embedding']
            last_error = None
            for method_name in methods:
                if hasattr(provider, method_name):
                    method = getattr(provider, method_name)
                    try:
                        if method_name in ['get_embeddings', 'embeddings']:
                            result = await method([text])
                            if result and len(result) > 0:
                                return result[0]
                        else:
                            result = await method(text)
                            if result:
                                return result if isinstance(result, list) else list(result)
                    except Exception as e:
                        last_error = e
                        logger.debug(f"[Memory Reboot] 方法 {method_name} 调用失败: {e}")
                        continue

            if last_error:
                logger.warning(f"[Memory Reboot] 所有embedding方法均失败，最后错误: {last_error}")

        except Exception as e:
            logger.error(f"[Memory Reboot] 获取embedding失败: {e}")

        return None
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        余弦相似度公式: cos(θ) = (A·B) / (|A| × |B|)
        
        余弦相似度的特点:
        - 值域为 [-1, 1]
        - 1 表示向量方向完全相同（最相似）
        - 0 表示向量正交（无关）
        - -1 表示向量方向完全相反
        
        Args:
            v1: 向量1
            v2: 向量2
            
        Returns:
            相似度值，范围 [-1, 1]
        """
        # 参数校验
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        
        a, b = np.array(v1), np.array(v2)
        # 添加极小值1e-9避免除零错误
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    
    # ==========================================================================
    # 4.6 图片处理方法
    # ==========================================================================
    
    async def _cache_image(
        self,
        image_source: str,
        timestamp: float,
        group_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        缓存远程或本地图片，同时计算感知哈希

        Args:
            image_source: 图片的HTTP(S)地址、本地路径或file URI
            timestamp: 消息的时间戳（用于生成文件名）
            group_id: 群组ID（用于分目录存储）
            
        Returns:
            元组 (缓存文件路径, 图片哈希值)
            如果下载或处理失败，返回 (None, None)
        """
        try:
            # 创建群组专属的缓存目录
            cache_dir = os.path.join(self.plugin_dir, "image_cache", group_id)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 生成文件名: 20260202_041900_123456.jpg
            dt = datetime.datetime.fromtimestamp(timestamp)
            # 使用微秒部分确保文件名唯一
            filename = dt.strftime("%Y%m%d_%H%M%S") + f"_{int((timestamp % 1) * 1000000)}.jpg"
            filepath = os.path.join(cache_dir, filename)

            source = str(image_source)
            if source.startswith(("http://", "https://")):
                # 远程图片仍通过HTTP下载
                async with aiohttp.ClientSession() as session:
                    async with session.get(source) as response:
                        if response.status != 200:
                            logger.warning(
                                f"[Memory Reboot] 图片下载失败: HTTP {response.status}, "
                                f"source={source}"
                            )
                            return None, None
                        with open(filepath, "wb") as f:
                            f.write(await response.read())
            else:
                # AstrBot v4.26.1会把收到的图片预处理为本地JPEG，并把
                # Image.url/file/path都改成本地临时路径，不能再交给aiohttp。
                if os.path.isabs(source):
                    local_path = source
                else:
                    parsed = urlparse(source)
                    if parsed.scheme == "file":
                        local_path = url2pathname(unquote(parsed.path))
                        if parsed.netloc:
                            local_path = f"//{parsed.netloc}{local_path}"
                    elif not parsed.scheme:
                        local_path = source
                    else:
                        raise ValueError(f"不支持的图片来源格式: {source}")

                if not os.path.isfile(local_path):
                    raise FileNotFoundError(f"本地图片不存在: {local_path}")
                shutil.copy2(local_path, filepath)

            logger.debug(f"[Memory Reboot] 图片缓存成功: {filename}")

            # 计算图片的感知哈希
            image_hash = self._compute_image_hash(filepath)
            if image_hash:
                logger.debug(f"[Memory Reboot] 图片哈希: {image_hash}")

            return filepath, image_hash

        except Exception as e:
            logger.error(
                f"[Memory Reboot] 图片缓存失败: {type(e).__name__}: {e}, "
                f"source={image_source}"
            )
        
        return None, None
    
    def _compute_image_hash(self, image_path: str) -> Optional[str]:
        """
        计算图片的感知哈希值（Difference Hash / dHash算法）
        """
        if HAS_PIL:
            try:
                img = PILImage.open(image_path)
                # 使用17x16尺寸，宽度多1用于计算差值，生成16x16=256位哈希
                img = img.convert('L').resize((17, 16), PILImage.Resampling.LANCZOS)
                pixels = list(img.getdata())
                
                # dHash：比较相邻像素差异
                bits = []
                for row in range(16):
                    for col in range(16):
                        left_pixel = pixels[row * 17 + col]
                        right_pixel = pixels[row * 17 + col + 1]
                        bits.append('1' if left_pixel > right_pixel else '0')
                
                # 256位 = 64个十六进制字符
                hash_value = hex(int(''.join(bits), 2))[2:].zfill(64)
                return hash_value
            except Exception as e:
                logger.error(f"[Memory Reboot] 计算图片dHash失败: {e}")
                return None
        else:
            logger.debug("[Memory Reboot] PIL未安装，使用MD5哈希")
            return self._compute_file_md5(image_path)
    
    def _compute_file_md5(self, file_path: str) -> Optional[str]:
        """计算文件的MD5哈希（后备方案）"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"[Memory Reboot] 计算MD5失败: {e}")
            return None
    
    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """计算两个哈希值的相似度（基于汉明距离）"""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        try:
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            diff = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            return 1.0 - (diff / len(bin1))
        except Exception:
            return 0.0

    # ==========================================================================
    # 4.7 视频处理与判重方法
    # ==========================================================================

    @staticmethod
    def _find_executable(command: str) -> Optional[str]:
        """解析外部命令，兼容PATH名称和显式文件路径。"""
        candidate = str(command or "").strip()
        if not candidate:
            return None
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        return shutil.which(candidate)

    def _resolve_ffmpeg_tools(self) -> Tuple[Optional[str], Optional[str]]:
        """解析FFmpeg和同目录的ffprobe。缺失时保留精确文件哈希能力。"""
        configured = str(self.config.get("ffmpeg_path", "ffmpeg") or "ffmpeg")
        ffmpeg_path = self._find_executable(configured)
        ffprobe_path = None
        if ffmpeg_path:
            extension = ".exe" if ffmpeg_path.lower().endswith(".exe") else ""
            adjacent = os.path.join(
                os.path.dirname(ffmpeg_path),
                f"ffprobe{extension}",
            )
            ffprobe_path = self._find_executable(adjacent)
        if not ffprobe_path:
            ffprobe_path = self._find_executable("ffprobe")

        if ffmpeg_path and ffprobe_path:
            logger.info(
                f"[Memory Reboot] 视频抽帧可用: ffmpeg={ffmpeg_path}, "
                f"ffprobe={ffprobe_path}"
            )
        else:
            logger.warning(
                "[Memory Reboot] FFmpeg或ffprobe不可用，"
                "视频将降级为文件SHA-256/元数据精确判重"
            )
        return ffmpeg_path, ffprobe_path

    def _get_video_max_size_bytes(self) -> int:
        """读取视频分析大小上限，限制在1 MiB到2 GiB。"""
        try:
            size_mb = int(
                self.config.get(
                    "video_max_analyze_size_mb",
                    DEFAULT_VIDEO_MAX_ANALYZE_SIZE_MB,
                )
            )
        except (TypeError, ValueError):
            size_mb = DEFAULT_VIDEO_MAX_ANALYZE_SIZE_MB
        size_mb = max(1, min(size_mb, 2048))
        return size_mb * 1024 * 1024

    def _get_video_frame_threshold(self) -> float:
        """读取视频帧平均相似度阈值。"""
        try:
            threshold = float(
                self.config.get(
                    "video_frame_hash_threshold",
                    DEFAULT_VIDEO_FRAME_HASH_THRESHOLD,
                )
            )
        except (TypeError, ValueError):
            threshold = DEFAULT_VIDEO_FRAME_HASH_THRESHOLD
        return max(0.5, min(threshold, 1.0))

    @staticmethod
    def _parse_positive_int(value) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _normalize_video_filename(value) -> str:
        """从路径、URI或平台文件标识中提取稳定的小写文件名。"""
        text = str(value or "").strip()
        if not text or text.startswith(("base64://", "data:")):
            return ""
        parsed = urlparse(text)
        if parsed.scheme:
            text = unquote(parsed.path)
        else:
            text = unquote(text.split("?", 1)[0])
        filename = re.split(r"[\\/]", text)[-1].strip().lower()
        if len(filename) > 512:
            return f"sha256-{hashlib.sha256(filename.encode('utf-8')).hexdigest()}"
        return filename

    @staticmethod
    def _local_video_path(source: str) -> Optional[str]:
        """将本地路径或file URI转换为文件系统路径。"""
        value = str(source or "")
        if not value:
            return None
        if os.path.isabs(value):
            return value
        parsed = urlparse(value)
        if parsed.scheme == "file":
            local_path = url2pathname(unquote(parsed.path))
            if parsed.netloc:
                local_path = f"//{parsed.netloc}{local_path}"
            return local_path
        if not parsed.scheme:
            return value
        return None

    def _extract_raw_video_segments(self, event: AstrMessageEvent) -> List[Dict]:
        """读取OneBot原始视频段，以保留AstrBot组件未声明的file_size。"""
        message_obj = getattr(event, "message_obj", None)
        raw_message = getattr(message_obj, "raw_message", None)
        if not hasattr(raw_message, "get"):
            return []
        raw_segments = raw_message.get("message")
        if not isinstance(raw_segments, list):
            return []
        return [
            segment.get("data", {})
            for segment in raw_segments
            if isinstance(segment, dict)
            and str(segment.get("type") or "").lower() == "video"
            and isinstance(segment.get("data"), dict)
        ]

    def _extract_video_components(
        self,
        event: AstrMessageEvent,
    ) -> List[Dict]:
        """提取普通消息中的Video组件，并与OneBot原始元数据按顺序对齐。"""
        message_obj = getattr(event, "message_obj", None)
        components = getattr(message_obj, "message", []) if message_obj else []
        videos = [comp for comp in components if isinstance(comp, Video)]
        raw_segments = self._extract_raw_video_segments(event)
        results = []

        for index, component in enumerate(
            videos[:MAX_VIDEO_COMPONENTS_PER_MESSAGE]
        ):
            raw_data = raw_segments[index] if index < len(raw_segments) else {}
            component_path = str(getattr(component, "path", None) or "")
            file_reference = (
                raw_data.get("file")
                or getattr(component, "file", None)
            )
            local_component_path = self._local_video_path(component_path)
            if (
                local_component_path
                and os.path.isfile(local_component_path)
            ):
                source = component_path
            else:
                source = (
                    getattr(component, "url", None)
                    or raw_data.get("url")
                    or getattr(component, "file", None)
                    or raw_data.get("file")
                )
            name_source = (
                raw_data.get("file")
                or raw_data.get("file_name")
                or raw_data.get("name")
                or source
            )
            results.append(
                {
                    "component": component,
                    "source": str(source or ""),
                    "file_reference": str(file_reference or ""),
                    "name": self._normalize_video_filename(name_source),
                    "size": self._parse_positive_int(
                        raw_data.get("file_size")
                        or raw_data.get("size")
                    ),
                }
            )

        if len(videos) > MAX_VIDEO_COMPONENTS_PER_MESSAGE:
            logger.warning(
                f"[Memory Reboot] 单条消息含{len(videos)}个视频，"
                f"仅处理前{MAX_VIDEO_COMPONENTS_PER_MESSAGE}个"
            )
        return results

    @staticmethod
    def _onebot_response_data(result) -> Optional[Dict]:
        """兼容直接数据和标准OneBot响应外壳。"""
        if not hasattr(result, "get"):
            return None
        data = result.get("data")
        if isinstance(data, dict):
            return data
        return result if isinstance(result, dict) else None

    async def _fetch_video_source_from_onebot(
        self,
        event: Optional[AstrMessageEvent],
        video_info: Dict,
    ) -> Optional[Dict]:
        """使用消息中的资源标识要求协议端重新下载视频。"""
        file_reference = str(video_info.get("file_reference") or "").strip()
        if not event or not file_reference:
            return None

        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            call_action = getattr(getattr(bot, "api", None), "call_action", None)
        if not callable(call_action):
            logger.warning(
                "[Memory Reboot] 当前消息平台不支持 get_file，无法取回视频"
            )
            return None

        routing_params = {}
        self_id = getattr(getattr(event, "message_obj", None), "self_id", None)
        if self_id:
            routing_params["self_id"] = self_id

        try:
            result = await asyncio.wait_for(
                call_action(
                    "get_file",
                    file=file_reference,
                    **routing_params,
                ),
                timeout=DEFAULT_VIDEO_DOWNLOAD_TIMEOUT,
            )
        except Exception as exc:
            logger.warning(
                f"[Memory Reboot] 通过 get_file 取回视频失败: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

        data = self._onebot_response_data(result)
        if not data:
            logger.warning(
                "[Memory Reboot] get_file 未返回可用的视频文件信息"
            )
            return None

        response_size = self._parse_positive_int(data.get("file_size"))
        response_name = self._normalize_video_filename(
            data.get("file_name")
        )
        for candidate in (data.get("url"), data.get("file")):
            source = str(candidate or "").strip()
            if not source:
                continue
            if source.startswith(("http://", "https://")):
                return {
                    "source": source,
                    "source_method": "get_file_http",
                    "size": response_size,
                    "name": response_name,
                }
            local_path = self._local_video_path(source)
            if local_path and os.path.isfile(local_path):
                return {
                    "source": source,
                    "source_method": "get_file_local",
                    "size": response_size,
                    "name": response_name,
                }

        logger.warning(
            "[Memory Reboot] get_file 已执行，但返回路径在AstrBot中不可读；"
            "请检查NapCat媒体目录的共享挂载"
        )
        return None

    @staticmethod
    def _compute_file_sha256(file_path: str) -> Optional[str]:
        """分块计算文件SHA-256，避免把整个视频读入内存。"""
        try:
            digest = hashlib.sha256()
            with open(file_path, "rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception as exc:
            logger.error(
                f"[Memory Reboot] 计算视频SHA-256失败: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    @staticmethod
    def _build_video_metadata_hash(
        filename: str,
        file_size: Optional[int],
    ) -> Optional[str]:
        """为免下载的大视频生成“文件名+大小”精确指纹。"""
        if not filename or not file_size:
            return None
        payload = f"{filename}\n{file_size}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _new_video_fingerprint(
        self,
        video_info: Dict,
        file_size: Optional[int],
        analysis: str,
    ) -> Dict:
        filename = str(video_info.get("name") or "")
        return {
            "version": VIDEO_FINGERPRINT_VERSION,
            "name": filename,
            "size": file_size,
            "sha256": None,
            "frame_hashes": [],
            "duration_ms": None,
            "width": None,
            "height": None,
            "metadata_hash": self._build_video_metadata_hash(
                filename,
                file_size,
            ),
            "analysis": analysis,
            "source_method": video_info.get("source_method", "unavailable"),
        }

    async def _download_video(
        self,
        source: str,
        destination: str,
        size_limit: int,
    ) -> Tuple[Optional[int], Optional[str], bool]:
        """限流下载视频，返回(大小, SHA-256, 是否超限)。"""
        timeout = aiohttp.ClientTimeout(total=DEFAULT_VIDEO_DOWNLOAD_TIMEOUT)
        digest = hashlib.sha256()
        downloaded = 0
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(source, allow_redirects=True) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    content_length = self._parse_positive_int(
                        response.headers.get("Content-Length")
                    )
                    if content_length and content_length > size_limit:
                        return content_length, None, True

                    with open(destination, "wb") as file:
                        async for chunk in response.content.iter_chunked(
                            1024 * 1024
                        ):
                            downloaded += len(chunk)
                            if downloaded > size_limit:
                                return None, None, True
                            digest.update(chunk)
                            file.write(chunk)
            return downloaded, digest.hexdigest(), False
        except Exception as exc:
            logger.warning(
                f"[Memory Reboot] 视频下载失败: "
                f"{type(exc).__name__}: {exc}"
            )
            return None, None, False

    async def _run_media_tool(
        self,
        *arguments: str,
    ) -> Tuple[int, bytes, bytes]:
        """无Shell执行FFmpeg工具，并对单次调用施加超时。"""
        process_kwargs = {}
        if os.name == "nt":
            process_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=DEFAULT_VIDEO_PROCESS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(
                f"媒体工具执行超过{DEFAULT_VIDEO_PROCESS_TIMEOUT}秒"
            )
        return process.returncode, stdout, stderr

    async def _probe_video(self, video_path: str) -> Optional[Dict]:
        """读取首个视频流的时长和尺寸。"""
        if not self._ffprobe_path:
            return None
        try:
            returncode, stdout, stderr = await self._run_media_tool(
                self._ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration:format=duration",
                "-of",
                "json",
                video_path,
            )
            if returncode != 0:
                error_text = stderr.decode("utf-8", errors="replace")[:300]
                logger.warning(
                    f"[Memory Reboot] ffprobe读取视频失败: {error_text}"
                )
                return None
            payload = json.loads(stdout.decode("utf-8"))
            streams = payload.get("streams") or []
            if not streams:
                return None
            stream = streams[0]
            raw_duration = (
                (payload.get("format") or {}).get("duration")
                or stream.get("duration")
            )
            duration = float(raw_duration)
            if duration <= 0:
                return None
            return {
                "duration_ms": int(round(duration * 1000)),
                "width": self._parse_positive_int(stream.get("width")),
                "height": self._parse_positive_int(stream.get("height")),
            }
        except Exception as exc:
            logger.warning(
                f"[Memory Reboot] ffprobe读取视频异常: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    async def _extract_video_frame_hashes(
        self,
        video_path: str,
        duration_ms: int,
        temp_dir: str,
    ) -> List[str]:
        """在五个归一化时间点抽帧并计算256位dHash。"""
        if not self._ffmpeg_path or not HAS_PIL or duration_ms <= 0:
            return []

        hashes = []
        duration_seconds = duration_ms / 1000.0
        for index, position in enumerate(VIDEO_FRAME_POSITIONS):
            frame_path = os.path.join(temp_dir, f"frame_{index}.png")
            timestamp = max(0.001, duration_seconds * position)
            try:
                returncode, _, stderr = await self._run_media_tool(
                    self._ffmpeg_path,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-an",
                    "-vf",
                    "scale=17:16:flags=lanczos,format=gray",
                    "-y",
                    frame_path,
                )
                if returncode != 0 or not os.path.isfile(frame_path):
                    error_text = stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[:300]
                    logger.warning(
                        f"[Memory Reboot] 视频第{index + 1}帧提取失败: "
                        f"{error_text}"
                    )
                    continue
                frame_hash = self._compute_image_hash(frame_path)
                if frame_hash:
                    hashes.append(frame_hash)
            except Exception as exc:
                logger.warning(
                    f"[Memory Reboot] 视频第{index + 1}帧处理异常: "
                    f"{type(exc).__name__}: {exc}"
                )
        return hashes

    async def _analyze_video(
        self,
        video_info: Dict,
        event: Optional[AstrMessageEvent] = None,
    ) -> Dict:
        """在大小限制内生成视频精确和感知指纹。"""
        resolved_info = dict(video_info)
        declared_size = self._parse_positive_int(resolved_info.get("size"))
        fingerprint = self._new_video_fingerprint(
            resolved_info,
            declared_size,
            "unavailable",
        )
        size_limit = self._get_video_max_size_bytes()
        if declared_size and declared_size > size_limit:
            fingerprint["analysis"] = "metadata"
            return fingerprint

        source = str(video_info.get("source") or "")
        if not source:
            return fingerprint

        async with self._video_analysis_semaphore:
            with tempfile.TemporaryDirectory(
                prefix="memory_reboot_video_",
                dir=self.data_dir,
            ) as temp_dir:
                local_path = self._local_video_path(source)
                video_path = None
                file_sha256 = None
                file_size = declared_size

                if not (
                    local_path
                    and os.path.isfile(local_path)
                ) and not source.startswith(("http://", "https://")):
                    fetched = await self._fetch_video_source_from_onebot(
                        event,
                        resolved_info,
                    )
                    if fetched:
                        resolved_info.update(
                            {
                                key: value
                                for key, value in fetched.items()
                                if value
                            }
                        )
                        source = str(resolved_info.get("source") or "")
                        local_path = self._local_video_path(source)
                        file_size = (
                            self._parse_positive_int(resolved_info.get("size"))
                            or declared_size
                        )
                        if file_size and file_size > size_limit:
                            return self._new_video_fingerprint(
                                resolved_info,
                                file_size,
                                "metadata",
                            )

                if local_path and os.path.isfile(local_path):
                    file_size = os.path.getsize(local_path)
                    resolved_info.setdefault("source_method", "local")
                    fingerprint = self._new_video_fingerprint(
                        resolved_info,
                        file_size,
                        "metadata" if file_size > size_limit else "sha256",
                    )
                    if file_size > size_limit:
                        return fingerprint
                    video_path = local_path
                    file_sha256 = await asyncio.to_thread(
                        self._compute_file_sha256,
                        video_path,
                    )
                elif source.startswith(("http://", "https://")):
                    resolved_info.setdefault("source_method", "http")
                    extension = os.path.splitext(
                        str(resolved_info.get("name") or "")
                    )[1]
                    if not re.fullmatch(r"\.[a-zA-Z0-9]{1,8}", extension):
                        extension = ".mp4"
                    video_path = os.path.join(temp_dir, f"video{extension}")
                    file_size, file_sha256, oversize = (
                        await self._download_video(
                            source,
                            video_path,
                            size_limit,
                        )
                    )
                    fingerprint = self._new_video_fingerprint(
                        resolved_info,
                        file_size or declared_size,
                        "metadata" if oversize else "sha256",
                    )
                    if oversize or not file_sha256:
                        return fingerprint
                else:
                    logger.warning(
                        "[Memory Reboot] 视频来源不可读，且未能通过 "
                        "get_file 取回"
                    )
                    return fingerprint

                fingerprint["sha256"] = file_sha256
                if not file_sha256 or not video_path:
                    return fingerprint

                probe = await self._probe_video(video_path)
                if not probe:
                    return fingerprint
                fingerprint.update(probe)
                frame_hashes = await self._extract_video_frame_hashes(
                    video_path,
                    probe["duration_ms"],
                    temp_dir,
                )
                if len(frame_hashes) == len(VIDEO_FRAME_POSITIONS):
                    fingerprint["frame_hashes"] = frame_hashes
                    fingerprint["analysis"] = "perceptual"
                return fingerprint

    async def _analyze_videos(
        self,
        video_infos: List[Dict],
        event: Optional[AstrMessageEvent] = None,
    ) -> List[Dict]:
        """按消息段顺序分析普通视频，单个失败不影响其余视频。"""
        if not self._get_bool_config("video_duplicate_enabled", True):
            return []
        results = []
        for video_info in video_infos:
            try:
                results.append(await self._analyze_video(video_info, event))
            except Exception as exc:
                logger.error(
                    f"[Memory Reboot] 视频分析异常: "
                    f"{type(exc).__name__}: {exc}"
                )
                results.append(
                    self._new_video_fingerprint(
                        video_info,
                        self._parse_positive_int(video_info.get("size")),
                        "unavailable",
                    )
                )
        return results

    def _log_video_debug(
        self,
        event: AstrMessageEvent,
        group_id: str,
        videos: List[Dict],
        match_type: Optional[str],
        matched_message: Optional[Dict],
    ) -> None:
        """分段输出视频处理结果，不记录视频URL、路径或文件名。"""
        if not self._is_video_debug_enabled():
            return

        message_obj = getattr(event, "message_obj", None)
        message_id = getattr(message_obj, "message_id", None)
        fingerprints = []
        for index, video in enumerate(videos):
            fingerprints.append(
                {
                    "index": index,
                    "version": video.get("version"),
                    "analysis": video.get("analysis"),
                    "size": video.get("size"),
                    "sha256": video.get("sha256"),
                    "frame_hashes": video.get("frame_hashes") or [],
                    "duration_ms": video.get("duration_ms"),
                    "width": video.get("width"),
                    "height": video.get("height"),
                    "metadata_hash": video.get("metadata_hash"),
                    "source_method": video.get("source_method"),
                }
            )

        payload = {
            "group_id": str(group_id),
            "message_id": str(message_id or ""),
            "match_type": match_type,
            "matched_record_id": (
                matched_message.get("id")
                if isinstance(matched_message, dict)
                else None
            ),
            "limits": {
                "max_analyze_size_bytes": self._get_video_max_size_bytes(),
                "frame_average_threshold": self._get_video_frame_threshold(),
                "frame_min_similarity": DEFAULT_VIDEO_MIN_FRAME_SIMILARITY,
                "required_strong_frames": 4,
                "duration_tolerance": "max(1000ms,3%)",
            },
            "fingerprints": fingerprints,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        chunk_size = 1800
        total_chunks = max(
            1,
            (len(serialized) + chunk_size - 1) // chunk_size,
        )
        log_id = str(message_id or "unknown")
        logger.info(
            "[Memory Reboot][VideoDebug] 视频指纹诊断已启用；"
            f"不包含视频URL、路径、文件名或正文。id={log_id}, "
            f"分段={total_chunks}"
        )
        for index in range(total_chunks):
            start = index * chunk_size
            chunk = serialized[start:start + chunk_size]
            logger.info(
                f"[Memory Reboot][VideoDebug] id={log_id} "
                f"part={index + 1}/{total_chunks} data={chunk}"
            )
        logger.info(f"[Memory Reboot][VideoDebug] id={log_id} END")

    def _video_frames_match(self, current: Dict, stored: Dict) -> bool:
        """比较两组对齐帧，并使用时长约束减少静态画面误报。"""
        current_hashes = current.get("frame_hashes") or []
        stored_hashes = stored.get("frame_hashes") or []
        if (
            len(current_hashes) != len(VIDEO_FRAME_POSITIONS)
            or len(stored_hashes) != len(VIDEO_FRAME_POSITIONS)
        ):
            return False

        current_duration = self._parse_positive_int(current.get("duration_ms"))
        stored_duration = self._parse_positive_int(stored.get("duration_ms"))
        if not current_duration or not stored_duration:
            return False
        duration_limit = max(
            1000,
            int(max(current_duration, stored_duration) * 0.03),
        )
        if abs(current_duration - stored_duration) > duration_limit:
            return False

        similarities = [
            self._hash_similarity(current_hash, stored_hash)
            for current_hash, stored_hash in zip(
                current_hashes,
                stored_hashes,
            )
        ]
        average = sum(similarities) / len(similarities)
        strong_frames = sum(
            similarity >= DEFAULT_VIDEO_MIN_FRAME_SIMILARITY
            for similarity in similarities
        )
        return (
            average >= self._get_video_frame_threshold()
            and strong_frames >= 4
        )

    def _video_match_type(
        self,
        current_videos: List[Dict],
        stored_videos: List[Dict],
        requested_type: Optional[str] = None,
    ) -> Optional[str]:
        """判断两条消息的视频是否按指定层级匹配。"""
        for current in current_videos:
            for stored in stored_videos:
                if requested_type in (None, "video_sha256"):
                    current_sha = current.get("sha256")
                    stored_sha = stored.get("sha256")
                    if current_sha and stored_sha and current_sha == stored_sha:
                        return "video_sha256"

                if requested_type in (None, "video_frame_hash"):
                    if self._video_frames_match(current, stored):
                        return "video_frame_hash"

                if requested_type in (None, "video_metadata"):
                    if current.get("analysis") != "metadata":
                        continue
                    current_metadata = current.get("metadata_hash")
                    stored_metadata = stored.get("metadata_hash")
                    if (
                        current_metadata
                        and stored_metadata
                        and current_metadata == stored_metadata
                    ):
                        return "video_metadata"
        return None

    def _find_video_match(
        self,
        messages: List[Dict],
        current_videos: List[Dict],
    ) -> Tuple[Optional[Dict], int, Optional[str]]:
        """按文件SHA、感知帧、超限元数据的顺序查找历史视频。"""
        if not current_videos:
            return None, -1, None
        for match_type in (
            "video_sha256",
            "video_frame_hash",
            "video_metadata",
        ):
            for index, message in enumerate(messages):
                stored_videos = message.get("videos")
                if not isinstance(stored_videos, list):
                    continue
                if self._video_match_type(
                    current_videos,
                    stored_videos,
                    match_type,
                ):
                    return message, index, match_type
        return None, -1, None

    def _count_unique_senders_by_video(
        self,
        messages: List[Dict],
        current_videos: List[Dict],
    ) -> Tuple[int, List[str]]:
        """统计与当前任一视频匹配的不同发送者。"""
        sender_ids = {
            message.get("sender_id")
            for message in messages
            if message.get("sender_id")
            and isinstance(message.get("videos"), list)
            and self._video_match_type(
                current_videos,
                message["videos"],
            )
        }
        return len(sender_ids), list(sender_ids)

    # ==========================================================================
    # 4.8 相似度匹配方法
    # ==========================================================================

    def _get_url_tracking_rules(self) -> List[str]:
        """读取需要从纯URL指纹中移除的跟踪参数规则。"""
        rules = self.config.get(
            "url_tracking_params",
            DEFAULT_URL_TRACKING_PARAMS,
        )
        if not isinstance(rules, list):
            return list(DEFAULT_URL_TRACKING_PARAMS)
        normalized_rules = []
        for rule in rules:
            value = str(rule or "").strip().lower()
            if value:
                normalized_rules.append(value)
        return normalized_rules

    @staticmethod
    def _is_url_tracking_param(name: str, rules: List[str]) -> bool:
        """支持精确参数名和utm_*形式的前缀规则。"""
        lowered = name.lower()
        for rule in rules:
            if rule.endswith("*") and lowered.startswith(rule[:-1]):
                return True
            if lowered == rule:
                return True
        return False

    def _normalize_url_only(self, text: str) -> Optional[str]:
        """识别纯HTTP(S) URL并进行保守、确定性的规范化。"""
        candidate = str(text or "").strip()
        if candidate.startswith("<") and candidate.endswith(">"):
            candidate = candidate[1:-1].strip()
        if not candidate or len(candidate) > 8192:
            return None
        if not re.fullmatch(r"https?://[^\s<>]+", candidate, re.IGNORECASE):
            return None

        try:
            parsed = urlparse(candidate)
            if parsed.scheme.lower() not in ("http", "https"):
                return None
            if not parsed.hostname or parsed.username or parsed.password:
                return None

            scheme = parsed.scheme.lower()
            host = parsed.hostname.encode("idna").decode("ascii").lower()
            if ":" in host:
                host = f"[{host}]"
            port = parsed.port
            if port and not (
                (scheme == "http" and port == 80)
                or (scheme == "https" and port == 443)
            ):
                host = f"{host}:{port}"

            query_items = parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            tracking_rules = self._get_url_tracking_rules()
            query_items = [
                (name, value)
                for name, value in query_items
                if not self._is_url_tracking_param(name, tracking_rules)
            ]
            query_items.sort(key=lambda item: (item[0], item[1]))
            normalized_query = urlencode(query_items, doseq=True)
            return urlunparse(
                (
                    scheme,
                    host,
                    parsed.path or "/",
                    parsed.params,
                    normalized_query,
                    "",
                )
            )
        except (UnicodeError, ValueError):
            return None

    def _extract_url_fingerprint(self, text: str) -> Optional[Dict]:
        """返回纯URL消息的规范化哈希和主机名。"""
        if not self._get_bool_config("url_only_exact_match", True):
            return None
        normalized_url = self._normalize_url_only(text)
        if not normalized_url:
            return None
        parsed = urlparse(normalized_url)
        return {
            "url_hash": hashlib.sha256(
                normalized_url.encode("utf-8")
            ).hexdigest(),
            "url_host": parsed.hostname or "",
        }

    def _get_message_url_hash(self, message: Dict) -> Optional[str]:
        """兼容尚未保存url_hash的旧消息记录。"""
        if not self._get_bool_config("url_only_exact_match", True):
            return None
        content = str(message.get("content") or "")
        if message.get("url_only") or content.lstrip().lower().startswith(
            ("http://", "https://", "<http://", "<https://")
        ):
            fingerprint = self._extract_url_fingerprint(content)
            if fingerprint:
                return fingerprint["url_hash"]
        stored_hash = message.get("url_hash")
        return str(stored_hash) if stored_hash else None

    def _find_url_match(
        self,
        messages: List[Dict],
        url_hash: Optional[str],
        exclude_recent: int = 0,
    ) -> Tuple[Optional[Dict], int, Optional[str]]:
        """按规范化完整URL哈希查找历史消息。"""
        if not url_hash:
            return None, -1, None
        search_range = len(messages)
        if exclude_recent > 0:
            search_range -= exclude_recent
        for index in range(search_range):
            if self._get_message_url_hash(messages[index]) == url_hash:
                return messages[index], index, "url_hash"
        return None, -1, None

    def _count_unique_senders_by_url(
        self,
        messages: List[Dict],
        url_hash: str,
    ) -> Tuple[int, List[str]]:
        """统计发送同一规范化URL的不同用户。"""
        sender_ids = {
            message.get("sender_id")
            for message in messages
            if self._get_message_url_hash(message) == url_hash
            and message.get("sender_id")
        }
        return len(sender_ids), list(sender_ids)

    @staticmethod
    def _is_same_forward(
        message: Dict,
        forward_id: Optional[str],
        forward_hash: Optional[str],
        forward_text_hash: Optional[str] = None,
    ) -> Optional[str]:
        """判断历史消息是否为同一合并转发，返回命中依据。"""
        stored_id = message.get("forward_id")
        if forward_id and stored_id and str(stored_id) == str(forward_id):
            return "forward_id"
        stored_hash = message.get("forward_hash")
        if forward_hash and stored_hash and stored_hash == forward_hash:
            return "forward_hash"
        stored_text_hash = message.get("forward_text_hash")
        if (
            forward_text_hash
            and stored_text_hash
            and stored_text_hash == forward_text_hash
        ):
            return "forward_text_hash"
        return None

    def _find_forward_match(
        self,
        messages: List[Dict],
        forward_id: Optional[str],
        forward_hash: Optional[str],
        forward_text_hash: Optional[str] = None,
        exclude_recent: int = 0,
    ) -> Tuple[Optional[Dict], int, Optional[str]]:
        """按资源ID、完整内容哈希、发送者+文本指纹依次查找。"""
        search_range = len(messages) - exclude_recent if exclude_recent > 0 else len(messages)
        for index in range(search_range):
            match_type = self._is_same_forward(
                messages[index],
                forward_id,
                forward_hash,
                forward_text_hash,
            )
            if match_type:
                return messages[index], index, match_type
        return None, -1, None

    def _count_unique_senders_by_forward(
        self,
        messages: List[Dict],
        forward_id: Optional[str],
        forward_hash: Optional[str],
        forward_text_hash: Optional[str] = None,
    ) -> Tuple[int, List[str]]:
        """统计发送同一转发ID或精确内容指纹的不同用户。"""
        sender_ids = set()
        for message in messages:
            if not self._is_same_forward(
                message,
                forward_id,
                forward_hash,
                forward_text_hash,
            ):
                continue
            sender_id = message.get("sender_id")
            if sender_id:
                sender_ids.add(sender_id)
        return len(sender_ids), list(sender_ids)
    
    def _find_best_match(self, messages: List[Dict], embedding: List[float],
                         threshold: float, exclude_recent: int = 0) -> Tuple[Optional[Dict], int, float]:
        """查找最相似的历史消息（基于文本Embedding）"""
        best_sim, best_msg, best_idx = 0.0, None, -1
        search_range = len(messages) - exclude_recent if exclude_recent > 0 else len(messages)
        skipped_no_emb, skipped_dim_mismatch = 0, 0

        for i in range(search_range):
            msg = messages[i]
            emb = msg.get("embedding")
            if not emb:
                skipped_no_emb += 1
                continue
            if len(emb) != len(embedding):
                skipped_dim_mismatch += 1
                continue

            sim = self._cosine_similarity(embedding, emb)
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_msg, best_idx = msg, i

        # 只在有跳过的消息时记录
        if skipped_no_emb > 0 or skipped_dim_mismatch > 0:
            logger.debug(f"[Memory Reboot] 匹配统计: 跳过{skipped_no_emb}条无embedding, {skipped_dim_mismatch}条维度不匹配")

        return best_msg, best_idx, best_sim
    
    def _find_similar_image(self, messages: List[Dict], current_hash: str,
                            threshold: float = DEFAULT_IMAGE_HASH_THRESHOLD, exclude_recent: int = 0) -> Tuple[Optional[Dict], int, float]:
        """查找相似图片（基于图片哈希）"""
        best_sim, best_msg, best_idx = 0.0, None, -1
        search_range = len(messages) - exclude_recent if exclude_recent > 0 else len(messages)
        
        for i in range(search_range):
            msg = messages[i]
            msg_hash = msg.get("image_hash")
            if not msg_hash:
                continue
            sim = self._hash_similarity(current_hash, msg_hash)
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_msg, best_idx = msg, i
        return best_msg, best_idx, best_sim
    
    def _count_unique_senders(self, messages: List[Dict], embedding: List[float],
                               threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[int, List[str]]:
        """统计发送相似内容的不同用户数量"""
        sender_ids = set()
        for msg in messages:
            emb = msg.get("embedding")
            if emb and len(emb) == len(embedding) and self._cosine_similarity(embedding, emb) >= threshold:
                if msg.get("sender_id"):
                    sender_ids.add(msg.get("sender_id"))
        return len(sender_ids), list(sender_ids)
    
    def _count_unique_senders_by_hash(self, messages: List[Dict], image_hash: str,
                                       threshold: float = DEFAULT_IMAGE_HASH_THRESHOLD) -> Tuple[int, List[str]]:
        """统计发送相似图片的不同用户数量"""
        sender_ids = set()
        for msg in messages:
            msg_hash = msg.get("image_hash")
            if msg_hash and self._hash_similarity(image_hash, msg_hash) >= threshold:
                if msg.get("sender_id"):
                    sender_ids.add(msg.get("sender_id"))
        return len(sender_ids), list(sender_ids)
    
    def _get_context_around(self, messages: List[Dict], index: int, before: int = 40, after: int = 40) -> List[Dict]:
        """获取指定消息前后的上下文"""
        start = max(0, index - before)
        end = min(len(messages), index + after + 1)
        return [{"sender_name": m.get("sender_name"), "content": m.get("content"), "timestamp": m.get("timestamp")} 
                for m in messages[start:end]]
    
    # ==========================================================================
    # 4.8 时间格式化方法
    # ==========================================================================
    
    def _format_time(self, timestamp: float) -> str:
        """格式化时间戳为 MM-DD HH:MM:SS"""
        return datetime.datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M:%S")
    
    def _format_time_ago(self, timestamp: float) -> str:
        """格式化为"多久之前"的形式"""
        diff = time.time() - timestamp
        if diff < 60:
            return f"{int(diff)}秒前"
        elif diff < 3600:
            return f"{int(diff / 60)}分钟前"
        elif diff < 86400:
            return f"{int(diff / 3600)}小时前"
        else:
            return f"{int(diff / 86400)}天前"
    
    # ==========================================================================
    # 4.9 LLM交互方法
    # ==========================================================================
    
    async def _image_to_text(self, url: str) -> Optional[Tuple[str, str]]:
        """使用LLM识别图片内容，返回(内容, 类型)或None（表情包）"""
        provider_id = self.config.get("vision_provider_id", "")
        if not provider_id:
            logger.debug("[Memory Reboot] 图片识别: 未配置vision_provider_id")
            return None
        
        try:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.debug(f"[Memory Reboot] 图片识别: 未找到provider={provider_id}")
                return None
            
            logger.debug(f"[Memory Reboot] 图片识别: 开始调用视觉模型...")
            
            prompt = self.config.get("vision_prompt") or self.DEFAULT_VISION_PROMPT
            response = await provider.text_chat(prompt=prompt, contexts=[], image_urls=[url])
            
            if response and response.completion_text:
                result = response.completion_text.strip()
                
                try:
                    text = result
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                    
                    parsed = json.loads(text.strip())
                    raw_sticker = parsed.get("is_sticker", False)
                    if isinstance(raw_sticker, str):
                        is_sticker = raw_sticker.lower() in ("true", "yes", "1", "t")
                    else:
                        is_sticker = bool(raw_sticker)
                    content = parsed.get("content", "")
                    
                    if "type" in parsed and parsed.get("type", "").lower() == "emoji":
                        is_sticker = True
                    
                    if is_sticker:
                        return None
                    
                    return (content, "content")
                    
                except json.JSONDecodeError:
                    sticker_keywords = ["is_sticker\": true", "表情包", "emoji", "sticker", "熊猫头", "滑稽"]
                    if any(kw in result.lower() for kw in sticker_keywords):
                        return None
                    return (result, "unknown")
        except Exception as e:
            logger.error(f"[Memory Reboot] 图片识别失败: {e}")
        return None
    
    async def _judge_remind(self, content: str, sender_name: str, matched_msg: Dict,
                            history_ctx: List[Dict], current_ctx: List[Dict], unique_count: int) -> bool:
        """LLM判断是否需要提醒"""
        provider_id = self.config.get("judge_provider_id", "")
        if not provider_id:
            logger.debug("[Memory Reboot] LLM判断: 未配置judge_provider_id，跳过判断")
            return False
        
        try:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.debug(f"[Memory Reboot] LLM判断: 未找到provider={provider_id}")
                return False
            
            def fmt(m):
                ts = m.get('timestamp', 0)
                time_str = self._format_time(ts) if ts else "??:??:??"
                return f"[{time_str}] {m.get('sender_name', '?')}: {m.get('content', '')}"
            
            history_str = "\n".join([fmt(m) for m in history_ctx])
            current_str = "\n".join([fmt(m) for m in current_ctx])
            
            matched_time_ago = self._format_time_ago(matched_msg.get("timestamp", 0))
            matched_sender = matched_msg.get("sender_name", "未知")
            matched_content = matched_msg.get("content", "")[:300]
            min_senders = self.config.get("min_unique_senders", 3)
            
            prompt_template = self.config.get("judge_prompt") or self.DEFAULT_JUDGE_PROMPT
            
            try:
                prompt = prompt_template.format(
                    matched_time=self._format_time(matched_msg.get("timestamp", 0)),
                    matched_time_ago=matched_time_ago, matched_sender=matched_sender,
                    matched_content=matched_content, history_str=history_str,
                    current_str=current_str, sender_name=sender_name,
                    content=content, min_senders=min_senders,
                    unique_count=unique_count
                )
            except Exception:
                # 兼容旧版提示词如果不包含 {unique_count} 的情况
                prompt = self.DEFAULT_JUDGE_PROMPT.format(
                    matched_time=self._format_time(matched_msg.get("timestamp", 0)),
                    matched_time_ago=matched_time_ago, matched_sender=matched_sender,
                    matched_content=matched_content, history_str=history_str,
                    current_str=current_str, sender_name=sender_name,
                    content=content, min_senders=min_senders,
                    unique_count=unique_count
                )

            response = await provider.text_chat(prompt=prompt, contexts=[])
            
            if response and response.completion_text:
                text = response.completion_text
                if "```" in text:
                    text = text.split("```json")[-1].split("```")[0] if "```json" in text else text.split("```")[1].split("```")[0]
                try:
                    result = json.loads(text.strip())
                    should_remind = result.get("should_remind", False)
                    reason = result.get('reason', '无')
                    logger.debug(f"[Memory Reboot] LLM返回: should_remind={should_remind}")
                    logger.debug(f"[Memory Reboot] LLM理由: {reason}")
                    return should_remind
                except Exception as e:
                    logger.debug(f"[Memory Reboot] LLM返回JSON解析失败: {e}, 尝试文本匹配")
                    matched = '"should_remind": true' in text.lower()
                    logger.debug(f"[Memory Reboot] 文本匹配结果: {matched}")
                    return matched
        except Exception as e:
            logger.error(f"[Memory Reboot] LLM判断异常: {e}")
        return False

    def _get_forward_limits(self) -> Dict[str, int]:
        """读取并限制合并转发的资源消耗上限。"""
        def read_limit(key: str, default: int, hard_max: int, minimum: int = 1) -> int:
            try:
                value = int(self.config.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(value, hard_max))

        return {
            "depth": read_limit("forward_max_depth", DEFAULT_FORWARD_MAX_DEPTH, 5),
            "fetches": read_limit("forward_max_fetches", DEFAULT_FORWARD_MAX_FETCHES, 20),
            "timeout": read_limit(
                "forward_fetch_timeout",
                DEFAULT_FORWARD_FETCH_TIMEOUT,
                30,
            ),
            "nodes": read_limit("forward_max_nodes", DEFAULT_FORWARD_MAX_NODES, 500),
            "segments": read_limit(
                "forward_max_segments",
                DEFAULT_FORWARD_MAX_SEGMENTS,
                2000,
            ),
            "chars": read_limit(
                "forward_max_chars",
                DEFAULT_FORWARD_MAX_CHARS,
                50000,
                minimum=500,
            ),
            "visible_chars": read_limit(
                "forward_min_visible_chars",
                DEFAULT_FORWARD_MIN_VISIBLE_CHARS,
                200,
            ),
        }

    @staticmethod
    def _normalize_forward_text(value) -> str:
        """标准化转发节点中的文本，避免无意义的空白差异。"""
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _take_forward_text(state: Dict, text: str) -> str:
        """按总字符上限截取文本，并在截断时禁止生成内容哈希。"""
        remaining = state["limits"]["chars"] - state["chars"]
        if remaining <= 0:
            state["complete"] = False
            state["halted"] = True
            return ""
        if len(text) > remaining:
            state["chars"] += remaining
            state["complete"] = False
            state["halted"] = True
            return text[:remaining]
        state["chars"] += len(text)
        return text

    @staticmethod
    def _get_forward_nodes(payload) -> Optional[List[Dict]]:
        """兼容 OneBot、NapCat 等实现的 get_forward_msg 返回结构。"""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return None

        data = payload.get("data")
        if isinstance(data, dict):
            payload = data
        elif isinstance(data, list):
            return data

        for key in ("messages", "message", "nodes", "nodeList"):
            nodes = payload.get(key)
            if isinstance(nodes, list):
                return nodes
        return None

    @staticmethod
    def _is_unavailable_forward_error(error: Exception) -> bool:
        """判断转发资源是否已过期或属于协议端禁止读取的内层消息。"""
        message = str(error).lower()
        unavailable_markers = (
            "消息已过期",
            "内层消息",
            "message expired",
            "expired message",
            "inner message",
        )
        return any(marker in message for marker in unavailable_markers)

    async def _fetch_forward_payload(
        self,
        event: AstrMessageEvent,
        forward_id: str,
        state: Dict,
        is_nested: bool = False,
    ) -> Optional[Dict]:
        """通过 OneBot get_forward_msg 拉取合并转发内容。"""
        state["is_unreadable_nested_forward"] = False
        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            call_action = getattr(getattr(bot, "api", None), "call_action", None)
        if not callable(call_action):
            logger.warning("[Memory Reboot] 当前消息平台不支持 get_forward_msg")
            return None

        values = [forward_id]
        if forward_id.isdigit():
            values.append(int(forward_id))

        routing_params = {}
        self_id = getattr(getattr(event, "message_obj", None), "self_id", None)
        if self_id:
            routing_params["self_id"] = self_id

        last_error = None
        permanently_unavailable = False
        # OneBot v11标准参数是id；message_id仅作为少数实现的兼容回退。
        for key in ("id", "message_id"):
            for value in values:
                if state["fetches"] >= state["limits"]["fetches"]:
                    state["complete"] = False
                    return None
                state["fetches"] += 1
                try:
                    result = await asyncio.wait_for(
                        call_action(
                            "get_forward_msg",
                            **{key: value},
                            **routing_params,
                        ),
                        timeout=state["limits"]["timeout"],
                    )
                    if self._get_forward_nodes(result) is not None:
                        return result
                    last_error = ValueError("响应中没有转发节点")
                except asyncio.TimeoutError as e:
                    state["complete"] = False
                    last_error = e
                    break
                except Exception as e:
                    last_error = e
                    if self._is_unavailable_forward_error(e):
                        permanently_unavailable = True
                        break
            if isinstance(last_error, asyncio.TimeoutError) or permanently_unavailable:
                break

        error_text = (
            f": {type(last_error).__name__}: {last_error}"
            if last_error
            else ""
        )
        if is_nested and permanently_unavailable:
            state["is_unreadable_nested_forward"] = True
            logger.info(
                f"[Memory Reboot] 内层合并转发无法继续展开，"
                f"改用稳定占位符参与外层内容指纹: id={forward_id}"
            )
        else:
            logger.warning(
                f"[Memory Reboot] 无法展开合并转发 id={forward_id}{error_text}"
            )
        return None

    def _stable_forward_media_key(
        self,
        data: Dict,
        allow_name: bool = True,
        media_type: Optional[str] = None,
    ) -> Optional[str]:
        """提取图片、文件等媒体段中相对稳定的内容标识。"""
        if media_type == "video":
            for key in ("md5", "file_uuid", "file_id"):
                value = str(data.get(key) or "").strip()
                if value:
                    return f"{key}:{value.lower()}"

            filename = self._normalize_video_filename(
                data.get("file")
                or data.get("file_name")
                or data.get("name")
                or data.get("url")
            )
            file_size = self._parse_positive_int(
                data.get("file_size") or data.get("size")
            )
            metadata_hash = self._build_video_metadata_hash(
                filename,
                file_size,
            )
            if metadata_hash:
                return f"name_size_sha256:{metadata_hash}"

            for source_key in ("file", "url"):
                source = str(data.get(source_key) or "").strip()
                if not source:
                    continue
                parsed = urlparse(source)
                query = {
                    str(query_key).lower(): query_value
                    for query_key, query_value in parse_qs(parsed.query).items()
                }
                for query_key in (
                    "md5",
                    "fileid",
                    "file_id",
                    "file_uuid",
                    "uuid",
                ):
                    query_value = query.get(query_key)
                    if query_value:
                        return (
                            f"{query_key}:"
                            f"{str(query_value[0]).strip().lower()}"
                        )
            return None

        keys = ["md5", "file_uuid", "file_id", "file"]
        if allow_name:
            keys.append("name")
        for key in keys:
            value = data.get(key)
            if value in (None, ""):
                continue
            value = str(value).strip()
            if not value:
                continue
            if key == "file":
                parsed = urlparse(value)
                if parsed.scheme:
                    query = {
                        str(query_key).lower(): query_value
                        for query_key, query_value in parse_qs(parsed.query).items()
                    }
                    stable_query = None
                    for query_key in (
                        "md5",
                        "fileid",
                        "file_id",
                        "file_uuid",
                        "uuid",
                    ):
                        query_value = query.get(query_key)
                        if query_value:
                            stable_query = f"{query_key}={query_value[0]}"
                            break
                    # 无法识别稳定查询参数时保留完整URL，宁可漏报也不误报。
                    if stable_query:
                        value = (
                            f"{parsed.netloc}{unquote(parsed.path)}?"
                            f"{stable_query}"
                        )
                else:
                    value = value.split("?", 1)[0]
            if len(value) > 512:
                value = hashlib.sha256(value.encode("utf-8")).hexdigest()
            return f"{key}:{value.lower()}"
        return None

    def _sanitize_forward_data(self, value, state: Dict, depth: int = 0):
        """清理通用消息段中的临时字段，生成可稳定序列化的数据。"""
        if depth > 4:
            state["complete"] = False
            return "[data-depth-limit]"
        if isinstance(value, dict):
            volatile_keys = {
                "url",
                "path",
                "time",
                "timestamp",
                "seq",
                "message_id",
                "real_id",
                "uniseq",
                "resid",
            }
            return {
                str(key): self._sanitize_forward_data(item, state, depth + 1)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if str(key).lower() not in volatile_keys
            }
        if isinstance(value, list):
            if len(value) > 100:
                state["complete"] = False
                value = value[:100]
            return [
                self._sanitize_forward_data(item, state, depth + 1)
                for item in value
            ]
        if isinstance(value, str) and len(value) > 2048:
            state["complete"] = False
            return value[:2048]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    async def _canonicalize_forward_segment(
        self,
        event: AstrMessageEvent,
        segment,
        depth: int,
        state: Dict,
    ) -> Tuple[Optional[Dict], str]:
        """将一个转发消息段转换为稳定结构和可读摘要。"""
        if state["halted"]:
            return None, ""
        if state["segments"] >= state["limits"]["segments"]:
            state["complete"] = False
            state["halted"] = True
            return None, ""
        state["segments"] += 1

        if isinstance(segment, str):
            text = self._normalize_forward_text(segment)
            text = self._take_forward_text(state, text)
            state["visible_chars"] += len(text)
            return {"type": "text", "text": text}, text
        if not isinstance(segment, dict):
            text = self._normalize_forward_text(segment)
            text = self._take_forward_text(state, text)
            state["visible_chars"] += len(text)
            return {"type": "unknown", "text": text}, text

        segment_type = str(segment.get("type") or "unknown").lower()
        data = segment.get("data")
        if not isinstance(data, dict):
            data = segment

        if segment_type in ("text", "plain"):
            text = self._normalize_forward_text(
                data.get("text") or data.get("content")
            )
            text = self._take_forward_text(state, text)
            state["visible_chars"] += len(text)
            return {"type": "text", "text": text}, text

        if segment_type == "forward":
            nested_id = (
                data.get("id")
                or data.get("message_id")
                or data.get("resid")
            )
            if not nested_id:
                state["complete"] = False
                return {"type": "forward", "missing": True}, "[无法展开的嵌套转发]"
            nested_id = str(nested_id)
            if len(nested_id) > 512:
                state["complete"] = False
                return {"type": "forward", "invalid": True}, "[无效的嵌套转发]"
            nested = await self._expand_forward_id(
                event,
                nested_id,
                depth + 1,
                state,
            )
            if nested.get("unreadable"):
                return {
                    "type": "forward",
                    "unreadable": True,
                }, "[内层转发：内容不可读取]"
            nested_text = "\n".join(nested["display_lines"])
            display = "[嵌套转发]"
            if nested_text:
                display += f"\n{nested_text}"
            return {
                "type": "forward",
                "nodes": nested["canonical_nodes"],
            }, display

        if segment_type == "image":
            state["has_image"] = True
            media_key = self._stable_forward_media_key(data, allow_name=False)
            if not media_key:
                state["complete"] = False
                media_key = "unknown"
            else:
                state["stable_media"] += 1
            return {"type": "image", "key": media_key}, "[图片]"

        if segment_type in ("record", "video", "file"):
            if segment_type == "video":
                state["has_video"] = True
            media_key = self._stable_forward_media_key(
                data,
                media_type=segment_type,
            )
            if not media_key:
                state["complete"] = False
                media_key = "unknown"
            else:
                state["stable_media"] += 1
            labels = {"record": "语音", "video": "视频", "file": "文件"}
            return {
                "type": segment_type,
                "key": media_key,
            }, f"[{labels[segment_type]}]"

        if segment_type == "face":
            face_id = str(data.get("id") or "")
            return {"type": "face", "id": face_id}, f"[表情:{face_id}]"

        if segment_type == "at":
            qq = str(data.get("qq") or data.get("user_id") or "")
            return {"type": "at", "qq": qq}, f"[@{qq}]"

        if segment_type == "reply":
            return {"type": "reply"}, "[回复]"

        sanitized = self._sanitize_forward_data(data, state)
        serialized = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        serialized = self._take_forward_text(state, serialized)
        canonical = {
            "type": segment_type,
            "data_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        }
        return canonical, f"[{segment_type}]"

    async def _expand_forward_id(
        self,
        event: AstrMessageEvent,
        forward_id: str,
        depth: int,
        state: Dict,
    ) -> Dict:
        """在全局资源上限内递归展开一个合并转发资源。"""
        empty_result = {"canonical_nodes": [], "display_lines": []}
        if depth > state["limits"]["depth"]:
            state["complete"] = False
            return empty_result

        cache_key = (forward_id, state["limits"]["depth"] - depth)
        cached = state["cache"].get(cache_key)
        if cached is not None:
            return cached
        if forward_id in state["active_ids"]:
            state["complete"] = False
            return empty_result
        if state["fetches"] >= state["limits"]["fetches"]:
            state["complete"] = False
            return empty_result

        state["active_ids"].add(forward_id)
        try:
            payload = await self._fetch_forward_payload(
                event,
                forward_id,
                state,
                is_nested=depth > 1,
            )
            can_use_placeholder = (
                payload is None
                and depth > 1
                and state["complete"]
                and state.get("is_unreadable_nested_forward", False)
            )
            if can_use_placeholder:
                # 协议端不提供内层内容，且其ID并不稳定。因此仅保留一个
                # 固定占位符，并在最终生成哈希前检查外层可见内容是否充足。
                state["unreadable_forwards"] += 1
                result = {
                    "canonical_nodes": [],
                    "display_lines": ["[内层转发：内容不可读取]"],
                    "unreadable": True,
                }
                state["cache"][cache_key] = result
                return result
            nodes = self._get_forward_nodes(payload)
            if nodes is None:
                state["complete"] = False
                return empty_result

            canonical_nodes = []
            display_lines = []
            for raw_node in nodes:
                if state["halted"]:
                    break
                if state["nodes"] >= state["limits"]["nodes"]:
                    state["complete"] = False
                    state["halted"] = True
                    break
                state["nodes"] += 1

                node = raw_node
                if (
                    isinstance(raw_node, dict)
                    and str(raw_node.get("type") or "").lower() == "node"
                    and isinstance(raw_node.get("data"), dict)
                ):
                    node = raw_node["data"]
                if not isinstance(node, dict):
                    state["complete"] = False
                    continue

                sender = node.get("sender")
                if not isinstance(sender, dict):
                    sender = {}
                sender_id = self._normalize_forward_text(
                    sender.get("user_id")
                    or sender.get("uin")
                    or node.get("user_id")
                    or node.get("uin")
                    or ""
                )
                sender_id = self._take_forward_text(state, sender_id)
                sender_name = self._normalize_forward_text(
                    sender.get("nickname")
                    or sender.get("card")
                    or node.get("nickname")
                    or node.get("name")
                    or sender_id
                    or "未知"
                )
                sender_name = self._take_forward_text(state, sender_name) or "未知"

                raw_content = node.get("message")
                if raw_content is None:
                    raw_content = node.get("content")
                if isinstance(raw_content, str):
                    raw_content = [
                        {"type": "text", "data": {"text": raw_content}}
                    ]
                elif isinstance(raw_content, dict):
                    raw_content = [raw_content]
                elif not isinstance(raw_content, list):
                    raw_content = []

                canonical_segments = []
                display_segments = []
                for segment in raw_content:
                    canonical, display = await self._canonicalize_forward_segment(
                        event,
                        segment,
                        depth,
                        state,
                    )
                    if canonical is not None:
                        canonical_segments.append(canonical)
                    if display:
                        display_segments.append(display)
                    if state["halted"]:
                        break

                canonical_nodes.append(
                    {
                        "sender_id": sender_id,
                        "sender_name": sender_name,
                        "segments": canonical_segments,
                    }
                )
                display_content = " ".join(display_segments) or "[空消息]"
                display_lines.append(f"{sender_name}: {display_content}")

            result = {
                "canonical_nodes": canonical_nodes,
                "display_lines": display_lines,
            }
            state["cache"][cache_key] = result
            return result
        finally:
            state["active_ids"].discard(forward_id)

    def _is_forward_debug_enabled(self) -> bool:
        """读取转发指纹诊断开关，并兼容字符串形式的布尔值。"""
        return self._get_bool_config("forward_debug_log", False)

    def _is_forward_sender_text_match_enabled(self) -> bool:
        """读取发送者ID+文本精确匹配开关。"""
        return self._get_bool_config("forward_sender_text_match", True)

    def _is_video_debug_enabled(self) -> bool:
        """读取视频特征指纹诊断开关。"""
        return self._get_bool_config("video_debug_log", False)

    def _get_bool_config(self, key: str, default: bool) -> bool:
        """读取布尔配置，并兼容字符串形式。"""
        enabled = self.config.get(key, default)
        if isinstance(enabled, str):
            return enabled.strip().lower() in ("1", "true", "yes", "on")
        return bool(enabled)

    @staticmethod
    def _serialize_forward_value(value) -> str:
        """使用内容指纹统一的规则序列化规范化数据。"""
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _build_forward_sender_text_nodes(
        canonical_nodes: List[Dict],
    ) -> Tuple[List[Dict], int]:
        """仅保留节点顺序、发送者ID和文本，并统计含文本节点数。"""
        sender_text_nodes = []
        text_node_count = 0
        for node in canonical_nodes:
            texts = [
                segment.get("text")
                for segment in node.get("segments", [])
                if isinstance(segment, dict)
                and isinstance(segment.get("text"), str)
                and segment["text"]
            ]
            if texts:
                text_node_count += 1
            sender_text_nodes.append(
                {
                    "sender_id": node.get("sender_id", ""),
                    "texts": texts,
                }
            )
        return sender_text_nodes, text_node_count

    def _log_forward_debug(
        self,
        forward_id: str,
        canonical_nodes: List[Dict],
        state: Dict,
        forward_hash: Optional[str],
        forward_text_hash: Optional[str],
        sender_text_nodes: List[Dict],
        text_node_count: int,
        has_enough_visible_content: bool,
    ) -> None:
        """分段输出内容指纹的完整规范化输入，便于对比两次转发。"""
        if not self._is_forward_debug_enabled():
            return

        if not state["complete"]:
            hash_status = "incomplete"
        elif not canonical_nodes:
            hash_status = "no_nodes"
        elif not has_enough_visible_content:
            hash_status = "insufficient_visible_content"
        else:
            hash_status = "generated"

        if not self._is_forward_sender_text_match_enabled():
            text_hash_status = "disabled"
        elif not state["complete"]:
            text_hash_status = "incomplete"
        elif state["visible_chars"] < DEFAULT_FORWARD_TEXT_MATCH_MIN_CHARS:
            text_hash_status = "insufficient_chars"
        elif text_node_count < DEFAULT_FORWARD_TEXT_MATCH_MIN_NODES:
            text_hash_status = "insufficient_text_nodes"
        else:
            text_hash_status = "generated"

        node_hashes = [
            hashlib.sha256(
                self._serialize_forward_value(node).encode("utf-8")
            ).hexdigest()
            for node in canonical_nodes
        ]
        payload = {
            "forward_id": forward_id,
            "hash_version": FORWARD_HASH_VERSION,
            "forward_hash": forward_hash,
            "hash_status": hash_status,
            "forward_text_hash": forward_text_hash,
            "text_hash_status": text_hash_status,
            "limits": state["limits"],
            "stats": {
                "fetches": state["fetches"],
                "nodes": state["nodes"],
                "segments": state["segments"],
                "chars": state["chars"],
                "visible_chars": state["visible_chars"],
                "text_nodes": text_node_count,
                "stable_media": state["stable_media"],
                "unreadable_forwards": state["unreadable_forwards"],
                "complete": state["complete"],
            },
            "canonical_node_hashes": node_hashes,
            "canonical_nodes": canonical_nodes,
            "sender_text_nodes": sender_text_nodes,
        }
        serialized = self._serialize_forward_value(payload)
        chunk_size = 1800
        total_chunks = max(1, (len(serialized) + chunk_size - 1) // chunk_size)

        logger.warning(
            "[Memory Reboot][ForwardDebug] 诊断已启用；以下日志包含"
            f"聊天正文和发送者信息。id={forward_id}, 分段={total_chunks}"
        )
        for index in range(total_chunks):
            start = index * chunk_size
            chunk = serialized[start:start + chunk_size]
            logger.info(
                f"[Memory Reboot][ForwardDebug] id={forward_id} "
                f"part={index + 1}/{total_chunks} data={chunk}"
            )
        logger.info(f"[Memory Reboot][ForwardDebug] id={forward_id} END")

    async def _extract_forward_content(
        self,
        event: AstrMessageEvent,
        forward_id: str,
        known_message: Optional[Dict] = None,
    ) -> Dict:
        """展开合并转发并生成内容哈希；相同 ID 优先复用历史结果。"""
        debug_enabled = self._is_forward_debug_enabled()
        sender_text_match_enabled = self._is_forward_sender_text_match_enabled()
        known_text_hash_ready = (
            not sender_text_match_enabled
            or bool((known_message or {}).get("forward_text_hash"))
            or (known_message or {}).get("forward_text_hash_eligible") is False
        )
        can_reuse_known = (
            known_message is not None
            and not debug_enabled
            and known_text_hash_ready
            and known_message.get("forward_hash_version") == FORWARD_HASH_VERSION
            and (
                known_message.get("forward_hash")
                or not known_message.get("forward_truncated", False)
            )
        )
        if can_reuse_known:
            return {
                "forward_content": (
                    known_message.get("forward_content")
                    or known_message.get("content")
                    or "[合并转发消息]"
                ),
                "forward_hash": known_message.get("forward_hash"),
                "forward_text_hash": (
                    known_message.get("forward_text_hash")
                    if sender_text_match_enabled
                    else None
                ),
                "forward_text_hash_eligible": known_message.get(
                    "forward_text_hash_eligible",
                    False,
                ),
                "forward_node_count": known_message.get("forward_node_count", 0),
                "forward_truncated": known_message.get("forward_truncated", False),
                "forward_partial": known_message.get("forward_partial", False),
                "forward_hash_version": FORWARD_HASH_VERSION,
                "has_image": known_message.get("has_image", False),
                "has_video": known_message.get("has_video", False),
                "reused": True,
            }
        if known_message is not None:
            reason = "诊断已开启" if debug_enabled else "指纹需要升级或重新生成"
            logger.info(f"[Memory Reboot] 历史转发记录不复用: {reason}")

        state = {
            "limits": self._get_forward_limits(),
            "fetches": 0,
            "nodes": 0,
            "segments": 0,
            "chars": 0,
            "complete": True,
            "halted": False,
            "has_image": False,
            "has_video": False,
            "visible_chars": 0,
            "stable_media": 0,
            "unreadable_forwards": 0,
            "active_ids": set(),
            "cache": {},
        }
        expanded = await self._expand_forward_id(event, forward_id, 1, state)
        display_lines = expanded["display_lines"]
        forward_content = "[合并转发]"
        if display_lines:
            forward_content += "\n" + "\n".join(display_lines)
        if not state["complete"]:
            forward_content += "\n[内容未完整展开]"

        forward_hash = None
        canonical_nodes = expanded["canonical_nodes"]
        sender_text_nodes, text_node_count = (
            self._build_forward_sender_text_nodes(canonical_nodes)
        )
        has_unreadable = state["unreadable_forwards"] > 0
        has_enough_visible_content = (
            not has_unreadable
            or state["visible_chars"] >= state["limits"]["visible_chars"]
            or state["stable_media"] > 0
        )
        if state["complete"] and canonical_nodes and has_enough_visible_content:
            serialized = self._serialize_forward_value(canonical_nodes)
            forward_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        forward_text_hash = None
        text_hash_eligible = (
            state["complete"]
            and state["visible_chars"] >= DEFAULT_FORWARD_TEXT_MATCH_MIN_CHARS
            and text_node_count >= DEFAULT_FORWARD_TEXT_MATCH_MIN_NODES
        )
        if sender_text_match_enabled and text_hash_eligible:
            serialized_text = self._serialize_forward_value(sender_text_nodes)
            forward_text_hash = hashlib.sha256(
                serialized_text.encode("utf-8")
            ).hexdigest()

        self._log_forward_debug(
            forward_id,
            canonical_nodes,
            state,
            forward_hash,
            forward_text_hash,
            sender_text_nodes,
            text_node_count,
            has_enough_visible_content,
        )

        logger.info(
            f"[Memory Reboot] 合并转发展开: id={forward_id}, "
            f"层数上限={state['limits']['depth']}, 拉取={state['fetches']}, "
            f"节点={state['nodes']}, 消息段={state['segments']}, "
            f"字符={state['chars']}, 可见正文={state['visible_chars']}, "
            f"文本节点={text_node_count}, "
            f"稳定媒体={state['stable_media']}, "
            f"不可读内层={state['unreadable_forwards']}, "
            f"内容指纹={'已生成' if forward_hash else '未生成'}, "
            f"发送者+文本指纹={'已生成' if forward_text_hash else '未生成'}"
        )
        return {
            "forward_content": forward_content,
            "forward_hash": forward_hash,
            "forward_text_hash": forward_text_hash,
            "forward_text_hash_eligible": text_hash_eligible,
            "forward_node_count": state["nodes"],
            "forward_truncated": not state["complete"],
            "forward_partial": has_unreadable,
            "forward_hash_version": FORWARD_HASH_VERSION,
            "has_image": state["has_image"],
            "has_video": state["has_video"],
            "reused": False,
        }
    
    def _build_message_summary(self, matched_msg: Dict) -> str:
        """生成匹配到的历史消息摘要。"""
        sender_name = str(matched_msg.get("sender_name") or "未知")

        timestamp = matched_msg.get("timestamp", 0)
        try:
            timestamp = float(timestamp)
            if timestamp <= 0:
                raise ValueError("无效时间戳")
            time_text = self._format_time_ago(timestamp)
        except (TypeError, ValueError, OverflowError, OSError):
            time_text = "时间未知"

        content = re.sub(r"\s+", " ", str(matched_msg.get("content") or "")).strip()
        if not content:
            content = "（无文字内容）"
        elif len(content) > REMINDER_SUMMARY_MAX_CHARS:
            content = content[:REMINDER_SUMMARY_MAX_CHARS].rstrip() + "…"

        is_forward = bool(matched_msg.get("forward_id"))
        details = [sender_name, time_text]
        if is_forward and matched_msg.get("forward_node_count"):
            details.append(f"转发{matched_msg['forward_node_count']}条")
        if is_forward and (
            matched_msg.get("forward_truncated")
            or matched_msg.get("forward_partial")
        ):
            details.append("部分内容")
        if matched_msg.get("has_image"):
            details.append("含图片")
        if matched_msg.get("has_video"):
            details.append("含视频")
        return f"\n\n📌 {' · '.join(details)}\n{content}"

    async def _send_reminder(self, event: AstrMessageEvent, matched_msg: Dict):
        """发送提醒图片，并附上匹配到的历史消息摘要。"""
        # 尝试获取消息ID用于引用回复
        msg_id = None
        try:
            msg_id = getattr(event.message_obj, "message_id", None)
        except Exception:
            msg_id = None
        
        chain = []
        # 使用引用回复
        if msg_id:
            chain.append(Reply(id=str(msg_id)))
        
        img_path = os.path.join(self.plugin_dir, REMINDER_IMAGE_FILENAME)
        if os.path.exists(img_path):
            chain.append(Image.fromFileSystem(img_path))
        else:
            chain.append(Plain("这个话题之前已经有人讨论过了哦~"))
        chain.append(Plain(self._build_message_summary(matched_msg)))
        yield event.chain_result(chain)

    @staticmethod
    def _embedding_content(result: Dict, content: str) -> str:
        """视频消息只使用用户正文，排除插件合成的媒体标签。"""
        if result.get("has_video"):
            return str(result.get("video_text") or "").strip()
        return content

    async def _extract_content(
        self,
        event: AstrMessageEvent,
        group_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """提取普通消息或合并转发内容。"""
        text = event.message_str.strip() if event.message_str else ""
        image_sources = []
        forward_components = []
        video_infos = self._extract_video_components(event)
        if hasattr(event, "message_obj") and event.message_obj:
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    image_source = (
                        getattr(comp, "path", None)
                        or getattr(comp, "url", None)
                        or getattr(comp, "file", None)
                    )
                    if image_source:
                        image_sources.append(image_source)
                elif isinstance(comp, Forward):
                    forward_components.append(comp)

        if forward_components:
            forward_id = str(getattr(forward_components[0], "id", "") or "").strip()
            if not forward_id:
                logger.warning("[Memory Reboot] 收到合并转发，但缺少 forward_id")
                return {
                    "content": text,
                    "image_source": None,
                    "has_image": False,
                } if text else None
            if len(forward_id) > 512:
                logger.warning("[Memory Reboot] 合并转发ID异常过长，已跳过展开")
                return {
                    "content": text,
                    "image_source": None,
                    "has_image": False,
                } if text else None

            known_message = None
            if group_id:
                messages = self._load_messages(group_id)
                known_message = next(
                    (
                        message
                        for message in messages
                        if str(message.get("forward_id") or "") == forward_id
                    ),
                    None,
                )
            forward_data = await self._extract_forward_content(
                event,
                forward_id,
                known_message,
            )
            forward_content = forward_data["forward_content"]
            content = f"{text}\n{forward_content}" if text else forward_content
            if forward_data["reused"]:
                logger.info(
                    f"[Memory Reboot] 合并转发ID已记录，复用展开结果: {forward_id}"
                )
            return {
                "content": content,
                "image_source": None,
                "has_image": forward_data["has_image"],
                "forward_id": forward_id,
                "forward_hash": forward_data["forward_hash"],
                "forward_text_hash": forward_data["forward_text_hash"],
                "forward_text_hash_eligible": forward_data[
                    "forward_text_hash_eligible"
                ],
                "forward_content": forward_content,
                "forward_node_count": forward_data["forward_node_count"],
                "forward_truncated": forward_data["forward_truncated"],
                "forward_partial": forward_data["forward_partial"],
                "forward_hash_version": forward_data["forward_hash_version"],
                "has_video": forward_data["has_video"],
            }

        if (
            video_infos
            and self._get_bool_config("video_duplicate_enabled", True)
        ):
            names = [
                video_info["name"]
                for video_info in video_infos
                if video_info.get("name")
            ]
            video_label = "[视频]"
            if names:
                video_label += " " + "、".join(names)
            return {
                "content": f"{text} {video_label}".strip(),
                "video_text": text,
                "image_source": None,
                "has_image": False,
                "has_video": True,
                "video_infos": video_infos,
            }

        if image_sources:
            for image_source in image_sources:
                result = await self._image_to_text(image_source)
                if result:
                    img_text, _ = result
                    content = f"{text} [图片内容: {img_text}]" if text else f"[图片内容: {img_text}]"
                    logger.debug(f"[Memory Reboot] 图片转文本成功")
                    return {
                        "content": content,
                        "image_source": image_source,
                        "has_image": True,
                    }
                else:
                    logger.info(f"[Memory Reboot] 跳过表情包")
            if not text:
                logger.debug(f"[Memory Reboot] 无有效内容，跳过")
                return None
        url_fingerprint = (
            self._extract_url_fingerprint(text)
            if text and not image_sources
            else None
        )
        return {
            "content": text,
            "image_source": None,
            "has_image": False,
            "has_video": False,
            "url_only": bool(url_fingerprint),
            "url_hash": (
                url_fingerprint.get("url_hash")
                if url_fingerprint
                else None
            ),
            "url_host": (
                url_fingerprint.get("url_host")
                if url_fingerprint
                else None
            ),
        } if text else None
    
    # ==========================================================================
    # 4.10 主消息处理器
    # ==========================================================================
    
    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息的主入口"""
        group_id = event.get_group_id()
        
        # 群组检查
        if not group_id:
            logger.debug("[Memory Reboot] 跳过: 非群组消息")
            return
        if not self._is_group_enabled(group_id):
            logger.debug(f"[Memory Reboot] 跳过: 群{group_id}在黑名单中")
            return
        
        sender_id = str(event.get_sender_id() or "")
        self_id = str(event.get_self_id() or "")
        if self_id and sender_id == self_id:
            logger.debug(
                f"[Memory Reboot] 跳过: 机器人自身消息({self_id})"
            )
            return

        sender_name = event.get_sender_name() or sender_id
        logger.debug(f"[Memory Reboot] ━━━ 收到消息 ━━━ 群:{group_id} 发送者:{sender_name}({sender_id})")
        
        result = await self._extract_content(event, group_id)
        if not result:
            logger.debug(f"[Memory Reboot] 跳过: 内容提取失败或为表情包")
            return
        
        content = result["content"]
        image_source = result.get("image_source")
        forward_id = result.get("forward_id")
        forward_hash = result.get("forward_hash")
        forward_text_hash = result.get("forward_text_hash")
        forward_truncated = result.get("forward_truncated", False)
        forward_partial = result.get("forward_partial", False)
        url_only = result.get("url_only", False)
        url_hash = result.get("url_hash")
        url_host = result.get("url_host")
        has_video = bool(result.get("has_video"))
        video_infos = result.get("video_infos") or []

        # 过滤：最小长度检查
        if not image_source and not forward_id and not has_video:
            min_length = self.config.get("min_text_length", DEFAULT_MIN_TEXT_LENGTH)
            if len(content) < min_length:
                logger.debug(f"[Memory Reboot] 跳过: 短文本({len(content)}<{min_length})")
                return

        # 过滤：正则表达式检查
        for pattern in self.config.get("ignore_regex", []):
            try:
                # 使用 fullmatch 确保完全匹配，避免误伤包含关键词的普通句子
                # 例如：pattern="何意味" 时
                # fullmatch: 匹配 "何意味"，不匹配 "你这是何意味啊"
                # search: 两者都匹配
                if re.fullmatch(pattern, content):
                    logger.info(f"[Memory Reboot] 已正则匹配: {pattern}")
                    return
            except re.error:
                pass

        # 过滤：插件命令自动过滤
        if self._is_plugin_command(content):
            logger.info(f"[Memory Reboot] 跳过: 检测到插件命令")
            return

        # 加载消息（使用内存缓存）
        messages = self._load_messages(group_id)
        logger.debug(f"[Memory Reboot] 历史消息: {len(messages)}条（缓存）")
        
        # 定期清理图片缓存（每100条消息触发一次）
        if len(messages) % 100 == 0 and len(messages) > 0:
            self._cleanup_image_cache(group_id)

        url_matched, url_idx, url_match_type = self._find_url_match(
            messages,
            url_hash,
        )
        if url_only:
            logger.info(
                f"[Memory Reboot] 纯URL匹配: "
                f"{url_match_type or '未命中'} "
                f"(host={url_host}, hash={url_hash[:12] if url_hash else '无'})"
            )

        # 合并转发先按资源ID和稳定内容哈希精确匹配。
        forward_matched, forward_idx, forward_match_type = self._find_forward_match(
            messages,
            forward_id,
            forward_hash,
            forward_text_hash,
        )
        if forward_id:
            logger.info(
                f"[Memory Reboot] 合并转发匹配: "
                f"{forward_match_type or '未命中'} "
                f"(内容哈希={'有' if forward_hash else '无'}, "
                f"发送者+文本指纹={'有' if forward_text_hash else '无'})"
            )

        video_fingerprints = await self._analyze_videos(video_infos, event)
        video_matched, video_idx, video_match_type = self._find_video_match(
            messages,
            video_fingerprints,
        )
        if has_video and not forward_id:
            self._log_video_debug(
                event,
                group_id,
                video_fingerprints,
                video_match_type,
                video_matched,
            )
        usable_video_fingerprint = any(
            video.get("sha256")
            or video.get("frame_hashes")
            or (
                video.get("analysis") == "metadata"
                and video.get("metadata_hash")
            )
            for video in video_fingerprints
        )
        if has_video and not forward_id:
            analysis_modes = ",".join(
                str(video.get("analysis") or "unavailable")
                for video in video_fingerprints
            ) or "无指纹"
            logger.info(
                f"[Memory Reboot] 视频匹配: "
                f"{video_match_type or '未命中'} "
                f"(数量={len(video_fingerprints)}, 分析={analysis_modes})"
            )

        embedding_content = self._embedding_content(result, content)

        # 精确命中的转发无需再次生成embedding；未完整或含不可读内层
        # 的转发也不参与语义匹配，避免隐藏内容不同却被误判。
        # 视频只允许使用用户实际输入的正文，不能把合成的文件名标签
        # 送入Embedding，否则视频取回失败时极易产生假阳性。
        if (
            url_only
            or forward_matched
            or forward_truncated
            or forward_partial
            or usable_video_fingerprint
        ):
            embedding = None
        elif has_video and not embedding_content:
            embedding = None
            logger.info(
                "[Memory Reboot] 视频指纹不可用且消息无正文，跳过Embedding"
            )
        else:
            embedding = await self._get_embedding(embedding_content)
        logger.debug(f"[Memory Reboot] Embedding: {'成功获取' if embedding else '获取失败'}, 维度={len(embedding) if embedding else 0}")
        now = time.time()
        cached_image, image_hash = None, None
        if image_source:
            cached_image, image_hash = await self._cache_image(image_source, now, group_id)
            logger.debug(f"[Memory Reboot] 图片缓存: {'成功' if cached_image else '失败'}, 哈希={'有' if image_hash else '无'}")
        
        # 创建当前消息记录
        msg = {
            "id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "sender_name": sender_name,
            "content": content,
            "timestamp": now,
            "embedding": embedding,
            "has_image": bool(result.get("has_image")),
            "has_video": has_video,
            "cached_image": cached_image,
            "image_hash": image_hash,
            "videos": video_fingerprints,
            "forward_id": forward_id,
            "forward_hash": forward_hash,
            "forward_text_hash": forward_text_hash,
            "forward_text_hash_eligible": result.get(
                "forward_text_hash_eligible",
                False,
            ),
            "forward_content": result.get("forward_content"),
            "forward_node_count": result.get("forward_node_count", 0),
            "forward_truncated": forward_truncated,
            "forward_partial": forward_partial,
            "forward_hash_version": result.get("forward_hash_version"),
            "url_only": url_only,
            "url_hash": url_hash,
            "url_host": url_host,
        }
        
        # 注意：此时不追加到 messages 列表，而是创建一个包含当前消息的临时列表用于匹配
        # 实际的追加会在 _append_message 中完成
        messages_with_current = messages + [msg]
        
        # 相似度匹配（使用包含当前消息的列表，但排除最后一条）
        matched_msg = url_matched or forward_matched or video_matched
        if url_matched:
            matched_idx = url_idx
        elif forward_matched:
            matched_idx = forward_idx
        else:
            matched_idx = video_idx
        match_type = url_match_type or forward_match_type or video_match_type
        text_threshold = self.config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        image_hash_threshold = self.config.get("image_hash_threshold", DEFAULT_IMAGE_HASH_THRESHOLD)
        
        if embedding:
            emb_matched, emb_idx, emb_sim = self._find_best_match(messages_with_current, embedding, text_threshold, exclude_recent=1)
            logger.info(f"[Memory Reboot] 文本相似度: {emb_sim:.4f} (阈值{text_threshold})")
            if emb_matched:
                matched_msg, matched_idx, match_type = emb_matched, emb_idx, "embedding"

        if image_hash:
            img_matched, img_idx, img_sim = self._find_similar_image(messages_with_current, image_hash, image_hash_threshold, exclude_recent=1)
            logger.info(f"[Memory Reboot] 图片哈希相似度: {img_sim:.4f} (阈值{image_hash_threshold})")
            if not matched_msg and img_matched:
                matched_msg, matched_idx, match_type = img_matched, img_idx, "image_hash"
            elif (
                matched_msg
                and match_type not in (
                    "url_hash",
                    "forward_id",
                    "forward_hash",
                    "forward_text_hash",
                    "video_sha256",
                    "video_frame_hash",
                    "video_metadata",
                )
                and img_matched
                and img_matched.get("timestamp", 0) < matched_msg.get("timestamp", 0)
            ):
                matched_msg, matched_idx, match_type = img_matched, img_idx, "image_hash"
        
        if not matched_msg:
            logger.debug(f"[Memory Reboot] 未找到匹配消息，仅保存记录")
            self._append_message(group_id, msg)
            return
        
        # 人数检测（使用包含当前消息的列表）
        min_unique_senders = self.config.get("min_unique_senders", 3)
        if match_type == "url_hash" and url_hash:
            unique_count, sender_list = self._count_unique_senders_by_url(
                messages_with_current,
                url_hash,
            )
            logger.debug(
                f"[Memory Reboot] 人数检测(纯URL): "
                f"{unique_count}人发送过相同链接"
            )
        elif match_type in ("forward_id", "forward_hash", "forward_text_hash"):
            unique_count, sender_list = self._count_unique_senders_by_forward(
                messages_with_current,
                forward_id,
                forward_hash,
                forward_text_hash,
            )
            logger.debug(
                f"[Memory Reboot] 人数检测(合并转发): "
                f"{unique_count}人发送过相同内容"
            )
        elif match_type in (
            "video_sha256",
            "video_frame_hash",
            "video_metadata",
        ):
            unique_count, sender_list = self._count_unique_senders_by_video(
                messages_with_current,
                video_fingerprints,
            )
            logger.debug(
                f"[Memory Reboot] 人数检测(视频): "
                f"{unique_count}人发送过相同视频"
            )
        elif match_type == "embedding" and embedding:
            unique_count, sender_list = self._count_unique_senders(messages_with_current, embedding, text_threshold)
            logger.debug(f"[Memory Reboot] 人数检测(文本): {unique_count}人发送过相似内容")
        elif match_type == "image_hash" and image_hash:
            unique_count, sender_list = self._count_unique_senders_by_hash(messages_with_current, image_hash, image_hash_threshold)
            logger.debug(f"[Memory Reboot] 人数检测(图片): {unique_count}人发送过相似图片")
        else:
            unique_count = 1
            logger.debug(f"[Memory Reboot] 人数检测: 无法统计(embedding或hash缺失)")
        
        if sender_id == matched_msg.get("sender_id"):
            logger.debug(f"[Memory Reboot] ✗ 跳过: 同一用户({sender_name})重发自己的内容")
            self._append_message(group_id, msg)
            return
        
        if unique_count < min_unique_senders:
            logger.debug(f"[Memory Reboot] ✗ 跳过: 不同用户数{unique_count}<{min_unique_senders}(阈值)")
            self._append_message(group_id, msg)
            return
        
        logger.debug(f"[Memory Reboot] ✓ 通过人数检测: {unique_count}人>={min_unique_senders}人")
        
        # 冷却时间检查
        time_diff = now - matched_msg.get("timestamp", 0)
        cooldown = self.config.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)
        if time_diff < cooldown:
            logger.debug(f"[Memory Reboot] ✗ 跳过: 冷却时间内({int(time_diff)}s<{cooldown}s)")
            self._append_message(group_id, msg)
            return
        
        logger.debug(f"[Memory Reboot] ✓ 通过冷却检测: 间隔{int(time_diff)}s>={cooldown}s")
        
        # 检查是否启用LLM判断
        enable_llm_judge = self.config.get("enable_llm_judge", True)
        
        if enable_llm_judge:
            # LLM判断
            history_ctx = self._get_context_around(messages_with_current, matched_idx, before=40, after=40)
            current_ctx = [{"sender_name": m.get("sender_name"), "content": m.get("content"), "timestamp": m.get("timestamp")}
                           for m in messages_with_current[:-1][-40:]]

            logger.debug(f"[Memory Reboot] 进入LLM判断: 匹配={match_type}, 来自={matched_msg.get('sender_name')}, {self._format_time_ago(matched_msg.get('timestamp', 0))}")

            should_remind = await self._judge_remind(content, sender_name, matched_msg, history_ctx, current_ctx, unique_count)
        else:
            # 关闭LLM判断时，直接触发提醒
            logger.debug(f"[Memory Reboot] LLM判断已关闭，直接触发提醒")
            should_remind = True

        self._append_message(group_id, msg)

        if should_remind:
            logger.info(f"[Memory Reboot] 最终判断: 触发提醒 -> {sender_name}")
            async for result in self._send_reminder(event, matched_msg):
                yield result
        else:
            logger.info(f"[Memory Reboot] 最终判断: 不提醒")
    
    # ==========================================================================
    # 4.11 命令处理器
    # ==========================================================================
    
    @filter.command("记忆状态")
    async def check_status(self, event: AstrMessageEvent):
        """查看插件状态（仅管理员）"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("请在群聊中使用")
            return
        if not event.is_admin():
            yield event.plain_result("❌ 仅管理员可执行")
            return
        
        messages = self._load_messages(group_id)
        img_path = os.path.join(self.plugin_dir, REMINDER_IMAGE_FILENAME)
        
        # 获取黑名单状态
        is_blocked = not self._is_group_enabled(group_id)
        blocked_status = "🚫 已拉黑 (不记录/不提醒)" if is_blocked else "✅ 正常工作"

        # 获取插件命令过滤状态
        auto_filter_enabled = self.config.get('auto_filter_commands', True)
        if auto_filter_enabled and HAS_COMMAND_FILTER:
            plugin_commands = self._get_all_plugin_commands()
            cmd_filter_status = f"✅ 已启用 (检测到{len(plugin_commands)}个命令)"
        elif auto_filter_enabled and not HAS_COMMAND_FILTER:
            cmd_filter_status = "⚠️ 已启用但模块不可用"
        else:
            cmd_filter_status = "❌ 已禁用"
        
        # 获取LLM判断开关状态
        enable_llm_judge = self.config.get('enable_llm_judge', True)
        judge_provider_id = self.config.get('judge_provider_id', '')
        if enable_llm_judge:
            if judge_provider_id:
                llm_judge_status = f"✅ 已启用 (提供商: {judge_provider_id})"
            else:
                llm_judge_status = "⚠️ 已启用但未配置提供商"
        else:
            llm_judge_status = "❌ 已禁用 (匹配即提醒)"

        forward_limits = self._get_forward_limits()
        stored_videos = [
            video
            for message in messages
            for video in (
                message.get("videos")
                if isinstance(message.get("videos"), list)
                else []
            )
            if isinstance(video, dict)
        ]
        video_limit_mb = self._get_video_max_size_bytes() // (1024 * 1024)
        ffmpeg_status = (
            "✅ 可用"
            if self._ffmpeg_path and self._ffprobe_path
            else "⚠️ 不可用（降级为SHA/元数据）"
        )
        status = f"""✅ Memory Reboot - 记忆状态

📌 群号: {group_id}
📊 消息数: {len(messages)}
🧠 含embedding: {sum(1 for m in messages if m.get("embedding"))}
🖼️ 含图片: {sum(1 for m in messages if m.get("has_image"))} (含哈希: {sum(1 for m in messages if m.get("image_hash"))})
🎞️ 含视频: {sum(1 for m in messages if m.get("has_video"))} (文件SHA: {sum(1 for v in stored_videos if v.get("sha256"))} / 五帧指纹: {sum(1 for v in stored_videos if v.get("frame_hashes"))} / 元数据兜底: {sum(1 for v in stored_videos if v.get("analysis") == "metadata")})
📨 合并转发: {sum(1 for m in messages if m.get("forward_id"))} (完整哈希: {sum(1 for m in messages if m.get("forward_hash"))} / 发送者+文本: {sum(1 for m in messages if m.get("forward_text_hash"))} / 部分可见: {sum(1 for m in messages if m.get("forward_partial"))})
🔗 纯URL: {sum(1 for m in messages if m.get("url_only"))} (含哈希: {sum(1 for m in messages if m.get("url_hash"))})

⚙️ 配置参数:
📏 文本相似度阈值: {self.config.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)}
🔍 图片哈希阈值: {self.config.get('image_hash_threshold', DEFAULT_IMAGE_HASH_THRESHOLD)}
👥 最少不同用户: {self.config.get('min_unique_senders', DEFAULT_MIN_UNIQUE_SENDERS)}人
⏰ 冷却时间: {self.config.get('cooldown_seconds', DEFAULT_COOLDOWN_SECONDS)}秒
📅 数据保留: {self.config.get('data_retention_days', DEFAULT_DATA_RETENTION_DAYS)}天
🧵 转发展开: 深度{forward_limits['depth']} / 拉取{forward_limits['fetches']}次 / 单次超时{forward_limits['timeout']}秒 / 节点{forward_limits['nodes']}条
🧱 转发内容: 消息段{forward_limits['segments']}个 / 字符{forward_limits['chars']}个
👁️ 部分可见哈希: 至少{forward_limits['visible_chars']}个正文字符，或包含稳定媒体标识
📝 发送者+文本匹配: {'✅ 已开启（至少20字且2个文本节点）' if self._is_forward_sender_text_match_enabled() else '❌ 已关闭'}
🧪 转发指纹诊断: {'⚠️ 已开启（日志含聊天内容）' if self._is_forward_debug_enabled() else '❌ 已关闭'}
🔗 纯URL精确匹配: {'✅ 已开启' if self._get_bool_config('url_only_exact_match', True) else '❌ 已关闭'} (跟踪参数规则: {len(self._get_url_tracking_rules())}条)
🎬 视频判重: {'✅ 已开启' if self._get_bool_config('video_duplicate_enabled', True) else '❌ 已关闭'} (分析上限: {video_limit_mb} MiB / 五帧均值阈值: {self._get_video_frame_threshold()})
🧪 视频指纹诊断: {'⚠️ 已开启（稳定后请关闭）' if self._is_video_debug_enabled() else '❌ 已关闭'}

🛠️ 环境检查:
- Pillow库: {'✅ 已安装 (dHash可用)' if HAS_PIL else '❌ 未安装 (降级为MD5)'}
- FFmpeg/ffprobe: {ffmpeg_status}
- 提醒图片: {'✅ 存在' if os.path.exists(img_path) else '⚠️ 不存在 (将发送纯文本)'}
- 命令过滤: {cmd_filter_status}
- LLM判断: {llm_judge_status}"""
        yield event.plain_result(status)
    
    @filter.command("查看过滤命令")
    async def show_filtered_commands(self, event: AstrMessageEvent):
        """查看当前检测到的所有插件命令（仅管理员）"""
        if not event.is_admin():
            yield event.plain_result("❌ 仅管理员可执行")
            return
        if not HAS_COMMAND_FILTER:
            yield event.plain_result("❌ 命令过滤模块不可用，请检查AstrBot版本")
            return
        
        commands = self._get_all_plugin_commands()
        
        if not commands:
            yield event.plain_result("📋 当前未检测到任何插件命令")
            return
        
        # 将命令分组显示，每行最多5个
        cmd_lines = []
        for i in range(0, len(commands), 5):
            cmd_lines.append("  ".join(commands[i:i+5]))
        
        auto_filter_enabled = self.config.get('auto_filter_commands', True)
        status = "✅ 已启用" if auto_filter_enabled else "❌ 已禁用"
        
        result = f"""📋 插件命令过滤列表

🔧 自动过滤状态: {status}
📊 检测到 {len(commands)} 个命令:

{chr(10).join(cmd_lines)}

💡 这些命令会被自动忽略，不会被记录或触发旧闻提醒。
可在配置中修改 "auto_filter_commands" 来启用/禁用此功能。"""
        
        yield event.plain_result(result)
    
    @filter.command("擦除记忆")
    async def clear_data(self, event: AstrMessageEvent):
        """清除群组数据（仅管理员，结果仅输出到日志）"""
        group_id = event.get_group_id()
        if not group_id:
            logger.info("[Memory Reboot] 擦除记忆命令失败: 非群聊环境")
            return
        if not event.is_admin():
            logger.info(f"[Memory Reboot] 擦除记忆命令被拒绝: 用户 {event.get_sender_id()} 非管理员")
            return
        
        group_id = str(group_id)
        
        # 清除内存缓存
        if group_id in self._cache:
            del self._cache[group_id]
        
        # 删除整个群组数据目录
        group_dir = self._get_group_dir(group_id)
        if os.path.exists(group_dir):
            try:
                shutil.rmtree(group_dir)
                logger.info(f"[Memory Reboot] 记忆已擦除 - 群{group_id}的所有历史数据已清空（含内存缓存）")
            except Exception as e:
                logger.error(f"[Memory Reboot] 清空数据失败: {e}")
        else:
            logger.info(f"[Memory Reboot] 记忆擦除完成 - 群{group_id}暂无数据")
