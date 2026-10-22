"""Program evidence for the text-only contextual judge."""

from .matching import compare, threshold


def exact_repeat(current, stored):
    # Semantic matches are not counted as exact repetition.
    match = compare(current, stored, {})
    return bool(match and match["kind"] == "exact")


def match_evidence(current, stored, match, config):
    a, b = current["multimodal"], stored["multimodal"]
    left, right = len(a.get("images", [])), len(b.get("images", []))
    kind = "精确匹配（规范化正文及全部图片的文件/像素指纹一致）"
    if match["kind"] != "exact":
        kind = "语义匹配（通过完整消息规则，不代表图片字节或事实完全一致）"
    if left and right:
        coverage = (
            f"当前{left}张、历史{right}张；全部图片已通过一一匹配，非仅一张公共配图命中"
        )
    elif left or right:
        coverage = (
            f"当前{left}张、历史{right}张；文字与单图跨模态匹配，未进行两图逐图比较"
        )
    else:
        coverage = "双方均为纯文本"
    current_description = str(current.get("image_description") or "").strip()
    stored_description = str(stored.get("image_description") or "").strip()
    description_block = ""
    if current_description or stored_description:
        description_block = (
            "\n【图片描述】\n"
            f"- 当前图片描述：{current_description[:500] or '无'}\n"
            f"- 历史图片描述：{stored_description[:500] or '无'}\n"
        )
    return (
        "【程序提供的匹配证据】\n"
        f"- 匹配类型：{kind}\n"
        f"- 完整消息匹配分数：{match['score']:.6f}（不是事实相同的概率）\n"
        f"- 图片覆盖：{coverage}\n"
        f"{description_block}"
        f"- 正文：当前{'有' if a.get('text') else '无'}，历史{'有' if b.get('text') else '无'}；已按完整消息规则检查\n"
        f"- 语义阈值：文本{threshold(config, 'multimodal_text_threshold', 0.90):.4f}、"
        f"图片{threshold(config, 'multimodal_image_threshold', 0.94):.4f}、"
        f"跨模态{threshold(config, 'multimodal_cross_threshold', 0.94):.4f}（精确匹配不依赖语义阈值）\n"
        "- [图片×N]只表示图片数量；相同占位符不代表同一图片。\n"
        "- 如果双方附有图片描述，必须比较描述中的具体对象、卡牌/人物/标题、"
        "文字和数字；描述指向不同内容时，即使向量分数高，也不得判为重复。\n"
        "- 近期精确复读统计按完整内容/指纹计算，不包含仅语义近似的消息；"
        "它与前置语义匹配的发送者人数口径不同。\n"
        "- 本请求没有提供原图；如附有图片描述，只能把它当作辅助文字，"
        "不可声称看到了原图细节。"
        "请依据上述匹配证据和聊天上下文判断是否适合提醒，"
        "不要仅因没有原图或图片描述就否定程序已检测到的匹配。"
    )
