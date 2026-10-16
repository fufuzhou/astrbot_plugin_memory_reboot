"""Shared-image retrieval and whole-message duplicate rules, independent of AstrBot."""

import math
import re
import time

import numpy as np

from .provider import EmbeddingError, normalized


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def cosine(a, b):
    try:
        if len(a) != len(b):
            return 0.0
        return float(np.dot(normalized(a), normalized(b)))
    except (TypeError, ValueError, EmbeddingError):
        return 0.0


def threshold(config, key, default):
    try:
        value = float(config.get(key, default))
        return value if math.isfinite(value) and 0 <= value <= 1 else default
    except (TypeError, ValueError):
        return default


def numbers_conflict(a, b):
    left, right = (
        set(re.findall(r"\d+(?:\.\d+)?", a)),
        set(re.findall(r"\d+(?:\.\d+)?", b)),
    )
    return bool(left and right and left != right)


def image_assignment(left, right, score, minimum):
    """Find a complete one-to-one assignment, not repeated matches to one image."""
    if not left or len(left) != len(right):
        return None
    edges = [[score(a, b) for b in right] for a in left]
    best = None

    def visit(index, used, scores):
        nonlocal best
        if index == len(left):
            value = sum(scores) / len(scores)
            best = value if best is None else max(best, value)
            return
        for j, value in enumerate(edges[index]):
            if j not in used and value >= minimum:
                visit(index + 1, used | {j}, scores + [value])

    visit(0, set(), [])
    return best


def compare(current, stored, config):
    a, b = current.get("multimodal"), stored.get("multimodal")
    if not a or not b or a.get("schema") != 1 or b.get("schema") != 1:
        return None
    if not a.get("complete") or not b.get("complete"):
        return None
    at, bt = clean_text(a.get("text")), clean_text(b.get("text"))
    ai, bi = a.get("images", []), b.get("images", [])
    if numbers_conflict(at, bt):
        return None

    def exact_image(x, y):
        return float(
            bool(
                (x.get("sha256") and x["sha256"] == y.get("sha256"))
                or (
                    x.get("pixel_sha256") and x["pixel_sha256"] == y.get("pixel_sha256")
                )
            )
        )

    if at == bt and (at or ai):
        exact = (not ai and not bi) or image_assignment(
            ai, bi, exact_image, 1.0
        ) is not None
        if exact:
            return {"score": 1.0, "kind": "exact"}

    if a.get("space") != b.get("space") or not a.get("space"):
        return None
    text_min = threshold(config, "multimodal_text_threshold", 0.90)
    image_min = threshold(config, "multimodal_image_threshold", 0.94)
    cross_min = threshold(config, "multimodal_cross_threshold", 0.94)
    scores = []
    if ai and bi:
        image_score = image_assignment(
            ai, bi, lambda x, y: cosine(x.get("vector"), y.get("vector")), image_min
        )
        if image_score is None:
            return None
        scores.append(image_score)
        # Different/additional captions must be checked, not erased by an image hit.
        if bool(at) != bool(bt):
            return None
        if at and bt:
            value = cosine(a.get("text_vector"), b.get("text_vector"))
            if at != bt and value < text_min:
                return None
            scores.append(1.0 if at == bt else value)
    elif not ai and not bi:
        if not at or not bt:
            return None
        value = cosine(a.get("text_vector"), b.get("text_vector"))
        if value < text_min:
            return None
        scores.append(value)
    else:
        image_features, text_features = (a, b) if ai else (b, a)
        if len(image_features.get("images", [])) != 1 or not text_features.get("text"):
            return None
        query = text_features.get("text_vector")
        value = cosine(query, image_features["images"][0].get("vector"))
        if value < cross_min:
            return None
        scores.append(value)
        if image_features.get("text"):
            caption_score = cosine(query, image_features.get("text_vector"))
            if caption_score < text_min:
                return None
            scores.append(caption_score)
    # Fusion may strengthen ranking, but cannot bypass image coverage or caption checks.
    if a.get("fusion_vector") and b.get("fusion_vector"):
        scores.append(cosine(a["fusion_vector"], b["fusion_vector"]))
    return {"score": sum(scores) / len(scores), "kind": "semantic"}


def candidates(messages, current, config):
    cutoff = time.time() - int(config.get("data_retention_days", 7)) * 86400
    matches = []
    for index, message in enumerate(messages):
        if message.get("timestamp", 0) < cutoff:
            continue
        match = compare(current, message, config)
        if match:
            matches.append({**match, "index": index, "message": message})
    return sorted(
        matches, key=lambda item: (-item["score"], item["message"].get("timestamp", 0))
    )


def search_images(messages, query_vector, space, config):
    cutoff = time.time() - int(config.get("data_retention_days", 7)) * 86400
    rows, owners = [], []
    for message in messages:
        features = message.get("multimodal") or {}
        if message.get("timestamp", 0) < cutoff or features.get("space") != space:
            continue
        for index, image in enumerate(features.get("images", [])):
            try:
                rows.append(normalized(image.get("vector"), len(query_vector)))
                owners.append((message, index))
            except EmbeddingError:
                continue
    if not rows:
        return []
    scores = np.asarray(rows) @ np.asarray(normalized(query_vector))
    minimum = threshold(config, "multimodal_search_threshold", 0.50)
    results = []
    for score, (message, index) in zip(scores, owners):
        if score >= minimum:
            results.append(
                {
                    **message,
                    "_image_index": index,
                    "_image_search_match_type": "multimodal",
                    "_image_search_semantic_similarity": float(score),
                }
            )
    return sorted(
        results,
        key=lambda m: (-m["_image_search_semantic_similarity"], -m.get("timestamp", 0)),
    )
