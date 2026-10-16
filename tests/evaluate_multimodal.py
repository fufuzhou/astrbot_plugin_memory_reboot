"""Evaluate labelled pairs from JSON, optionally encoding with the live HTTP backend.

Input: {"messages":[{"id":"a","text":"...", "images":["path-or-url"]}, ...],
        "pairs":[{"a":"a","b":"b","duplicate":true}], "config": {...}}
Use --online to encode inputs. Offline messages must include existing multimodal features.
No credentials or message contents are included in the result.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from memory_embedding.manager import EmbeddingManager
from memory_embedding.matching import compare


async def evaluate(dataset, online=False, root=None):
    config = dict(dataset.get("config", {}))
    records = {str(m["id"]): dict(m) for m in dataset["messages"]}
    latencies = []
    if online:
        config.update(
            embedding_api_url=os.environ["EMBEDDING_API_BASE"],
            embedding_api_key=os.environ.get("EMBEDDING_API_KEY", ""),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "embeddinggemma-2-440m"),
        )
        if root:
            config["multimodal_local_image_roots"] = [str(root)]
        manager = EmbeddingManager(config)
        try:
            for record in records.values():
                sources = record.get("images", [])
                if root:
                    sources = [
                        str(root / s)
                        if "://" not in s and not s.startswith("data:")
                        else s
                        for s in sources
                    ]
                start = time.perf_counter()
                record["multimodal"] = await manager.features(
                    record.get("text", ""), sources
                )
                latencies.append((time.perf_counter() - start) * 1000)
        finally:
            await manager.close()
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    decisions = []
    for pair in dataset["pairs"]:
        if not isinstance(pair["duplicate"], bool):
            raise ValueError("duplicate labels must be boolean")
        match = compare(records[str(pair["a"])], records[str(pair["b"])], config)
        predicted, actual = match is not None, pair["duplicate"]
        category = {
            (True, True): "tp",
            (True, False): "fp",
            (False, True): "fn",
            (False, False): "tn",
        }[predicted, actual]
        counts[category] += 1
        decisions.append(
            {
                "a": pair["a"],
                "b": pair["b"],
                "expected": actual,
                "predicted": predicted,
                "score": match["score"] if match else None,
            }
        )

    def ratio(a, b):
        return a / b if b else None

    return {
        **counts,
        "precision": ratio(counts["tp"], counts["tp"] + counts["fp"]),
        "recall": ratio(counts["tp"], counts["tp"] + counts["fn"]),
        "false_positive_rate": ratio(counts["fp"], counts["fp"] + counts["tn"]),
        "median_encoding_ms": statistics.median(latencies) if latencies else None,
        "pairs": decisions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = asyncio.run(evaluate(data, args.online, args.dataset.resolve().parent))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
