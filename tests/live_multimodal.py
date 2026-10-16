"""Opt-in live smoke/evaluation. Reads credentials from environment only."""
import argparse
import asyncio
import io
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from memory_embedding.manager import EmbeddingManager
from memory_embedding.matching import cosine, search_images


def fixture(headline, detail, color="navy"):
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    try:
        title = ImageFont.truetype("arial.ttf", 32)
        body = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        title = body = ImageFont.load_default(size=24)
    draw.rectangle((0, 0, 900, 90), fill=color)
    draw.text((30, 25), "CITY NEWS", fill="white", font=title)
    draw.text((30, 140), headline, fill="black", font=title)
    draw.text((30, 230), detail, fill="black", font=body)
    return image


async def run():
    config = {
        "embedding_api_url": os.environ["EMBEDDING_API_BASE"],
        "embedding_api_key": os.environ.get("EMBEDDING_API_KEY", ""),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "embeddinggemma-2-440m"),
        "embedding_dimensions": 768,
    }
    manager = EmbeddingManager(config)
    timings = {}

    async def encode(image, name):
        output = io.BytesIO()
        image.save(output, format="PNG")
        import base64
        uri = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        start = time.perf_counter()
        result = await manager.features("", [uri])
        timings[name] = round((time.perf_counter()-start)*1000, 2)
        assert result["complete"] and result["images"][0]["vector"], manager.last_error
        return result

    try:
        capabilities = await manager.provider.test_connection()
        original = fixture("Council approves a new public park", "Budget: 10 million dollars. Date: 2026-10-01.")
        changed = fixture("Council approves a new public park", "Budget: 90 million dollars. Date: 2026-11-01.")
        unrelated = fixture("Database server upgrade completed", "New storage cluster supports encrypted backups.", "green")
        a = await encode(original, "original")
        output = io.BytesIO()
        original.resize((600,400)).save(output, format="JPEG", quality=50)
        resized = Image.open(io.BytesIO(output.getvalue()))
        b = await encode(resized, "resized_jpeg")
        c = await encode(changed, "changed_numbers")
        d = await encode(unrelated, "unrelated")
        image_vector = a["images"][0]["vector"]
        query = await manager.provider.embed_text("a news report about a new public park", task="search")
        dup_text = await manager.provider.embed_text("Council approves a new public park", task="similarity")
        caption = await manager.features("Council approves a new public park", [
            (await manager.images.load("data:image/png;base64," + __import__("base64").b64encode(output.getvalue()).decode()))["data_uri"]
        ])
        assert caption["fusion_vector"]
        records = [{"timestamp":time.time(),"message_id":str(i),"multimodal":f} for i,f in enumerate([a,d])]
        result = {
            "capabilities":capabilities,
            "timings_ms":timings,
            "scores":{
                "resize_recompress":cosine(image_vector,b["images"][0]["vector"]),
                "changed_numbers_same_layout":cosine(image_vector,c["images"][0]["vector"]),
                "unrelated_same_news_layout":cosine(image_vector,d["images"][0]["vector"]),
                "search_query_to_image":cosine(query,image_vector),
                "similarity_text_to_image":cosine(dup_text,image_vector),
            },
            "search_rank":[m["message_id"] for m in search_images(records,query,manager.provider.space,{})],
            "joint_vector_dimension":len(caption["fusion_vector"]),
            "limits":"Synthetic smoke fixtures only; not a calibrated benchmark or real AstrBot acceptance.",
        }
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return result
    finally:
        await manager.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    args=parser.parse_args()
    result=asyncio.run(run())
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
