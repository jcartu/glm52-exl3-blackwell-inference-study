#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx


def load_prompts(path: Path, limit_chars: int) -> list[str]:
    prompts = []
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            text = str(row["text"])
            if len(text) > limit_chars:
                text = text[:limit_chars]
            prompts.append(text)
    return prompts


async def main() -> None:
    parser = argparse.ArgumentParser(description="Drive owner-corpus MTP activation capture")
    parser.add_argument("--url", default="http://127.0.0.1:5001/v1/chat/completions")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--capture-layer-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit-chars", type=int, default=12000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an incomplete, already-enabled capture",
    )
    args = parser.parse_args()

    if args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError("--concurrency and --max-tokens must be positive")
    prompts = load_prompts(args.corpus, args.limit_chars)
    enable = args.capture_layer_dir / "ENABLE"
    complete = args.capture_layer_dir / "COMPLETE"
    args.capture_layer_dir.mkdir(parents=True, exist_ok=True)
    stale = [
        path.name
        for path in (
            enable,
            complete,
            args.capture_layer_dir / "x.bin",
            args.capture_layer_dir / "ids.bin",
            args.capture_layer_dir / "layer_manifest.json",
        )
        if path.exists()
    ]
    if args.resume:
        required = {"ENABLE", "x.bin", "ids.bin"}
        missing = required - set(stale)
        if missing or complete.exists():
            raise RuntimeError(
                f"resume requires ENABLE/x.bin/ids.bin and no COMPLETE: "
                f"missing={sorted(missing)}, complete={complete.exists()}"
            )
    elif stale:
        raise RuntimeError(f"capture directory is not clean: {sorted(stale)}")
    else:
        enable.write_text("enabled\n")

    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for item in enumerate(prompts):
        queue.put_nowait(item)
    totals = {"requests": 0, "completion_tokens": 0, "failures": 0}
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        async def worker() -> None:
            while not complete.exists():
                try:
                    index, prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": args.max_tokens,
                    "seed": 20260724 + index,
                    "ignore_eos": True,
                }
                try:
                    response = await client.post(args.url, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    completion_tokens = int(body.get("usage", {}).get("completion_tokens", 0))
                    async with lock:
                        totals["requests"] += 1
                        totals["completion_tokens"] += completion_tokens
                        if totals["requests"] % 16 == 0:
                            print(json.dumps(totals, sort_keys=True), flush=True)
                except Exception as exc:
                    async with lock:
                        totals["failures"] += 1
                        print(f"request {index} failed: {exc}", flush=True)
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))

    if not complete.exists():
        raise RuntimeError(
            f"corpus exhausted before capture completed: {json.dumps(totals, sort_keys=True)}"
        )
    print(json.dumps({**totals, "complete": complete.read_text().strip()}, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
