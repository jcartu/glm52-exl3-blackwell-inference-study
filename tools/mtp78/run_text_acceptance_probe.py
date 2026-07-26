#!/usr/bin/env python3
"""Measure MTP acceptance on deterministic text-only controls."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    """Write a self-contained technical explanation of exactly 420 words about how a compiler lowers a counted loop into machine code. Cover parsing, intermediate representation, loop optimizations, register allocation, instruction selection, and final code emission. Use paragraphs, no bullets, and do not mention this instruction.""",
    """Write a self-contained technical explanation of exactly 420 words about how a database executes a selective indexed query. Cover parsing, planning, cardinality estimates, index traversal, row visibility, joins, and result materialization. Use paragraphs, no bullets, and do not mention this instruction.""",
]

COUNTERS = (
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)


def request_bytes(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_counters(raw: bytes) -> dict[str, float]:
    text = raw.decode("utf-8", "replace")
    values: dict[str, float] = {}
    for counter in COUNTERS:
        pattern = re.compile(
            rf'^vllm:{re.escape(counter)}\{{[^\n]*\}}\s+([0-9.eE+-]+)$',
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match:
            values[counter] = float(match.group(1))
    for position, value in re.findall(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^\n]*position="(\d+)"[^\n]*\}\s+([0-9.eE+-]+)$',
        text,
        re.MULTILINE,
    ):
        values[f"spec_decode_num_accepted_tokens_per_pos_total:{position}"] = float(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--model", default="GLM-5.2-Vision")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "text-acceptance-runs",
    )
    args = parser.parse_args()

    output_dir = args.output_root / args.label
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    before_status, before_raw = request_bytes(f"{args.base_url}/metrics", None, args.timeout)
    (output_dir / "metrics.before.prom").write_bytes(before_raw)
    before = metric_counters(before_raw)
    results = []

    for index, prompt in enumerate(PROMPTS, start=1):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
        }
        atomic_json(output_dir / f"request-{index}.json", payload)
        request_started = time.monotonic()
        status, raw = request_bytes(f"{args.base_url}/v1/chat/completions", payload, args.timeout)
        elapsed = time.monotonic() - request_started
        raw_path = output_dir / f"response-{index}.raw.json"
        raw_path.write_bytes(raw)  # Persist untouched server bytes before decoding.
        response = json.loads(raw)
        choice = response.get("choices", [{}])[0]
        results.append(
            {
                "index": index,
                "http_status": status,
                "elapsed_s": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "raw_response": str(raw_path),
            }
        )
        print(f"text_control_{index}: HTTP {status} ({elapsed:.2f}s)", flush=True)

    after_status, after_raw = request_bytes(f"{args.base_url}/metrics", None, args.timeout)
    (output_dir / "metrics.after.prom").write_bytes(after_raw)
    after = metric_counters(after_raw)
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after.keys() | before.keys()}
    draft_tokens = delta.get("spec_decode_num_draft_tokens_total", 0.0)
    drafts = delta.get("spec_decode_num_drafts_total", 0.0)
    accepted = delta.get("spec_decode_num_accepted_tokens_total", 0.0)
    summary = {
        "schema": "glm52-text-acceptance-probe-v1",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "metrics_status": {"before": before_status, "after": after_status},
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": delta,
        "acceptance": {
            "accepted_per_draft_token": accepted / draft_tokens if draft_tokens else None,
            "accepted_tokens_per_draft": accepted / drafts if drafts else None,
            "draft_tokens_per_draft": draft_tokens / drafts if drafts else None,
        },
        "results": results,
        "elapsed_s": time.time() - started,
        "pass": all(result["http_status"] == 200 for result in results) and draft_tokens > 0,
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps({"label": args.label, "pass": summary["pass"], "acceptance": summary["acceptance"]}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
