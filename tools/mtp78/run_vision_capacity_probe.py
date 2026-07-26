#!/usr/bin/env python3
"""Exercise a 200k+ token GLM-5.2 Vision request and persist raw evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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


def build_varied_filler(lines: int) -> str:
    topics = (
        "amber circuit",
        "blue orchard",
        "cedar bridge",
        "delta harbor",
        "emerald engine",
        "frozen galaxy",
        "granite library",
        "hidden meadow",
        "indigo turbine",
        "jade archive",
        "kinetic valley",
        "lunar workshop",
        "mosaic river",
        "northern compass",
        "opal reactor",
        "plasma garden",
    )
    return "".join(
        f"Record {index:06d} describes the {topics[index % len(topics)]}; "
        f"checksum {(index * 7919) % 1_000_003:06d}.\\n"
        for index in range(lines)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--model", default="GLM-5.2-Vision")
    parser.add_argument("--lines", type=int, default=12_220)
    parser.add_argument("--filler-file", type=Path)
    parser.add_argument("--image-position", choices=("head", "tail"), default="tail")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--plain-output", action="store_true")
    parser.add_argument("--minimum-prompt-tokens", type=int, default=200_000)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "vision-canary-images" / "A-alpha17-red-triangle.png",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "vision-capacity-runs",
    )
    args = parser.parse_args()

    output_dir = args.output_root / args.label
    output_dir.mkdir(parents=True, exist_ok=False)
    image_raw = args.image.read_bytes()
    if args.filler_file is None:
        filler = build_varied_filler(args.lines)
        filler_metadata = {
            "filler_mode": "varied_numbered_records",
            "lines": args.lines,
        }
    else:
        filler = args.filler_file.read_text(encoding="utf-8")
        filler_metadata = {
            "filler_mode": "file",
            "filler_file": str(args.filler_file),
            "filler_sha256": hashlib.sha256(filler.encode("utf-8")).hexdigest(),
        }
    data_url = "data:image/png;base64," + base64.b64encode(image_raw).decode("ascii")

    tokenize_status, tokenize_raw = request_bytes(
        f"{args.base_url}/tokenize",
        {"model": args.model, "prompt": filler},
        args.timeout,
    )
    (output_dir / "tokenize.raw.json").write_bytes(tokenize_raw)
    tokenize_response = json.loads(tokenize_raw)

    image_item = {"type": "image_url", "image_url": {"url": data_url}}
    query_text = (
        "After reading the image immediately above, reply with only its large first-line title."
        if args.plain_output
        else 'After reading the image immediately above, return JSON only: {"title":"large first-line title"}.'
    )
    query_item = {"type": "text", "text": query_text}
    filler_item = {"type": "text", "text": filler}
    if args.image_position == "head":
        content = [image_item, {"type": "text", "text": filler + "\\n" + query_item["text"]}]
    else:
        content = [filler_item, image_item, query_item]


    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    if not args.plain_output:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "vision_capacity_answer",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    atomic_json(
        output_dir / "request.manifest.json",
        {
            "model": args.model,
            **filler_metadata,
            "image_position": args.image_position,
            "output_mode": "plain" if args.plain_output else "strict_json_schema",
            "filler_tokens": tokenize_response.get("count"),
            "tokenize_status": tokenize_status,
            "image": str(args.image),
            "image_sha256": hashlib.sha256(image_raw).hexdigest(),
            "image_bytes": len(image_raw),
            "max_tokens": payload["max_tokens"],
        },
    )

    before_status, before_metrics = request_bytes(f"{args.base_url}/metrics", None, args.timeout)
    (output_dir / "metrics.before.prom").write_bytes(before_metrics)
    started = time.monotonic()
    status, raw = request_bytes(f"{args.base_url}/v1/chat/completions", payload, args.timeout)
    elapsed = time.monotonic() - started
    raw_path = output_dir / "response.raw.json"
    raw_path.write_bytes(raw)  # Persist untouched bytes before decoding or grading.
    after_status, after_metrics = request_bytes(f"{args.base_url}/metrics", None, args.timeout)
    (output_dir / "metrics.after.prom").write_bytes(after_metrics)

    response = json.loads(raw)
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    if args.plain_output:
        parsed_content = None
        answer_matches = content.strip().upper() == "ALPHA 17"
    else:
        try:
            parsed_content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed_content = None
        answer_matches = (
            isinstance(parsed_content, dict)
            and str(parsed_content.get("title", "")).strip().upper() == "ALPHA 17"
        )
    passed = (
        status == 200
        and isinstance(prompt_tokens, int)
        and prompt_tokens >= args.minimum_prompt_tokens
        and answer_matches
        and choice.get("finish_reason") == "stop"
    )
    summary = {
        "schema": "glm52-vision-capacity-probe-v2",
        "label": args.label,
        "http_status": status,
        "elapsed_s": elapsed,
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "assistant_content": content,
        "assistant_reasoning": message.get("reasoning"),
        "parsed_content": parsed_content,
        "output_mode": "plain" if args.plain_output else "strict_json_schema",
        "minimum_prompt_tokens": args.minimum_prompt_tokens,
        "filler_token_count": tokenize_response.get("count"),
        "metrics_status": {"before": before_status, "after": after_status},
        "raw_response": str(raw_path),
        "pass": passed,
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
