#!/usr/bin/env python3
"""Run deterministic GLM-5.2 Vision ordering and interleaving canaries.

Every HTTP response body is persisted before JSON decoding or classification.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Card:
    key: str
    filename: str
    title: str
    background: str
    shape: str
    count: int

    def expected(self, slot: str) -> dict[str, Any]:
        return {
            "slot": slot,
            "title": self.title,
            "background": self.background,
            "shape": self.shape,
            "count": self.count,
        }


CARDS = {
    "A": Card("A", "A-alpha17-red-triangle.png", "ALPHA 17", "red", "triangle", 1),
    "B": Card("B", "B-beta42-blue-circles.png", "BETA 42", "blue", "circle", 2),
    "C": Card("C", "C-gamma73-green-squares.png", "GAMMA 73", "green", "square", 3),
    "D": Card("D", "D-delta98-yellow-stars.png", "DELTA 98", "yellow", "star", 4),
}

DETAIL_INSTRUCTION = """Output one compact JSON object only: {\"slots\":[{\"slot\":\"image_1\",\"title\":\"...\",\"background\":\"...\",\"shape\":\"...\",\"count\":0}]}.
For title, copy only the large first line and omit the subtitle. Background must be red, blue, green, or yellow. Shape must be triangle, circle, square, or star. Count the large shapes."""


@dataclass(frozen=True)
class Case:
    name: str
    content: list[tuple[str, str]]
    expected: dict[str, Any]


def image_item(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def text_item(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def build_cases() -> list[Case]:
    a, b, c, d = (CARDS[key] for key in "ABCD")
    return [
        Case(
            "single_a",
            [
                ("image", "A"),
                ("text", f"{DETAIL_INSTRUCTION}\nDescribe this image as image_1."),
            ],
            {"slots": [a.expected("image_1")]},
        ),
        Case(
            "two_ab_order",
            [
                ("text", "Read the following two images in encounter order."),
                ("image", "A"),
                ("image", "B"),
                (
                    "text",
                    'Return compact JSON only: {\"order\":[\"first large title\",\"second large title\"]}. Copy only each large first-line title, not its subtitle.',
                ),
            ],
            {"order": [a.title, b.title]},
        ),
        Case(
            "two_ba_order",
            [
                ("text", "Read the following two images in encounter order."),
                ("image", "B"),
                ("image", "A"),
                (
                    "text",
                    'Return compact JSON only: {\"order\":[\"first large title\",\"second large title\"]}. Copy only each large first-line title, not its subtitle.',
                ),
            ],
            {"order": [b.title, a.title]},
        ),
        Case(
            "four_abcd_order",
            [
                ("text", "Read the following four images in encounter order."),
                ("image", "A"),
                ("image", "B"),
                ("image", "C"),
                ("image", "D"),
                (
                    "text",
                    'Return compact JSON only: {\"order\":[\"title 1\",\"title 2\",\"title 3\",\"title 4\"]}. Copy only each large first-line title in encounter order, not its subtitle.',
                ),
            ],
            {"order": [a.title, b.title, c.title, d.title]},
        ),
        Case(
            "interleaved_ab",
            [
                ("image", "A"),
                (
                    "text",
                    "Bind the image immediately before this sentence to slot_a. Distractor: a violet lighthouse has nine windows; this sentence describes neither image.",
                ),
                ("image", "B"),
                (
                    "text",
                    'Bind the image immediately before this sentence to slot_b. Return compact JSON only: {\"slot_a\":\"large first-line title\",\"slot_b\":\"large first-line title\"}. Copy only each bound image title, not its subtitle.',
                ),
            ],
            {"slot_a": a.title, "slot_b": b.title},
        ),
        Case(
            "interleaved_ba",
            [
                ("image", "B"),
                (
                    "text",
                    "Bind the image immediately before this sentence to slot_a. Distractor: a crimson submarine carries seven hexagons; this sentence describes neither image.",
                ),
                ("image", "A"),
                (
                    "text",
                    'Bind the image immediately before this sentence to slot_b. Return compact JSON only: {\"slot_a\":\"large first-line title\",\"slot_b\":\"large first-line title\"}. Copy only each bound image title, not its subtitle.',
                ),
            ],
            {"slot_a": b.title, "slot_b": a.title},
        ),
    ]


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
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def extract_assistant_content(response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    return str(content)


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("assistant response is not a JSON object")
    return value


def normalized_scalar(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip().lower()
        return int(stripped) if stripped.isdigit() else stripped
    return value


def compare_expected(actual: Any, expected: Any, path: str = "$") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        mismatches: list[str] = []
        for key, expected_value in expected.items():
            if key not in actual:
                mismatches.append(f"{path}.{key}: missing")
            else:
                mismatches.extend(compare_expected(actual[key], expected_value, f"{path}.{key}"))
        return mismatches
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}"]
        mismatches = []
        if len(actual) != len(expected):
            mismatches.append(f"{path}: expected {len(expected)} items, got {len(actual)}")
        for index, expected_value in enumerate(expected[: len(actual)]):
            mismatches.extend(compare_expected(actual[index], expected_value, f"{path}[{index}]"))
        return mismatches
    actual_value = normalized_scalar(actual)
    expected_value = normalized_scalar(expected)
    if actual_value != expected_value:
        return [f"{path}: expected {expected_value!r}, got {actual_value!r}"]
    return []


def json_schema_from_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {
                key: json_schema_from_shape(item) for key, item in value.items()
            },
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item_schema = json_schema_from_shape(value[0]) if value else {}
        return {
            "type": "array",
            "items": item_schema,
            "minItems": len(value),
            "maxItems": len(value),
        }
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "string"}


def materialize_content(case: Case, image_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for item_type, value in case.content:
        if item_type == "text":
            content.append(text_item(value))
            manifest.append({"type": "text", "text": value})
            continue
        card = CARDS[value]
        path = image_dir / card.filename
        raw = path.read_bytes()
        content.append(image_item(path))
        manifest.append(
            {
                "type": "image",
                "card": card.key,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    return content, manifest


def capture_metrics(base_url: str, path: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        status, body = request_bytes(f"{base_url}/metrics", None, timeout)
        path.write_bytes(body)
        return {"status": status, "bytes": len(body), "elapsed_s": time.monotonic() - started}
    except Exception as error:  # keep canaries running when metrics are unavailable
        return {"error": f"{type(error).__name__}: {error}", "elapsed_s": time.monotonic() - started}


def run_case(
    case: Case,
    base_url: str,
    model: str,
    image_dir: Path,
    output_dir: Path,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    content, manifest = materialize_content(case, image_dir)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"canary_{case.name}",
                "schema": json_schema_from_shape(case.expected),
                "strict": True,
            },
        },
    }
    atomic_json(
        output_dir / f"{case.name}.request.json",
        {"model": model, "content_manifest": manifest, "expected": case.expected, "max_tokens": max_tokens, "response_format": payload["response_format"]},
    )
    raw_path = output_dir / f"{case.name}.response.raw.json"
    started = time.monotonic()
    try:
        status, raw = request_bytes(f"{base_url}/v1/chat/completions", payload, timeout)
        elapsed = time.monotonic() - started
        raw_path.write_bytes(raw)  # Required ordering: persist untouched bytes before decoding.
        response = json.loads(raw)
        assistant = extract_assistant_content(response)
        (output_dir / f"{case.name}.assistant.txt").write_text(assistant, encoding="utf-8")
        parsed = parse_json_object(assistant)
        mismatches = compare_expected(parsed, case.expected)
        return {
            "case": case.name,
            "http_status": status,
            "elapsed_s": elapsed,
            "usage": response.get("usage"),
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            "assistant": assistant,
            "parsed": parsed,
            "expected": case.expected,
            "pass": status == 200 and not mismatches,
            "mismatches": mismatches,
            "raw_response": str(raw_path),
        }
    except Exception as error:
        return {
            "case": case.name,
            "elapsed_s": time.monotonic() - started,
            "expected": case.expected,
            "pass": False,
            "error": f"{type(error).__name__}: {error}",
            "raw_response": str(raw_path) if raw_path.exists() else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Run label, such as V0-bf16-mtp0")
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--model", default="GLM-5.2-Vision")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "vision-canary-images",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "vision-canary-runs",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case names, or all",
    )
    args = parser.parse_args()

    cases = build_cases()
    if args.cases != "all":
        requested = {name.strip() for name in args.cases.split(",") if name.strip()}
        known = {case.name for case in cases}
        unknown = requested - known
        if unknown:
            parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.name in requested]

    missing = [str(args.images_dir / card.filename) for card in CARDS.values() if not (args.images_dir / card.filename).is_file()]
    if missing:
        parser.error(f"missing image files: {', '.join(missing)}")

    output_dir = args.output_root / args.label
    output_dir.mkdir(parents=True, exist_ok=False)
    run_started = time.time()
    summary: dict[str, Any] = {
        "schema": "glm52-vision-canary-v2",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "started_unix": run_started,
        "output_dir": str(output_dir),
        "metrics_before": capture_metrics(args.base_url, output_dir / "metrics.before.prom", args.timeout),
        "results": [],
    }
    atomic_json(output_dir / "summary.json", summary)

    for case in cases:
        result = run_case(
            case,
            args.base_url,
            args.model,
            args.images_dir,
            output_dir,
            args.max_tokens,
            args.timeout,
        )
        summary["results"].append(result)
        atomic_json(output_dir / "summary.json", summary)
        print(f"{case.name}: {'PASS' if result['pass'] else 'FAIL'} ({result.get('elapsed_s', 0):.2f}s)", flush=True)

    summary["metrics_after"] = capture_metrics(args.base_url, output_dir / "metrics.after.prom", args.timeout)
    summary["elapsed_s"] = time.time() - run_started
    summary["pass"] = all(result["pass"] for result in summary["results"])
    summary["passed"] = sum(result["pass"] for result in summary["results"])
    summary["total"] = len(summary["results"])
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("label", "pass", "passed", "total", "elapsed_s")}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
