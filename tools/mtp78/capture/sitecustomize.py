from __future__ import annotations

import os
from pathlib import Path
import threading

_CAPTURE_DIR = os.environ.get("VLLM_MTP_CAPTURE_DIR")

if _CAPTURE_DIR:
    import torch

    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    _root = Path(_CAPTURE_DIR)
    _root.mkdir(parents=True, exist_ok=True)
    _enable = _root / "ENABLE"
    _x_path = _root / "x.bin"
    _ids_path = _root / "ids.bin"
    _hidden = 6144
    _topk = 8
    _limit = int(os.environ.get("VLLM_MTP_CAPTURE_MAX_TOKENS", "131072"))
    if _limit <= 0:
        raise ValueError("VLLM_MTP_CAPTURE_MAX_TOKENS must be positive")
    _lock = threading.Lock()
    _captured = min(
        _x_path.stat().st_size // (_hidden * 2) if _x_path.exists() else 0,
        _ids_path.stat().st_size // _topk if _ids_path.exists() else 0,
    )
    _original_apply_quant_method = MoERunner._apply_quant_method

    def _capture_mtp_rows(runner, hidden_states, router_logits, input_ids):
        global _captured
        if not _enable.exists() or _captured >= _limit:
            return
        if "layers.78." not in runner.layer_name:
            return
        if runner.moe_config.moe_parallel_config.tp_rank != 0:
            return
        if runner.routed_experts.quant_method.is_monolithic:
            raise RuntimeError("MTP capture requires a modular MoE backend")

        _, topk_ids = runner.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=torch.int32,
            input_ids=input_ids,
        )
        if (
            runner.moe_config.num_experts != 256
            or runner.moe_config.experts_per_token != _topk
            or topk_ids.ndim != 2
            or topk_ids.shape[0] != hidden_states.shape[0]
            or topk_ids.shape[1] < _topk
        ):
            raise RuntimeError(
                "unexpected MTP routing contract: "
                f"experts={runner.moe_config.num_experts}, "
                f"top_k={runner.moe_config.experts_per_token}, "
                f"hidden_shape={tuple(hidden_states.shape)}, "
                f"ids_shape={tuple(topk_ids.shape)}"
            )
        finite_rows = torch.isfinite(hidden_states[:, :_hidden]).all(dim=1)
        finite_rows &= torch.isfinite(router_logits).all(dim=1)
        if not finite_rows.any():
            return
        x = hidden_states[finite_rows, :_hidden].detach()
        ids = topk_ids[finite_rows, :_topk].detach()
        rows = min(int(x.shape[0]), _limit - _captured)
        if rows <= 0:
            return
        x = x[:rows].to(device="cpu", dtype=torch.bfloat16)
        ids = ids[:rows].to(device="cpu", dtype=torch.uint8)
        if tuple(x.shape) != (rows, _hidden) or tuple(ids.shape) != (rows, _topk):
            raise RuntimeError(
                f"unexpected MTP capture shapes: hidden={tuple(x.shape)}, ids={tuple(ids.shape)}"
            )
        with _lock:
            rows = min(rows, _limit - _captured)
            if rows <= 0:
                return
            with _x_path.open("ab", buffering=0) as x_file:
                x_file.write(x[:rows].view(torch.int16).numpy().tobytes())
            with _ids_path.open("ab", buffering=0) as ids_file:
                ids_file.write(ids[:rows].numpy().tobytes())
            _captured += rows
            if _captured >= _limit:
                (_root / "COMPLETE").write_text(f"{_captured}\n")

    def _capturing_apply_quant_method(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        _capture_mtp_rows(self, hidden_states, router_logits, input_ids)
        return _original_apply_quant_method(
            self,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )

    MoERunner._apply_quant_method = _capturing_apply_quant_method
