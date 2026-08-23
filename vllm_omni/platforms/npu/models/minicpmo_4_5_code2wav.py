# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inject MiniCPM-o Code2Wav NPUGraph acceleration on Ascend."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from contextlib import nullcontext
from weakref import WeakKeyDictionary

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)

_PATCHED = False
_original_build_backend = None
_original_estimator_step = None
_original_setup_batch = None
_original_decode_batch = None
_backend_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_ENABLE_KEY = "code2wav_enable_npu_graph"
_MAX_GRAPHS_KEY = "code2wav_max_npu_graphs"
# --- exp/static-freqs: CPU 小常量 H2D 缓存（timestep freqs） ---
_static_freqs_patched = False
_freqs_cache: dict[tuple[object, ...], torch.Tensor] = {}


def _patched_timestep_embedding(t, dim, max_period=10000):
    """freqs 只依赖 (dim, max_period)，缓存后消除每 chunk×step 的 CPU 生成 + H2D + 同步。

    数值一致性：CPU 上按原始公式/dtype 计算 → 一次性 .to(t)（保持原 dtype/device 跟随语义）
    → 按 (dim, max_period, device, dtype) 缓存。缓存键含 dtype/device，切换时自动重缓存。
    不用 register_buffer：浮点 buffer 会随 module.half()/bfloat16() 转换。
    """
    key = (dim, max_period, t.device, t.dtype)
    freqs = _freqs_cache.get(key)
    if freqs is None:
        half = dim // 2
        freqs_cpu = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        )
        freqs = freqs_cpu.to(t)  # 原语义：同时匹配 device 和 dtype
        _freqs_cache[key] = freqs
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _static_patch_timestep_embedding(estimator) -> None:
    """惰性替换 TimestepEmbedder.timestep_embedding 为 freqs 缓存版（只做一次）。

    从实际加载的模型实例取 class（不硬编码 cosyvoice2/stepaudio2.cosyvoice2 包路径）。
    失败必须显式报错，禁止静默回退。
    """
    global _static_freqs_patched
    if _static_freqs_patched:
        return
    t_embedder = getattr(estimator, "t_embedder", None)
    if t_embedder is None:
        raise RuntimeError(
            "[static-freqs] estimator has no t_embedder attribute; cannot apply timestep freqs cache"
        )
    cls = type(t_embedder)
    if not hasattr(cls, "timestep_embedding"):
        raise RuntimeError(
            f"[static-freqs] {cls.__module__}.{cls.__name__} has no timestep_embedding; cannot patch"
        )
    cls.timestep_embedding = staticmethod(_patched_timestep_embedding)  # type: ignore[method-assign]
    _static_freqs_patched = True
    logger.info(
        "[static-freqs] enabled: freqs cache installed on %s.%s (timestep embedding)",
        cls.__module__,
        cls.__name__,
    )


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _graph_config(model: object) -> dict[str, object]:
    config = getattr(getattr(model, "vllm_config", None), "additional_config", None)
    return dict(config) if isinstance(config, Mapping) else {}


def prepare_code2wav_graph_runtime() -> None:
    """Select graph-capturable ACLNN kernels before Token2Wav is loaded."""
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError(
            "MiniCPM-o Code2Wav NPUGraph capture is incompatible with "
            "ASCEND_LAUNCH_BLOCKING=1; unset it or set it to 0 before startup."
        )
    npu = torch.npu
    npu.config.allow_internal_format = False
    npu.set_compile_mode(jit_compile=False)
    logger.info("Configured MiniCPM-o Code2Wav NPUGraph runtime (allow_internal_format=False, jit_compile=False)")


def _flow_execution_context(device: torch.device, *, require_math: bool):
    if device.type != "npu":
        return nullcontext()
    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    return npu_token2wav_sdpa_context(require_math=require_math)


def _graphable_estimator_step(
    backend,
    estimator,
    *,
    x,
    mu,
    time_embedding,
    speakers,
    cond,
    cnn_cache,
    att_cache,
):
    """Run the CFM estimator body after host-backed timestep embedding."""
    width = int(x.shape[-1])
    speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
    estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
    cnn_out, att_out = backend._estimator_buffers(estimator, estimator_input, att_cache)
    old_cnn = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
    old_att = att_cache if att_cache is not None else [None] * len(estimator.blocks)
    result = estimator.blocks_forward_chunk(
        estimator_input,
        time_embedding,
        None,
        old_cnn,
        old_att,
        cnn_out,
        att_out,
    )
    return result, cnn_out, att_out


def _patched_estimator_step(
    self,
    estimator,
    *,
    x,
    mu,
    time,
    speakers,
    cond,
    cnn_cache,
    att_cache,
):
    assert _original_estimator_step is not None
    graph_runner = _backend_graph_runners.get(self)
    # exp/static-freqs: 首次 estimator 调用即安装 freqs 缓存（此时 t_embedder 实例可用；
    # warmup/首次请求完成缓存填充，之后 graph capture 不再创建新张量）。
    _static_patch_timestep_embedding(estimator)
    # NOTE: adapted for our 0.25 tree — upstream's newer BatchedToken2Wav
    # carries `_trt_stepper` / `_cfm_graph_wrapper` backends which do not exist
    # here; getattr defaults keep the guard equivalent (None == no TRT/graph
    # wrapper backend).
    if (
        graph_runner is None
        or getattr(self, "_trt_stepper", None) is not None
        or getattr(self, "_cfm_graph_wrapper", None) is not None
    ):
        return _original_estimator_step(
            self,
            estimator,
            x=x,
            mu=mu,
            time=time,
            speakers=speakers,
            cond=cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
    if (cnn_cache is None) != (att_cache is None):
        raise ValueError("estimator CNN and attention caches must both be present or absent")

    # The upstream embedder creates a frequency tensor on the host. Keep it
    # outside capture while retaining the tensor-only estimator body in graph.
    time_embedding = estimator.t_embedder(time).unsqueeze(1)
    if cnn_cache is None:
        return graph_runner.run(
            "cfm_estimator",
            (x, mu, time_embedding, speakers, cond),
            (False,),
            lambda step_x, step_mu, step_time, step_speakers, step_cond: _graphable_estimator_step(
                self,
                estimator,
                x=step_x,
                mu=step_mu,
                time_embedding=step_time,
                speakers=step_speakers,
                cond=step_cond,
                cnn_cache=None,
                att_cache=None,
            ),
        )

    return graph_runner.run(
        "cfm_estimator",
        (x, mu, time_embedding, speakers, cond, cnn_cache, att_cache),
        (True,),
        lambda step_x, step_mu, step_time, step_speakers, step_cond, step_cnn, step_att: _graphable_estimator_step(
            self,
            estimator,
            x=step_x,
            mu=step_mu,
            time_embedding=step_time,
            speakers=step_speakers,
            cond=step_cond,
            cnn_cache=step_cnn,
            att_cache=step_att,
        ),
    )


def _patched_setup_batch(self, features, batch_size):
    assert _original_setup_batch is not None
    with _flow_execution_context(
        features.speech_tokens.device,
        require_math=self in _backend_graph_runners,
    ):
        return _original_setup_batch(self, features, batch_size)


def _patched_decode_batch(
    self,
    tokens,
    features,
    states,
    *,
    last_chunk,
    flush_encoder=False,
):
    assert _original_decode_batch is not None
    with _flow_execution_context(
        tokens.device,
        require_math=self in _backend_graph_runners,
    ):
        return _original_decode_batch(
            self,
            tokens,
            features,
            states,
            last_chunk=last_chunk,
            flush_encoder=flush_encoder,
        )


def _patched_build_backend(self) -> None:
    if self.backend is not None:
        return

    config = _graph_config(self)
    max_graphs = max(0, int(config.get(_MAX_GRAPHS_KEY, 32)))
    graph_enabled = max_graphs > 0 and _config_bool(config.get(_ENABLE_KEY), False)
    if graph_enabled:
        # NPUOmniPlatform enables internal format for quantized LLM kernels.
        # Code2Wav uses regular convolution kernels that must remain in the
        # graph-capturable ACLNN path.
        prepare_code2wav_graph_runtime()

    assert _original_build_backend is not None
    _original_build_backend(self)

    graph_runner = None
    if graph_enabled:
        graph_runner = NPUExactGraphRunner(
            max_graphs=max_graphs,
            component_name="MiniCPM-o Code2Wav",
            disable_config_hint=(
                "set platforms.npu.stages[stage_id=2].additional_config.code2wav_enable_npu_graph=false"
            ),
        )
        if self.backend.speech_window.device.type == "npu" and not graph_runner.is_supported():
            raise RuntimeError(
                "MiniCPM-o Code2Wav NPUGraph capture requires torch.npu "
                "NPUGraph, graph, is_current_stream_capturing, and synchronize APIs."
            )
        if self.backend.flow.training:
            raise ValueError("MiniCPM-o Code2Wav NPUGraph capture requires flow.eval()")
        _backend_graph_runners[self.backend] = graph_runner

    if graph_enabled:
        logger.info(
            "MiniCPM-o Code2Wav NPUGraph replay enabled (max_graphs=%d)",
            max_graphs,
        )


def apply_minicpmo_4_5_code2wav_patch() -> None:
    """Patch the generic Code2Wav backend builder with Ascend acceleration."""
    global _PATCHED, _original_build_backend
    global _original_decode_batch, _original_estimator_step, _original_setup_batch
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2Wav,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
        MiniCPMO45Code2Wav,
    )

    _original_build_backend = MiniCPMO45Code2Wav._build_backend
    _original_estimator_step = BatchedToken2Wav._estimator_step
    _original_setup_batch = BatchedToken2Wav.setup_batch
    _original_decode_batch = BatchedToken2Wav.decode_batch

    MiniCPMO45Code2Wav._build_backend = _patched_build_backend  # type: ignore[method-assign]
    BatchedToken2Wav._estimator_step = _patched_estimator_step  # type: ignore[method-assign]
    BatchedToken2Wav.setup_batch = _patched_setup_batch  # type: ignore[method-assign]
    BatchedToken2Wav.decode_batch = _patched_decode_batch  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for MiniCPM-o 4.5 Code2Wav")
