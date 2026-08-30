"""Opt-in Stage-1 Talker decode attribution.

This module is deliberately independent of the benchmark and disabled unless
``VLLM_OMNI_STAGE1_ATTRIBUTION`` is truthy.  It records host intervals directly
and uses NPU events without synchronizing the hot path.  Pending event pairs are
resolved opportunistically on later decode steps; a forced drain is reserved
for process shutdown.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

try:  # Keep imports harmless for CPU-only unit tests.
    import torch
except Exception:  # pragma: no cover - only exercised by broken environments.
    torch = None  # type: ignore[assignment]


_ENV_ENABLE = "VLLM_OMNI_STAGE1_ATTRIBUTION"
_ENV_DIR = "VLLM_OMNI_STAGE1_ATTRIBUTION_DIR"
_ENV_MAX_STEPS = "VLLM_OMNI_STAGE1_ATTRIBUTION_MAX_STEPS"

_TLS = threading.local()
_COLLECTOR_LOCK = threading.Lock()
_COLLECTORS: list["AttributionCollector"] = []


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_ENABLED = _truthy(os.environ.get(_ENV_ENABLE))


def enabled() -> bool:
    return _ENABLED


def _mode_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _safe_repr(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    try:
        text = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    return text[:limit]


def _event_record() -> Any | None:
    """Record an enabled-timing event on the current NPU stream, if possible."""
    if torch is None:
        return None
    try:
        # Do not inject event operations into an ACLGraph capture. The outer
        # runner events are recorded outside the wrapper and still cover the
        # captured model body; per-sample events are intentionally unavailable
        # only for the one-time capture call.
        try:
            from vllm.forward_context import get_forward_context

            if bool(getattr(get_forward_context(), "capturing", False)):
                return None
        except Exception:
            pass
        npu = getattr(torch, "npu", None)
        if npu is None or not hasattr(npu, "Event") or not hasattr(npu, "current_stream"):
            return None
        event = npu.Event(enable_timing=True)
        event.record(npu.current_stream())
        return event
    except Exception:
        # Attribution must never break serving if the vendor runtime rejects an
        # event in a particular capture/eager context.
        return None


def _event_elapsed_ms(start: Any | None, end: Any | None, *, force: bool) -> float | None:
    if start is None or end is None:
        return None
    try:
        if force:
            end.synchronize()
        elif not bool(end.query()):
            return None
        return float(start.elapsed_time(end))
    except Exception:
        return None


def _graph_dispatch_info(model: Any) -> dict[str, Any]:
    """Describe the observed ACLGraphWrapper dispatch without changing it."""
    info: dict[str, Any] = {
        "runner_model_type": type(model).__name__,
        "runner_model_module": type(model).__module__,
    }
    try:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
    except Exception:
        context = None

    requested_mode = getattr(context, "cudagraph_runtime_mode", None)
    descriptor = getattr(context, "batch_descriptor", None)
    info["requested_runtime_mode"] = _mode_name(requested_mode)
    info["batch_descriptor"] = _safe_repr(descriptor)

    # In the Ascend v1 runner the outer model is normally the ACLGraphWrapper.
    # The bounded fallbacks cover trees where a wrapper is nested one level down.
    candidates: list[tuple[str, Any]] = [("model", model)]
    for attr in ("runnable", "model", "talker"):
        try:
            child = getattr(model, attr, None)
        except Exception:
            child = None
        if child is not None and all(child is not obj for _, obj in candidates):
            candidates.append((f"model.{attr}", child))

    for location, candidate in candidates:
        entries = getattr(candidate, "concrete_aclgraph_entries", None)
        wrapper_mode = getattr(candidate, "runtime_mode", None)
        if entries is None or wrapper_mode is None:
            continue
        info.update(
            {
                "graph_wrapper": type(candidate).__name__,
                "graph_wrapper_location": location,
                "graph_wrapper_runtime_mode": _mode_name(wrapper_mode),
                "graph_entry_count_before": len(entries),
            }
        )
        if requested_mode is None or _mode_name(requested_mode) == "NONE":
            info["dispatch"] = "eager_runtime_none"
        elif requested_mode != wrapper_mode:
            info["dispatch"] = "eager_mode_mismatch"
        else:
            try:
                present = descriptor in entries
            except Exception:
                present = None
            info["graph_key_present_before"] = present
            info["dispatch"] = "graph_replay" if present else "graph_capture_or_first_run"
        return info

    info["dispatch"] = "no_observable_acl_wrapper"
    return info


class _SampleRecord:
    def __init__(self, request_id: str, codec_step: int, history_len: int, vocab_size: int) -> None:
        self.request_id = request_id
        self.codec_step = int(codec_step)
        self.history_len = int(history_len)
        self.vocab_size = int(vocab_size)
        self._phases: "OrderedDict[str, int]" = OrderedDict()
        self._start_ns = time.monotonic_ns()
        self._end_ns: int | None = None
        self._device_start_event = _event_record()
        self._device_end_event: Any | None = None
        self._device_finished = False

    def phase(self, name: str, start_ns: int, end_ns: int) -> None:
        if end_ns < start_ns:
            return
        self._phases[name] = self._phases.get(name, 0) + end_ns - start_ns

    def finish(self) -> None:
        self.finish_at(time.monotonic_ns())

    def finish_at(self, end_ns: int) -> None:
        if self._end_ns is None:
            self._end_ns = end_ns

    def device_finish(self) -> None:
        if not self._device_finished:
            self._device_finished = True
            self._device_end_event = _event_record()

    def as_dict(self, *, force: bool) -> dict[str, Any]:
        end_ns = self._end_ns if self._end_ns is not None else time.monotonic_ns()
        phases = {key: value / 1_000_000.0 for key, value in self._phases.items()}
        return {
            "request_id": self.request_id,
            "codec_step": self.codec_step,
            "history_len": self.history_len,
            "vocab_size": self.vocab_size,
            "host_sample_ms": (end_ns - self._start_ns) / 1_000_000.0,
            "host_phase_ms": phases,
            "device_sample_ms": _event_elapsed_ms(
                self._device_start_event,
                self._device_end_event,
                force=force,
            ),
        }


class _StepRecord:
    def __init__(self, collector: "AttributionCollector", metadata: dict[str, Any]) -> None:
        self.collector = collector
        self.metadata = dict(metadata)
        self.host_start_ns = time.monotonic_ns()
        self.host_end_ns: int | None = None
        self.step_start_event = _event_record()
        self.step_end_event: Any | None = None
        self.body_start_event: Any | None = None
        self.body_end_event: Any | None = None
        self.body_host_start_ns: int | None = None
        self.body_host_end_ns: int | None = None
        self.model_start_event: Any | None = None
        self.model_end_event: Any | None = None
        self.model_host_start_ns: int | None = None
        self.model_host_end_ns: int | None = None
        self.samples: list[_SampleRecord] = []
        self.extra_host_ms: dict[str, float] = {}
        self._finished = False

    def begin_body(self) -> None:
        if self.body_host_start_ns is None:
            self.body_host_start_ns = time.monotonic_ns()
            self.body_start_event = _event_record()

    def end_body(self) -> None:
        if self.body_host_start_ns is not None and self.body_host_end_ns is None:
            self.body_host_end_ns = time.monotonic_ns()
            self.body_end_event = _event_record()

    def begin_model(self) -> None:
        if self.model_host_start_ns is None:
            self.model_host_start_ns = time.monotonic_ns()
            self.model_start_event = _event_record()

    def end_model(self) -> None:
        if self.model_host_start_ns is not None and self.model_host_end_ns is None:
            self.model_host_end_ns = time.monotonic_ns()
            self.model_end_event = _event_record()

    def sample_begin(self, request_id: str, codec_step: int, history_len: int, vocab_size: int) -> _SampleRecord:
        sample = _SampleRecord(request_id, codec_step, history_len, vocab_size)
        self.samples.append(sample)
        return sample


    def add_host_interval(self, name: str, start_ns: int, end_ns: int) -> None:
        if end_ns < start_ns:
            return
        self.extra_host_ms[name] = self.extra_host_ms.get(name, 0.0) + (end_ns - start_ns) / 1_000_000.0

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.host_end_ns = time.monotonic_ns()
        self.step_end_event = _event_record()
        for sample in self.samples:
            sample.device_finish()
            sample.finish()

    def resolve(self, *, force: bool) -> dict[str, Any] | None:
        if not self._finished:
            return None
        device_step_ms = _event_elapsed_ms(self.step_start_event, self.step_end_event, force=force)
        if self.step_start_event is not None and self.step_end_event is not None and device_step_ms is None:
            return None
        device_body_ms = _event_elapsed_ms(self.body_start_event, self.body_end_event, force=force)
        if self.body_start_event is not None and self.body_end_event is not None and device_body_ms is None:
            return None
        device_model_ms = _event_elapsed_ms(self.model_start_event, self.model_end_event, force=force)
        if self.model_start_event is not None and self.model_end_event is not None and device_model_ms is None:
            return None

        host_end_ns = self.host_end_ns if self.host_end_ns is not None else time.monotonic_ns()
        host_step_ms = (host_end_ns - self.host_start_ns) / 1_000_000.0
        host_body_ms = None
        if self.body_host_start_ns is not None and self.body_host_end_ns is not None:
            host_body_ms = (self.body_host_end_ns - self.body_host_start_ns) / 1_000_000.0
        host_model_ms = None
        if self.model_host_start_ns is not None and self.model_host_end_ns is not None:
            host_model_ms = (self.model_host_end_ns - self.model_host_start_ns) / 1_000_000.0
        sample_payloads = [sample.as_dict(force=force) for sample in self.samples]
        if any(
            sample._device_start_event is not None
            and sample._device_end_event is not None
            and payload.get("device_sample_ms") is None
            for sample, payload in zip(self.samples, sample_payloads, strict=True)
        ):
            return None
        device_codec_sampling_ms = sum(
            float(sample_payload["device_sample_ms"])
            for sample_payload in sample_payloads
            if sample_payload.get("device_sample_ms") is not None
        )
        host_codec_sampling_ms = sum(float(sample_payload["host_sample_ms"]) for sample_payload in sample_payloads)

        payload: dict[str, Any] = {
            "ev": "stage1_decode_step",
            "pid": os.getpid(),
            "host_step_ms": host_step_ms,
            "host_body_ms": host_body_ms,
            "host_after_body_ms": host_step_ms - host_body_ms if host_body_ms is not None else None,
            "host_model_ms": host_model_ms,
            "host_after_model_ms": host_step_ms - host_model_ms if host_model_ms is not None else None,
            "device_step_ms": device_step_ms,
            "device_body_ms": device_body_ms,
            "device_model_ms": device_model_ms,
            "device_non_model_ms": (
                device_body_ms - device_model_ms
                if device_body_ms is not None and device_model_ms is not None
                else None
            ),
            "device_codec_sampling_ms": device_codec_sampling_ms or None,
            "host_codec_sampling_ms": host_codec_sampling_ms or None,
            "host_device_gap_ms": (
                host_body_ms - device_body_ms
                if host_body_ms is not None and device_body_ms is not None
                else None
            ),
            "device_timing_available": device_step_ms is not None,
            "host_components_ms": dict(self.extra_host_ms),
            "samples": sample_payloads,
            **self.metadata,
        }
        return payload


class AttributionCollector:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.path = os.path.join(directory, f"stage1_pid{os.getpid()}.jsonl")
        self._fp = None
        self._pending: list[_StepRecord] = []
        self._written_steps = 0
        try:
            os.makedirs(directory, exist_ok=True)
            self._fp = open(self.path, "a", encoding="utf-8", buffering=1)
            self._write(
                {
                    "ev": "meta",
                    "pid": os.getpid(),
                    "clock": "CLOCK_MONOTONIC",
                    "python": os.sys.version.split()[0],
                    "torch": getattr(torch, "__version__", None),
                    "max_steps": self.max_steps,
                }
            )
        except Exception:
            self._fp = None

    @property
    def max_steps(self) -> int:
        try:
            return max(0, int(os.environ.get(_ENV_MAX_STEPS, "0")))
        except (TypeError, ValueError):
            return 0

    def _write(self, payload: dict[str, Any]) -> None:
        if self._fp is None:
            return
        try:
            self._fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            # Do not turn optional diagnostics into a serving failure.
            self._fp = None

    def drain(self, *, force: bool = False) -> None:
        remaining: list[_StepRecord] = []
        for step in self._pending:
            payload = step.resolve(force=force)
            if payload is None:
                remaining.append(step)
                continue
            if self.max_steps <= 0 or self._written_steps < self.max_steps:
                self._write(payload)
                self._written_steps += 1
        self._pending = remaining

    def begin_step(self, metadata: dict[str, Any]) -> _StepRecord | None:
        self.drain()
        if self.max_steps > 0 and self._written_steps >= self.max_steps:
            return None
        step = _StepRecord(self, metadata)
        _TLS.step = step
        return step

    def finish_step(self, step: _StepRecord | None) -> None:
        if step is None:
            return
        step.finish()
        if getattr(_TLS, "step", None) is step:
            _TLS.step = None
        self._pending.append(step)
        self.drain()

    def close(self) -> None:
        try:
            self.drain(force=True)
        finally:
            if self._fp is not None:
                try:
                    self._fp.flush()
                    self._fp.close()
                except Exception:
                    pass
                self._fp = None


def get_collector() -> AttributionCollector | None:
    if not enabled():
        return None
    collector = getattr(_TLS, "collector", None)
    if collector is not None:
        return collector
    directory = os.environ.get(_ENV_DIR, "/workspace/experiments/p0_stage1_attribution")
    with _COLLECTOR_LOCK:
        collector = AttributionCollector(directory)
        _COLLECTORS.append(collector)
        _TLS.collector = collector
        atexit.register(collector.close)
    return collector


def current_step() -> _StepRecord | None:
    return getattr(_TLS, "step", None)


def begin_decode_step(runner: Any, *, num_scheduled_tokens: Any, input_ids: Any, inputs_embeds: Any) -> _StepRecord | None:
    """Create a record only for a Stage-1 all-decode runner invocation."""
    if not enabled():
        return None
    try:
        runner_config = getattr(runner, "model_config", None)
        stage = getattr(runner_config, "model_stage", None)
    except Exception:
        stage = None
    if stage != "tts":
        return None
    try:
        values = [int(value) for value in num_scheduled_tokens]
    except Exception:
        return None
    if not values or any(value != 1 for value in values):
        return None

    collector = get_collector()
    if collector is None:
        return None
    metadata: dict[str, Any] = {
        "stage": 1,
        "num_reqs": len(values),
        "num_scheduled_tokens": values,
        "input_ids_shape": list(getattr(input_ids, "shape", ())) if input_ids is not None else None,
        "inputs_embeds_shape": list(getattr(inputs_embeds, "shape", ())) if inputs_embeds is not None else None,
    }
    metadata.update(_graph_dispatch_info(getattr(runner, "model", runner)))
    return collector.begin_step(metadata)


def finish_decode_step(step: _StepRecord | None) -> None:
    if step is None:
        return
    step.collector.finish_step(step)


def update_step_metadata(step: _StepRecord | None, **values: Any) -> None:
    if step is not None:
        step.metadata.update(values)


def body_begin() -> _StepRecord | None:
    step = current_step()
    if step is not None:
        step.begin_body()
    return step


def body_end(step: _StepRecord | None) -> None:
    if step is not None:
        step.end_body()


def model_begin() -> _StepRecord | None:
    step = current_step()
    if step is not None:
        step.begin_model()
    return step


def model_end(step: _StepRecord | None) -> None:
    if step is not None:
        step.end_model()


def refresh_dispatch(step: _StepRecord | None, model: Any, *, prefix: str) -> None:
    """Snapshot graph dispatch state at a forward-context boundary."""
    if step is None:
        return
    info = _graph_dispatch_info(model)
    for key in (
        "requested_runtime_mode",
        "batch_descriptor",
        "graph_wrapper",
        "graph_wrapper_location",
        "graph_wrapper_runtime_mode",
        "graph_entry_count_before",
        "graph_key_present_before",
        "dispatch",
    ):
        if key in info:
            step.metadata[f"{prefix}_{key}"] = info[key]
    if prefix != "after":
        return
    before_dispatch = step.metadata.get("context_dispatch", step.metadata.get("dispatch"))
    before_count = step.metadata.get("context_graph_entry_count_before")
    after_count = info.get("graph_entry_count_before")
    if before_dispatch == "graph_replay":
        step.metadata["dispatch_actual"] = "graph_replay"
    elif (
        before_dispatch == "graph_capture_or_first_run"
        and isinstance(before_count, int)
        and isinstance(after_count, int)
        and after_count > before_count
    ):
        step.metadata["dispatch_actual"] = "graph_capture"
    elif before_dispatch in {"eager_runtime_none", "eager_mode_mismatch"}:
        step.metadata["dispatch_actual"] = before_dispatch
    else:
        step.metadata["dispatch_actual"] = "unknown"


def sample_begin(request_id: str, codec_step: int, history_len: int, vocab_size: int) -> _SampleRecord | None:
    step = current_step()
    if step is None:
        return None
    return step.sample_begin(request_id, codec_step, history_len, vocab_size)


def sample_phase(sample: _SampleRecord | None, name: str, start_ns: int, end_ns: int) -> None:
    if sample is not None:
        sample.phase(name, start_ns, end_ns)


def sample_finish(sample: _SampleRecord | None) -> None:
    if sample is not None:
        sample.device_finish()
        sample.finish()


def set_last_sample(sample: _SampleRecord | None) -> None:
    _TLS.last_sample = sample


def take_last_sample() -> _SampleRecord | None:
    sample = getattr(_TLS, "last_sample", None)
    _TLS.last_sample = None
    return sample


def sample_value_materialized(sample: _SampleRecord | None, start_ns: int, end_ns: int) -> None:
    if sample is not None:
        sample.phase("sampled_id_item", start_ns, end_ns)
        sample.device_finish()
        sample.finish_at(end_ns)


def add_host_interval(name: str, start_ns: int, end_ns: int) -> None:
    step = current_step()
    if step is not None:
        step.add_host_interval(name, start_ns, end_ns)


def add_step_host_interval(step: _StepRecord | None, name: str, start_ns: int, end_ns: int) -> None:
    if step is not None:
        step.add_host_interval(name, start_ns, end_ns)


def now_ns() -> int:
    return time.monotonic_ns() if _ENABLED else 0


def graph_dispatch_info_for_test(model: Any) -> dict[str, Any]:
    """Expose the pure-ish classifier to unit tests without enabling tracing."""
    return _graph_dispatch_info(model)
