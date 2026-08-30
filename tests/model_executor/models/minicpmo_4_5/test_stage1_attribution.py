import json
import math
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5 import stage1_attribution


class _TTSConfig:
    model_stage = "tts"


class _Runner:
    def __init__(self, model=None):
        self.model_config = _TTSConfig()
        self.model = model if model is not None else object()


@pytest.fixture
def enabled_attribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(stage1_attribution, "_ENABLED", True)
    monkeypatch.setattr(stage1_attribution, "_event_record", lambda: None)
    monkeypatch.setenv(stage1_attribution._ENV_DIR, str(tmp_path))
    monkeypatch.delenv(stage1_attribution._ENV_MAX_STEPS, raising=False)
    for name in ("collector", "step", "last_sample"):
        monkeypatch.delattr(stage1_attribution._TLS, name, raising=False)
    yield tmp_path
    collector = getattr(stage1_attribution._TLS, "collector", None)
    if collector is not None:
        collector.close()


def _read_step(directory: Path) -> dict:
    paths = sorted(directory.glob("stage1_pid*.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    return next(row for row in rows if row.get("ev") == "stage1_decode_step")


def test_disabled_attribution_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage1_attribution, "_ENABLED", False)
    monkeypatch.setenv(stage1_attribution._ENV_DIR, str(tmp_path))
    for name in ("collector", "step", "last_sample"):
        monkeypatch.delattr(stage1_attribution._TLS, name, raising=False)

    runner = _Runner()
    before = dict(runner.__dict__)

    assert stage1_attribution.begin_decode_step(
        runner,
        num_scheduled_tokens=[1],
        input_ids=None,
        inputs_embeds=None,
    ) is None
    assert stage1_attribution.now_ns() == 0
    assert runner.__dict__ == before
    assert not list(tmp_path.iterdir())


def test_enabled_single_token_row_has_three_finite_p2_measurements(
    enabled_attribution: Path,
) -> None:
    runner = _Runner()
    step = stage1_attribution.begin_decode_step(
        runner,
        num_scheduled_tokens=[1],
        input_ids=None,
        inputs_embeds=None,
    )
    assert step is not None

    # These calls model the three runner-owner boundaries. No device event is
    # required for these host-only regions in the CPU test environment.
    stage1_attribution.add_host_interval("input_prepare_ms", 0, 310_000)
    stage1_attribution.add_host_interval("runner_adapt_ms", 1_000_000, 2_180_000)
    stage1_attribution.add_host_interval("output_state_ms", 3_000_000, 3_720_000)
    stage1_attribution.add_host_interval("runner_input_prepare", 0, 400_000)
    stage1_attribution.finish_decode_step(step)

    row = _read_step(enabled_attribution)
    components = row["host_components_ms"]
    assert set(("input_prepare_ms", "runner_adapt_ms", "output_state_ms")) <= set(components)
    assert all(
        isinstance(components[key], (int, float))
        and math.isfinite(float(components[key]))
        and components[key] >= 0
        for key in ("input_prepare_ms", "runner_adapt_ms", "output_state_ms")
    )
    assert components["runner_input_prepare"] == pytest.approx(0.4)
    assert row["host_step_ms"] >= 0
    assert row["stage"] == 1
    assert row["num_scheduled_tokens"] == [1]


def test_ineligible_prefill_is_not_labeled_as_stage1_decode(enabled_attribution: Path) -> None:
    runner = _Runner()

    assert stage1_attribution.begin_decode_step(
        runner,
        num_scheduled_tokens=[2],
        input_ids=None,
        inputs_embeds=None,
    ) is None
    assert stage1_attribution.begin_decode_step(
        runner,
        num_scheduled_tokens=[],
        input_ids=None,
        inputs_embeds=None,
    ) is None

    runner.model_config.model_stage = "llm"
    assert stage1_attribution.begin_decode_step(
        runner,
        num_scheduled_tokens=[1],
        input_ids=None,
        inputs_embeds=None,
    ) is None
    assert not list(enabled_attribution.glob("stage1_pid*.jsonl"))


def test_boundary_markers_are_at_the_intended_source_spans() -> None:
    repo = Path("/vllm-workspace/vllm-omni")
    ar_source = (repo / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py").read_text()
    model_source = (repo / "vllm_omni/platforms/npu/worker/npu_model_runner.py").read_text()

    prepare_start = ar_source.index("stage1_input_prepare_start_ns =")
    preprocess_call = ar_source.index(") = self._preprocess(", prepare_start)
    input_marker = ar_source.index('"input_prepare_ms"', preprocess_call)
    assert input_marker > preprocess_call
    assert ar_source.count('"input_prepare_ms"') == 1

    adapt_start = ar_source.index("stage1_runner_adapt_start_ns")
    prepare_runner = ar_source.index("self.model.prepare_runner_inputs", adapt_start)
    update_rope = ar_source.index("update_cos_sin(positions)", prepare_runner)
    adapt_marker = ar_source.index('"runner_adapt_ms"', update_rope)
    assert adapt_start < prepare_runner < update_rope < adapt_marker
    assert ar_source.count('"runner_adapt_ms"') == 1

    make_output = model_source.index("self.model.make_omni_output")
    output_marker = model_source.index('"output_state_ms"', make_output)
    assert output_marker > make_output
    assert model_source.count('"output_state_ms"') == 1
