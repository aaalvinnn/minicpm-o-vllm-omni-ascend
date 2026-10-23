# MiniCPM-o 4.5 on Ascend 910C: vLLM-Omni 推理性能优化

在昇腾 Ascend 910C 单卡上，用 vLLM-Omni 部署 MiniCPM-o 4.5 全模态语音对话模型，围绕 **RTF / TTFP / TTFT** 做系统级优化（MiniCPM 挑战赛）。

本仓库基于 [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)（Apache-2.0）。第一个提交是上游快照（`009b80d6` + 上游 PR #5945），之后的每个提交都是我自己的改动，可以直接用 `git log` / `git show` 逐项查看。

## 结果

Seed-TTS 中文全量 2020 条，单并发，0 失败。

| 指标 | 官方基线 | 优化后 | 变化 |
|---|---|---|---|
| Audio RTF（mean） | 0.4423 | **0.263** | −40.5%（1.69× 加速） |
| Audio TTFP（mean） | 986 ms | **537 ms** | −45.6% |
| Text TTFT（mean） | 333 ms | **62 ms** | −81.4% |
| Text TPOT | ~17.9 ms | 16.8 ms | −6.3% |
| WER（门槛 ≤ 1.56%） | 1.41% | 1.47% | 达标 |
| SIM（门槛 ≥ 0.689） | 0.847 | 0.847 | 达标 |

多模态理解精度（全量）：Daily-Omni 78.11%（门槛 77.5%），Video-MME 69.44%（门槛 67.0%）。

## 系统与瓶颈

流水线分三个 stage：

| Stage | 作用 | 特点 |
|---|---|---|
| 0 | LLM，文本生成 | shape 相对静态，可 FULL 图捕获 |
| 1 | Talker，语音 token 自回归 | 逐 token 到达，context 动态增长，host-bound |
| 2 | Token2Wav（CFM flow + HiFT 声码器） | 动态 chunk 长度，host 调用密集 |

先把 RTF 拆成 `TTFP + 稳态 chunk 间隔`，再用 NPU profiling 定位：TTFP 占 RTF 分子约 45%，稳态节奏由 Stage 2 主导，且 Stage 2 是 host-bound（单会话 CANN 调用约 182 万次、编译约 4.1 万次），NPU 算力没吃满。后面每一项优化都对应拆解表中的一行，并用同日交错 A/B 和 WER/SIM 守卫验证。

## 优化清单

| 优化 | 要点 | 提交 |
|---|---|---|
| TTFT 流式释放 | serving 层把 LLM 阶段强制为 `FINAL_ONLY`，文本被攒到最后才发出；改为 `DELTA`。TTFT 333 → 约 62 ms | `9eb4ebb0` |
| Token2Wav flow 步数 10→3 | 3 步是最低可用步数（2 步启动失败）；connector 轮询 10 ms → 1 ms | `4a95bc3a` |
| Stage 0 cudagraph FULL | PIECEWISE → FULL，降低 TPOT | `71863acf` |
| Code2Wav NPU Graph | 移植上游 NPU Graph 到 0.25 树，只对 CFM estimator 做精确 shape 捕获与重放，HiFT 保持 eager；device 任务 −74.8%，host 调用 −78.8%，编译清零 | `82bca9b3`, `eb7bb7e6` |
| static-freqs 缓存 | 消除 DiT 时间步嵌入里每个 chunk 的小块 H2D 拷贝同步栅栏 | `27efd231` |
| Stage 1 decode 入图 | `FULL_DECODE_ONLY`，后升级为 `FULL_AND_PIECEWISE`，解决逐 token 动态序列无法 FULL 捕获的问题 | `98549005`, `019e2543` |
| CFG 单分支 | `token2wav_cfg_rate: 0.0`，去掉 CFM 无条件分支，DiT 前向 batch 减半，音质无损 | `e5067c22` |
| 首包 8 帧 + 稳态 50 帧 | 首包只等 8 个 codec 帧；稳态 50 帧摊薄固定调度成本 | `489684f3`, `f03d9bea` |

Stage 1 的 Talker 采样 host 快路径来自上游 PR #5945，已包含在基线快照里，不算作本仓库的改动。

## 做过但没有采用的实验

每一项都是单变量 A/B，不满足条件就不合入：

- Stage 2 整体 cudagraph：动态 chunk 长度与 request-owned cache 不兼容，50 个请求失败 48 个。
- Stage 1 直接 FULL：启动即崩，后来用 `FULL_DECODE_ONLY` 绕过。
- HiFT 局部 NPUGraph：图命中 50–52 ms，反而慢于 eager 43–49 ms，说明 Stage 2 在该处是设备受限而非 dispatch 受限。
- CFM 宏图、Stage 2 双流重叠：微基准否决。
- Talker RepPen 稀疏化：局部 P50 提速 14%，但 5 对同日 A/B 的配对 bootstrap 95% CI 跨过 0，端到端未脱离噪声，止损不合入。

## 实验分支（未合入 main）

0827 提交之后继续做的尝试，都基于当前 `main`，保留下来作为工作记录。

| 分支 | 内容 | 结论 |
|---|---|---|
| `exp/opt-talker-reppen` | Talker 重复惩罚：`bincount` 全词表改为 `unique` + 稀疏更新，再改为原位更新 | 局部 P50 −14%，完整 sampling P50 −9%；5 对同日 A/B 的 95% CI 跨过 0，端到端未稳定脱离噪声，止损 |
| `exp/opt-chunk-8-100` | 稳态 chunk 50 → 100（8/100 两 chunk 模式） | 实验分支，未合入 |
| `exp/opt-chunk8-100-v2` | 在上一项基础上把 graph pool 32 → 64，修复 shape 耗尽 | 实验分支，未合入 |
| `exp/opt-combo-c100-reppen` | 上面 chunk 100 与 RepPen 稀疏化的组合 | 实验分支，未合入 |
| `exp/p2-stage1-boundary-attribution` | Stage 1 runner / output 边界的耗时归因打点 | `input_prepare` 约 1.2 ms、`output_state` 约 1.5 ms，但没找到能在保持语义的前提下直接删除的冗余操作，结论为 Stop；`runner_adapt` 仅 0.007 ms |

## 关键配置

全部在 [`vllm_omni/deploy/minicpmo_4_5.yaml`](vllm_omni/deploy/minicpmo_4_5.yaml)：

```yaml
connector_get_sleep_s: 0.001
token2wav_n_timesteps: 3
token2wav_cfg_rate: 0.0
initial_codec_chunk_frames: 8
codec_chunk_frames: 50
# stage 0: cudagraph_mode: FULL
# stage 1: cudagraph_mode: FULL_AND_PIECEWISE
# stage 2: code2wav_enable_npu_graph: true, code2wav_max_npu_graphs: 32
```

`token2wav_n_timesteps` 从 Stage 2 的 connector `extra` 读取，写在 stage 顶层会被静默忽略。

## 运行

环境：昇腾 910C，CANN，torch_npu，vLLM 0.25，vllm-ascend 0.19.1rc2。服务占用 `npu:0`，ASR / SIM 评测放在 `npu:1`。

```bash
pip install -e . --no-build-isolation

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1

vllm serve <MiniCPM-o-4_5 权重路径> --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --stage-init-timeout 1200 \
  --host 0.0.0.0 --port 8091
```

Seed-TTS 评测：

```bash
vllm bench serve --omni --port 8091 --trust-remote-code \
  --max-concurrency 1 --num-warmup 3 \
  --dataset-name seed-tts --dataset-path <seed-tts-eval 路径> \
  --num-prompts 2020 --no-oversample \
  --seed-tts-wer-eval --seed-tts-wer-save-items --seed-tts-locale zh \
  --model openbmb/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
  --tokenizer <MiniCPM-o-4_5 权重路径> --seed-tts-file-ref-audio
```

## 许可

沿用上游 [Apache-2.0](LICENSE)。
