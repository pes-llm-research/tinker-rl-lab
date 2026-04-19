# Per-Framework Config Dumps (Qwen3-8B / GSM8K)

This directory is the reproducibility supplement for
`paper/sections/framework_configs_appendix.tex`. It ships one exact
hyperparameter dump per framework for the canonical
Qwen3-8B + GSM8K matched-configuration run.

## Contents

| File | Framework | Build                          |
|------|-----------|--------------------------------|
| `trl_qwen3_8b_gsm8k.yaml`      | huggingface/trl `GRPOTrainer`    | on-device reference model |
| `tinker_qwen3_8b_gsm8k.yaml`   | Thinking Machines Tinker (managed) | managed reference-model offload |
| `openrlhf_qwen3_8b_gsm8k.yaml` | OpenRLHF (vLLM + Ray)            | separate reference-serving process |
| `verl_qwen3_8b_gsm8k.yaml`     | Bytedance veRL (Ray + vLLM)      | sharded Ray reference placement |

Together these four YAMLs enumerate the 47 hyperparameter fields the
appendix reports. Readers verify fairness by diffing them pairwise.

## The 31 / 11 / 5 Split

Per Section "Complete Configuration Dumps" of the appendix:

- **31 identical** across all four frameworks: LoRA `rank`/`alpha`/
  `target_modules`/`dropout`; GRPO `group_size`/`rollouts_per_group`/
  `max_new_tokens`; sampling `temperature`/`top_p`/`top_k`/`do_sample`;
  optimizer `name`/`lr`/`weight_decay`/`betas`/`eps`; reward `type`;
  metadata `model`/`model_sha256`/`task`/`seed`/`source_git_sha`; plus
  runtime fields present in every framework (bf16, `tensor_parallel_size`,
  etc.). These are the 31 cells in the four YAMLs that match byte-for-byte.
- **11 framework-specific but documented**: KL `beta` (0.04 TRL/veRL,
  0.02 OpenRLHF), `kl.surrogate`, `importance_sampling.granularity`
  (token for TRL/veRL, sequence for OpenRLHF), `reference_model.placement`
  (`on_device` / `separate_process` / `sharded_ray`),
  `reference_model.offload`, `reference_model.recompute`,
  `runtime.gradient_accumulation_steps`,
  `runtime.inference_backend` (vLLM for OpenRLHF/veRL),
  `runtime.gpu_memory_utilization`, `runtime.num_minibatches` (veRL
  only), and `reward.broadcast_precision`.
- **5 Tinker-managed**: `kl.beta`, `kl.surrogate`,
  `importance_sampling.granularity`, `tokens_per_optimizer_step`, and
  `reward.broadcast_precision`. All five are serialized as
  `null  # managed_by_tinker` in `tinker_qwen3_8b_gsm8k.yaml` and are
  the reason the appendix retracts the original "byte-identical"
  framing.

## Per-YAML Notes

### `trl_qwen3_8b_gsm8k.yaml`
* `kl.beta: 0.04` matches the DeepSeekMath default that `trl`
  inherited.
* `importance_sampling.granularity: token` is the TRL
  GRPOTrainer default (advantage applied at the token level).
* `reference_model.placement: on_device` means `pi_ref` is colocated
  with the actor; no offload.
* `tokens_per_optimizer_step: 3072` = `G (=8) x K (=1) x L_bar (=384)`.

### `tinker_qwen3_8b_gsm8k.yaml`
* Five fields are `null  # managed_by_tinker` (see list above). These
  are **not** set by the user and therefore cannot be harmonised with
  the other frameworks at the API level.
* `reference_model.placement: managed` indicates Tinker transparently
  offloads/recomputes `pi_ref`, consistent with the text of the
  appendix.
* All surfaced values match the cross-framework defaults.

### `openrlhf_qwen3_8b_gsm8k.yaml`
* `kl.beta: 0.02` is OpenRLHF's default (half the TRL/veRL value).
* `importance_sampling.granularity: sequence` is the OpenRLHF default
  at the time of writing; the appendix flags this as a genuine
  framework-specific deviation.
* `reference_model.placement: separate_process` reflects the Ray-actor
  placement OpenRLHF uses for `pi_ref`.

### `verl_qwen3_8b_gsm8k.yaml`
* `kl.beta: 0.04` matches DeepSeekMath / TRL.
* `importance_sampling.granularity: token` matches TRL.
* `reference_model.placement: sharded_ray` and
  `reference_model.offload: true` reflect veRL's Ray-managed sharded
  reference model placement.
* `runtime.num_minibatches: 1` is veRL-specific (minibatched PPO
  workflow; unused in GRPO mode but surfaced by the config schema).

## Reproducing the Dumps from Live Configs

Each framework's live Pydantic config module exposes a `to_yaml()`
helper (see `trl_integrations/config.py`, `openrlhf/config.py`,
`verl/config.py`). To regenerate a dump after a live run:

```bash
# TRL
python3 -c "from trl_integrations.config import TRLConfig; \
  TRLConfig().to_yaml('experiments/framework_config_dumps/trl_qwen3_8b_gsm8k.yaml')"

# OpenRLHF
python3 -c "from openrlhf.config import OpenRLHFConfig; \
  OpenRLHFConfig().to_yaml('experiments/framework_config_dumps/openrlhf_qwen3_8b_gsm8k.yaml')"

# veRL
python3 -c "from verl.config import VERLConfig; \
  VERLConfig().to_yaml('experiments/framework_config_dumps/verl_qwen3_8b_gsm8k.yaml')"
```

Tinker does not expose a Pydantic config; the dump is built by
`tinker.train_config` introspection, then the five Tinker-managed
fields are hand-redacted to `null  # managed_by_tinker`.

## Validation

Every YAML is `yaml.safe_load`-parseable:

```bash
python3 -c "import yaml,glob; \
  [yaml.safe_load(open(p)) for p in glob.glob('experiments/framework_config_dumps/*.yaml')]"
```

The trailing `# W6/Q3 addresses: reviewer request for exact config
dumps` comment in every YAML is a grep-able marker for the
wave-6 review-addressing pipeline.
