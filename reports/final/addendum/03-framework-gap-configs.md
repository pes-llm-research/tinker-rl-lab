### 10.3 Framework-Gap Deep-Dive: Configuration Disclosure & Retraction

**Addresses reviewer concerns:** W6 (byte-identical contradiction), Q3 (exact config dumps)

**Paper section added:** `paper/sections/framework_configs_appendix.tex`
**Reproducibility:** `experiments/framework_config_dumps/{trl,tinker,openrlhf,verl}_qwen3_8b_gsm8k.yaml` + `README.md`

#### Retraction of the "Byte-Identical" Phrasing

The main text previously described the cross-framework comparison as using "byte-identical training configs." A reviewer (W6) correctly flagged this as internally inconsistent: we simultaneously acknowledged that the Tinker-managed runtime applies "managed reference-model offload and tuned rollout defaults," which cannot be byte-identical to anything. We retract the phrase. The comparison is now framed as a **matched-configuration protocol**: every hyperparameter the user can set is harmonized across frameworks, while acknowledging explicitly that five Tinker-internal fields are managed and cannot be controlled from the API. This is a behavioural, not a bit-level, matching.

#### What Was Held Constant (31 of 47 tracked fields)

Verified byte-for-byte identical across all four YAML dumps:

- Base model weights and tokenizer (same SHA-256 of the Qwen3-8B HuggingFace snapshot)
- LoRA `rank=16`, `alpha=32`, `target_modules={q,k,v,o}_proj`, `dropout=0.0`
- GRPO `group_size G=8`, `rollouts_per_group K=1`, `max_new_tokens=384`
- Sampling `temperature=0.7`, `top_p=0.95`, `do_sample=true`
- Optimizer `adamw`, `lr=1e-6`, `weight_decay=0.0`, `betas=(0.9,0.999)`, `eps=1e-8`
- Reward `type=binary_outcome`; seed `42`
- Prompt template and response-only loss mask
- Total tokens seen by the optimizer (matched, not total steps)
- Runtime fields present in every framework (`bf16=true`, `tensor_parallel_size`, etc.)

#### What Was Framework-Managed (11 documented + 5 Tinker-managed)

**11 framework-specific but documented** (user-visible, intentionally different):
reference-model placement (`on_device` / `managed` / `separate_process` / `sharded_ray`), `reference_model.offload`, `reference_model.recompute`, rollout micro-partitioning, importance-sampling granularity (token vs sequence), KL β default (0.04 vs 0.02), `kl.surrogate`, `runtime.gradient_accumulation_steps`, `runtime.inference_backend` (vLLM in OpenRLHF/veRL), `runtime.gpu_memory_utilization`, `runtime.num_minibatches` (veRL-only), and `reward.broadcast_precision`.

**5 Tinker-managed** (serialized as `null  # managed_by_tinker` in `tinker_qwen3_8b_gsm8k.yaml`, cannot be harmonized at the API level):
`kl.beta`, `kl.surrogate`, `importance_sampling.granularity`, `tokens_per_optimizer_step`, `reward.broadcast_precision`.

#### Per-Framework Configuration Table

| Hyperparameter | TRL | TINKER | OpenRLHF | veRL |
|---|---|---|---|---|
| LoRA rank *r* | 16 | 16 | 16 | 16 |
| LoRA α | 32 | 32 | 32 | 32 |
| group size *G* | 8 | 8 | 8 | 8 |
| rollouts per group *K* | 1 | 1 | 1 | 1 |
| max new tokens | 384 | 384 | 384 | 384 |
| sampling temperature | 0.7 | 0.7 | 0.7 | 0.7 |
| KL β (surfaced) | 0.04 | 0.04 | 0.02 | 0.04 |
| IS granularity | token | token | seq | token |
| π_ref placement | on-dev | *managed* | separate | sharded |
| AdamW lr | 1×10⁻⁶ | 1×10⁻⁶ | 1×10⁻⁶ | 1×10⁻⁶ |
| tokens/optimizer step | 3072 | *managed* | 3072 | 3072 |

*Italicized cells are Tinker-managed and unavailable to the user.* Full YAML dumps enumerate all 47 fields; the table above is the reviewer-facing subset.

#### Revised Interpretation

Reported framework-gap differences should be read as **behavioural differences under the matched-configuration protocol**, not as "differences due to algorithmic variants alone." In particular, any advantage attributed to Tinker bakes in its managed reference-model offload, rollout micro-partitioning, and proprietary KL schedule—these are **genuine engineering contributions** but they are not algorithmic contributions, and a reader comparing algorithmic designs alone should discount them. Conversely, the OpenRLHF sequence-level IS granularity and β=0.02 default are intentional framework choices, not bugs, and remain surfaced in the YAML rather than coerced to match TRL/veRL.
