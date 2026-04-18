# Latent Skill RL

End-to-end RL training of latent skill representations for LLM agents. Built on SkillRL/verl + LatentMem-inspired Composer.

**Goal**: Compress SkillRL's text skills (~800-1200 tokens) into trainable latent tokens (~2 tokens per skill), then optimize them via RL on ALFWorld. Target NeurIPS.

## Architecture (Plan C)

```
Two model copies (both load from same v2 checkpoint):
├── Actor (FSDP-wrapped, 7B frozen): standard verl RL pipeline
└── Encoder (frozen, no FSDP, +14GB/GPU): used by Composer to encode skills

Composer:
- query_latents (2 × 3584 = 7K trainable params, the ONLY trainable thing)
- encode_skills(): appends query_latents to skill text → forward through Encoder
                   → extract last k hidden states (latent vectors with grad)

Integration:
- Actor's embed_tokens monkey-patched: SKILL token positions replaced with Composer output
- LM head: v2 checkpoint has lm_head[SKILL_tokens] = 0 (model never generates SKILL tokens)
- vLLM rollout: writes Composer's latents into VocabParallelEmbedding (TP-aware)
- Per micro-batch re-encode: ensures fresh gradient graph for backward
```

See `CLAUDE.md` for detailed architecture, gotchas, and historical context.

## Setup

### 1. Clone with submodules
```bash
git clone git@github.com:WillChow66/LatentSkill.git
cd LatentSkill
```

### 2. Clone SkillRL/verl (separate, large)
```bash
git clone <SkillRL repo> SkillRL
```

### 3. Apply our verl patches
The `verl_patches/` directory mirrors SkillRL's verl directory structure with our modifications. Apply with:

```bash
cp -r verl_patches/verl/* SkillRL/verl/
```

Modified verl files:
- `verl/workers/fsdp_workers.py` - load Composer + separate Encoder copy
- `verl/workers/sharding_manager/fsdp_vllm.py` - sync Composer output to vLLM (TP-aware)
- `verl/workers/actor/dp_actor.py` - per-micro-batch re-encode + tensordict fix
- `verl/workers/critic/dp_critic.py` - tensordict `.keys()` fix
- `verl/trainer/ppo/ray_trainer.py` - best-checkpoint tracking + tensordict fix
- `verl/utils/vllm_utils.py` - LoRA import fallback for vLLM 0.19.0

### 4. Environment setup
DeltaAI specific (modify `scripts/setup_env.sh` for your cluster):
```bash
source scripts/setup_env.sh
```

Required versions:
- vLLM 0.19.0 (system-provided on DeltaAI; may need adjustment elsewhere)
- transformers 4.57.3
- tokenizers 0.22.0

### 5. Download data
```bash
sbatch scripts/alfworld_official_download.sbatch
```

## Training Pipeline

### Stage 1: Pretrain Composer (offline distillation)
Trains `query_latents` so that latent tokens replacing text skills produce equivalent agent behavior.

```bash
sbatch scripts/train_composer.sbatch
# Output: data/debug_composer/trained_query_latents.pt
```

### Stage 2: Expand vocabulary
Adds 88 SKILL tokens to the SFT checkpoint, untie LM head, set SKILL output weights to 0.

```bash
sbatch scripts/expand_vocab_v2.sbatch
# Output: /work/.../checkpoints/latent_skills_token_v2/
```

### Stage 3: End-to-end RL training
Composer in the RL loop, query_latents updated by RL reward.

```bash
# Quick verification (5 epochs, ~2h)
sbatch scripts/debug_rl_e2e.sbatch

# Full run (150 epochs, ~48h) - matches SkillRL hyperparameters
sbatch scripts/run_rl_e2e_full.sbatch
```

Monitor via wandb (project=`latentskill`).

## Key Files

| File | Purpose |
|------|---------|
| `src/skill_composer.py` | SkillComposer module + monkey-patch setup |
| `src/expand_vocab_with_skills.py` | Vocabulary expansion + LM head untie |
| `src/train_composer.py` | Stage 1 distillation training |
| `src/eval_latent_alfworld.py` | Evaluation on ALFWorld valid_seen |
| `src/encode_skill_library.py` | Offline skill encoding utility |
| `src/verl_latent/` | Custom verl extensions (SFT trainer, dataset) |
| `verl_patches/verl/` | Modified verl files (apply via cp) |
| `scripts/*.sbatch` | DeltaAI SLURM scripts |
| `CLAUDE.md` | Detailed architecture notes & historical context |

## Hyperparameters (matching SkillRL)

```
RL:
  algorithm:        GRPO
  lr (composer):    5e-3 (only query_latents)
  base model:       frozen
  train_batch_size: 16
  val_batch_size:   64
  group_size:       8
  max_prompt_length: 4096
  max_response_length: 512
  epochs: 150
  KL coef: 0.01
```

## Checkpoints

Two types saved:
1. **Latest ckpt** (`global_step_N/actor/`, ~14GB) - rotated by `max_actor_ckpt_to_keep=1`
2. **Best ckpt** (`best/actor/`, ~14GB) - by validation success rate
3. **`query_latents_step{N}.pt`** (~28KB each) - all kept, for analysis

## Paper Story

**Problem**: SkillRL's text skills are (1) token-expensive ~800-1200 tokens, (2) non-differentiable, (3) baked into model weights after RL.

**Contribution**:
1. Differentiable skills: latent tokens are continuous, RL reward directly optimizes Composer's `query_latents`
2. Modular: base model frozen, skills are pluggable SKILL tokens
3. Token-efficient: 2 tokens per skill vs 800-1200

## Acknowledgments

- [SkillRL](https://arxiv.org/abs/2602.08234) — RL infrastructure (verl)
- [LatentMem](https://arxiv.org/abs/2602.03036) — Composer architecture
- [TokMem](https://github.com/MANGA-UOFA/TokMem) — vocab integration pattern
