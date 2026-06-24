# Latent Skill RL Project

## Goal
Compress SkillRL's text skills into trainable latent tokens via a SMALL Composer
model, then RL-adapt a BIG Actor to use them. Target: NeurIPS.

---

## Two SFT Paradigms (Path 1 vs Path 2)

We explore two designs for Stage 1 SFT, which produce different starting points
for the same Stage 2/3 pipeline:

| | **Path 1** (frozen-Actor SFT) | **Path 2** (joint Composer+Actor SFT) |
|---|---|---|
| Composer trainable | ✓ | ✓ |
| Actor trainable | ✗ frozen | ✓ trains too |
| query_latents + latent_proj trainable | ✓ | ✓ (both); UNTRAINED variant freezes them |
| Distributed strategy | DDP (Composer) + per-rank Actor replica | **DeepSpeed ZeRO-2** (one JointWrapper) |
| Stage 3 RL starting Actor | original Qwen2.5-7B-Instruct | trained-by-Path-2 Actor |
| Failure mode observed | k>2 → encoder-output collapse (cossim 1.0 across skills) | resists collapse via Actor's discriminative gradient |
| Reviewer story | "Composer SFT only; minimal Actor change" | "LatentMem-style joint SFT then RL" |

**Why ZeRO-2 not FSDP for Path 2**: We must call `composer.embed(skill_ids)` and
`actor.embed(before_ids)` OUTSIDE model.forward to compose `inputs_embeds`.
FSDP shards params → embed.weight is 1-D flat_param outside forward → "weight
must be 2-D" crash (we hit this 5+ times). ZeRO-2 keeps params replicated per
GPU (only grads + optimizer states sharded), so embed.weight stays 2-D and
direct access works. ZeRO-2 uses ~95 GB/GPU vs FSDP ~80 GB; both fit in 120 GB,
so we trade a bit more memory for code that just works.

---

## Architecture (3 Stages)

Two MODELS used throughout:
- **Composer** = `Qwen2.5-3B-Instruct` (small, encodes text skills → latent vectors)
- **Actor** = `Qwen2.5-7B-Instruct` (big, learns to use latent skills to solve tasks)

### Stage 1 — Composer SFT (Path 1 OR Path 2)

```
Per training sample (one parquet row from SkillRL SFT data):
  1. Parse instruction → before_text, [skill_text_1..N], after_text
  2. Compose chat template with [LATENT_SKILLS] placeholder
  3. Tokenize before_ids, after_ids, skill_ids_list (each skill separately)

Composer forward (per skill):
  4. composer.embed(skill_ids ⊕ query_latents) → composer.transformer
     → take last K hidden states = latent_repr (with grad)
  5. latent_proj(latent_repr) → D_actor space  (LLaVA-style projection)

Actor forward:
  6. inputs_embeds = [actor.embed(before_ids) | latent_reprs | actor.embed(after_ids)]
  7. actor(inputs_embeds=...) → logits
  8. CE loss on response tokens only

Backward (depends on Path):
  Path 1 (frozen-Actor):  loss → Composer + ql + proj  ✓ updated
                          loss → Actor                  ✗ frozen
  Path 2 (joint):         loss → Composer + ql + proj + Actor  ✓ all updated
                          UNTRAINED variant: only Actor updates (composer/ql/proj frozen)
```

**Key design decisions**:
- **No forward_hook on embed_tokens** (caused FSDP collapse / grad loss in earlier joint-SFT v2/v3 attempts). Direct `embed(ids)` call only safe under DDP/ZeRO-2 (params not sharded).
- **`tie_word_embeddings` must be untied before freezing lm_head**, otherwise `lm_head.weight is embed_tokens.weight` causes embed to silently freeze too. Confirmed bug Apr 30 — embed_tokens diff was 0.0; fixed by explicit untie + clone.
- **Composer.lm_head frozen** in trainers (we take hidden states, not logits).
- **Actor.lm_head trainable** in Path 2 (loss is on Actor logits).

### Stage 2 — Encode latent library (`src/encode_latent_library.py`)

One-shot. Load Stage 1 Composer + ql + proj, run encode on the 44 fixed base
skills, get 44×K latent vectors. Two variants per Path:
- TRAINED: from SFT-end Composer ckpt
- UNTRAINED: from raw `Qwen2.5-3B-Instruct` (random ql + proj, seed=42)

Save library .pt, then bake into Actor's vocab embedding (`expand_vocab_with_skills.py`)
to produce vocab-expanded Actor with 44×K extra SKILL tokens.

For Path 2: Actor used for baking is the Stage-1-trained Actor (warm-start RL).
For Path 1: Actor used for baking is original Qwen2.5-7B-Instruct.

### Stage 3 — RL: Actor + latent vocab rows trained, Composer frozen alive

Standard SkillRL/verl GRPO stack on the vocab-expanded Actor. The 44×K SKILL
token rows in vocab embedding are trainable (untied LM head trick).

**Dynamic skill bank (X2)**: failed trajectories → o3-mini → new skill text →
Composer.encode on rank 0 → broadcast → write into vLLM VocabParallelEmbedding +
Actor's vocab. New SKILL_N tokens enter future rollouts. Composer kept alive
on each rank (single-GPU per rank, frozen, eval mode).

Trainable: Actor weights + initial 44×K + dynamically-added SKILL vocab rows
Frozen but kept alive: Composer (single-GPU per rank replica, ~6GB)

### Ablation Designs (NeurIPS core experiments)

**Path 1 ablation** (Frozen Actor):

| Variant | Composer state | What we test |
|---|---|---|
| TRAINED | SFT-trained Composer | latent representation learned by Composer |
| UNTRAINED | raw Qwen2.5-3B-Instruct + random ql/proj | scaffold-only baseline |

**Path 2 ablation** (Joint train):

| Variant | Composer body | ql + latent_proj | Actor | What we test |
|---|---|---|---|---|
| TRAINED | ✓ trains | ✓ trains | ✓ trains | full joint training |
| UNTRAINED | ✗ frozen | ✗ frozen | ✓ trains | "does Composer training matter beyond Actor adaptation?" |

If TRAINED ≫ UNTRAINED (in either Path) → Composer training has discriminative
value beyond what scaffold + Actor adaptation provides.

---

## ⚠️ PARSER FIX WATERSHED (Jun 9) — ALL numbers below this section are OLD-PARSER and DEPRECATED

**Root cause found by user questioning "no-skill 0.7% vs paper 14.8%"**: the strict
`alfworld_projection` parser ONLY accepted literal `<action>...</action>`. Weak /
format-unstable models (vanilla base, k=1, partially all latent variants) emit
`[action]`/`[action>` variants ~half the time → every such step rejected → scores
systematically deflated. RL ckpt unaffected (always emits `<action>`).

**Fix**: robust parser in `SkillRL/.../alfworld/projection.py` extracts action content
from any `<`/`[` action wrapper (strict form is a subset; think-tag + Chinese checks kept).
Unit-tested on real base-model outputs.

**Validated anchors (NEW parser, 140 ep, T=0.4)**:
| Model | OLD parser | NEW parser |
|---|:-:|:-:|
| SkillRL RL ckpt (anchor) | 87.9% | **88.6%** ✓ holds |
| vanilla 7B, NO skill | 0.7% | **42.1%** (!!) |

**Implication**: the entire old control suite (no-skill 0.7% ≪ UNTRAINED 23.6% ≪ TRAINED
38-50%) conflated SKILL INFORMATION with FORMAT ANCHORING (latents trained on `<action>`
SFT data stabilize the actor's output format → looked stronger under the strict parser).
The k=1 "format corruption 0%" was the tip of this iceberg. All numbers below this block
were measured with the OLD parser and are DEPRECATED.

## ✅ FINAL NEW-PARSER RESULTS (Jun 10-11) — paper-grade tables

### Control suite (vanilla frozen 7B actor, 140 ep, T=0.4, robust parser)

| condition | success | tok/ep |
|---|:-:|:-:|
| UNTRAINED latents same-7B (raw-7B encode, k=8/k=32) | 7.1% / 0% | — |
| RANDOM latents (norm-matched noise) | 36.4% | — |
| **no skill** | **42.1%** | 34K |
| text skills in-context (top-6, ~16 skills) | 43.6% | 62K |
| text skills FULL (top-12, ~17 skills = same set as latent) | 44.3% | 69K |
| UNTRAINED latents (raw-**3B** encode, has random proj) | 44.3% | — |
| TRAINED latents k=8 (best single lib) | 50.0-52.9% | ~33K |
| SkillRL **SFT** ckpt (text; their cold-start *damages* the actor) | **15.0%** | 74K |
| SkillRL **RL** ckpt (text, ceiling) | **88.6%** | 80K |

**UNTRAINED same-7B = 7.1% (not a bug)**: metadata verified identical to TRAINED
(44 skills, Identity proj since D==D, seed 42, latent norm 246≈235). Raw-7B encodes a
*structured-but-harmful* direction (worse than random noise's 36.4%). Contrast the raw-**3B**
UNTRAINED 44.3%: its random 3B→7B projection scrambles the harmful direction into benign
noise; same-7B has Identity proj so the harm hits the frozen actor undiluted. → "training
the composer" provides +43-49pp over untrained same-model encode. Actor frozen confirmed
(q_proj=74.5417 unchanged all of training; "leak" log was a wrong-ref false alarm).

### Full k-table (same-7B composer, frozen actor, epoch4; every draw = independent training, NEW parser)

| k | n | mean ± std | min–max | note |
|---|:-:|:-:|:-:|---|
| 1 | 1 | 51.4 | — | old "0%" was PURE parser artifact (format bias, content fine) |
| 2 | 6 | 39.9 ± 9.8 | 25.0–50.7 | unstable, below baseline on avg |
| 4 | 3 | 43.8 ± 2.0 | 41.4–46.4 | ≈ baseline |
| 6 | 3 | 43.6 ± 3.6 | 40.7–48.6 | ≈ baseline |
| 8 | 6 | 45.1 ± 6.3 | 32.9–52.9 | +3.0pp, n.s. (t≈1.2) |
| 16 | 3 | 40.0 ± 3.5 | 35.7–44.3 | ≈ baseline |
| **32** | **6** | **49.1 ± 6.2** | 41.4–58.6 | **+7.0pp vs no-skill, t≈2.8 — only k with a significant edge** |

**Honest k-story**: NO LatentMem-style monotonicity under frozen-actor SFT. Training-run
variance (solutions nearly orthogonal even at same seed — cross-batch lib cossim 0.08-0.2)
dominates the k-effect; only k=32 clears the baseline significantly. k=1's 51.4 (n=1) shows
even a single latent can carry the info once format noise is tolerated by the parser.

### LoRA composer (B-arm) vs full-FT (A-arm) — same protocol, seed42 single draw, NEW parser

LoRA = LatentMem-exact recipe (r16/α32/dropout0.1, targets q_proj,v_proj). Trained on Modal
(DeltaAI quota exhausted Jun 12); eval calibrated (Modal vs cluster lib agree within ±2pp).

| k | full-FT (A) | LoRA (B) |
|---|:-:|:-:|
| 2 | 32.9 | 13.6 |
| 4 | 36.4 | 38.6 |
| 6 | 40.7 | 25.7 |
| 8 | **52.9** | 25.7 (cluster) / 33.6 (Modal) |
| 16 | 52.9 | 26.4 |
| 32 | 45.7 | 38.6 |

**LoRA verdict**: LoRA does NOT beat full-FT and is less stable (k8/16: 25-34 vs 52.9). The
"LoRA prevents overfitting" hypothesis is FALSE in our frozen-actor + two-model regime.
LatentMem uses LoRA because of its single-model + adapting-consumer regime; ours differs →
full-FT is the right choice. **RL starts from full-FT k=8** (baked actor ready on Modal).
(LoRA k8 cluster 25.7 vs Modal 33.6 = training-draw variance across two independent runs,
not an env effect — eval calibration is ±2pp.)

**Pipeline story (the paper)**: latent SFT (frozen actor, 45-49%) preserves the actor and adds
skills externally; SkillRL's text path destroys the actor at SFT (15.0%) and only recovers via
RL (88.6%). Stage-3 RL on latents = the open match point. Eval noise ±2pp; training-draw
noise ±2-10pp per config; report mean±std over ≥3 draws, never single runs.

---

# DEPRECATED OLD-PARSER HISTORY BELOW

---

## Current Eval Results — OLD PARSER (DEPRECATED, kept for history)

**Eval protocol notes (validated Jun 4-5)**: T=0 greedy DEGRADES SkillRL RL ckpt 87.9%→70.7%
(action loops) — paper protocol is T=0.4 sampling. Run-to-run sampling noise ≈ ±2pp
(two T=0.4 runs: identical 123/140 aggregate but 6/140 episode flips). Gaps <4pp need 3 seeds.
Training-seed variance is FAR larger: ±8-18pp on same config (grid, Jun 7-8).

### Control suite (all on vanilla frozen Qwen2.5-7B actor)

| vanilla 7B + | Success | tok/ep | Meaning |
|---|:-:|:-:|---|
| no skill | 1/140 (0.7%) | 41K | floor |
| text skills in-context | 1/140 (0.7%) | 80K | vanilla can't use verbose text skills |
| RANDOM latents (norm-matched noise) | 2/132 (~1.5%) | — | scaffold/any-vector effect ≈ nil |
| latent UNTRAINED (raw 3B encodes) | 33/140 (23.6%) | 39K | real info flows through pretrained encoder |
| latent TRAINED 3B k=2 | 54/140 (38.6%) | 33K | composer SFT +15pp |
| **latent TRAINED same-7B k=8** | **71/140 (50.7%)** 🏆 | — | same-model composer, no projection |

### Pipeline comparison (the paper headline)

| Stage | SkillRL (text) | Ours (latent) |
|---|:-:|:-:|
| SFT-level | 11.4% (Jianwen SFT ckpt) | **50.7%** (same7b k=8) |
| RL-level | **87.9%** (123/140) | TBD — Stage 3 pending |
| tok/ep | 76-80K | 33-39K (**-55%**) |

### K-sweep (Stage-2 eval; the projection story)

| k | 1 | 2 | 4 | 8 |
|---|:-:|:-:|:-:|:-:|
| 3B composer (latent_proj 2048→3584) | 32.1% | 38.6% | 20.0% | 32.9% |
| **same-7B composer (Identity proj)** | **0%** ⚠ anomaly | **45.7%** | **36.4%** | **50.7%** |

**Key paper-grade findings**:
1. **Three-way control complete**: no-skill ≈ text ≈ random-latent ≈ 0-1.5% ≪ UNTRAINED-latent 23.6% ≪ TRAINED. Latent representation works on a frozen actor where text fails; the gain is real information through the encoder (garbage-latent control), not scaffold.
2. **Removing cross-model projection lifts everything**: same-7B composer beats 3B at every k≥2 (+7~+18pp); k-direction flips positive (k=8 best) — closer to LatentMem's monotonic pattern. Cross-skill cossim: no dead (1.00) slots in same-7B libs vs 3B k=2 pos-1 = 1.000.
3. **Collapse mechanism (3B + frozen actor)**: cossim→1.0 at k≥4 (all slots), partial at k=2. Frozen actor gives no discriminative gradient; cross-model projection makes it worse.
4. **ql init scale ×0.02 is CRITICAL**: unscaled randn (LatentMem-style) k=2 = 10.0% vs 38.6%. Matches composer embedding std (3B 0.024 / 7B 0.014). Do not remove.
5. **same-7B k=1 anomaly RESOLVED (Jun 6): format corruption, not capacity**. k=1 latents bias the frozen actor's output toward code-symbol token directions (`_FRAGMENT`/`_BOX` nearest-vocab) → emits `[action>` instead of `<action>` in 9/9 debug responses → projection.py parser rejects every action → 0%. k=2 emits proper `<action>` 11/11; multiple latents average the bias out. `<think>` reasoning coherent in both. Paper: annotate k=1 cell as "format corruption" (robustness discussion); k≥2 conclusions unaffected. Debug evidence: logs/debug_k1_2427690.out, src/debug_k1_generations.py.
6. **Path 2 (joint 3B+7B SFT) hurt multi-step tasks** (16.4%); LoRA Path 1 weaker overall but k-direction positive (22.9%→27.9%) — low-rank resists collapse.

---

## Key Paths

| Path | What |
|------|------|
| `/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet` | SFT data (SkillRL parquet) — **re-downloaded to bfdz Jun 5; the entire `/work/nvme/bdns/xzhou10/` dir was purged** |
| `/work/nvme/bfdz/xzhou10/checkpoints/skillrl_alfworld_rl_hf/` | Jianwen SkillRL RL ckpt merged to HF (text-skill RL baseline reference) |
| **Path 1 outputs** | |
| `/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean/` | Stage 1 trained Composer (k=2, embed-untied) |
| `/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_clean/` | Stage 2 TRAINED k=2 latent library + actor_expanded_vocab |
| `/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_untrained/` | Stage 2 UNTRAINED k=2 latent library + actor_expanded_vocab |
| (k=8 ckpts deleted post-collapse-finding; eval JSONs preserved in `results/`) | |
| **Same-model (7B composer) outputs — CURRENT CHAMPIONS (Jun 5)** | |
| `/work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_k8/` | 7B composer full-FT ZeRO-2, frozen 7B actor, k=8 → **50.7% champion** |
| `/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_same7b_k8/` | k=8 library + actor_expanded_vocab (baked into vanilla 7B) — **Stage 3 RL starting point** |
| `/work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_k2/` + `latent_lib_same7b_k2/` | k=2 runner-up (45.7%) — token-efficiency candidate for RL |
| `/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_same7b_k1/` | k=1 lib.pt only (anomaly resolved: format corruption; baked actor deleted) |
| (Path 2 / LoRA / noscale / 3B-sweep k1,3,4 ckpts deleted post-eval; all eval JSONs in `results/`, small lib.pt files preserved) | |
| **Stage 3 RL outputs** | |
| `/work/nvme/bfdz/xzhou10/checkpoints/rl_trained_composer/` | Stage 3 TRAINED RL ckpts |
| `/work/nvme/bfdz/xzhou10/checkpoints/rl_untrained_composer/` | Stage 3 UNTRAINED RL ckpts |

### Key Code

| File | Purpose |
|------|------|
| **`src/train_composer_only.py`** | **Path 1**: Composer SFT (DDP, Actor frozen + per-rank replica). Supports `--use-lora` for LoRA variant. |
| **`src/train_composer_actor_joint.py`** | **Path 2**: Joint Composer + Actor SFT (DeepSpeed ZeRO-2, JointWrapper). `--freeze-composer` for UNTRAINED ablation. |
| `src/encode_latent_library.py` | Stage 2: encode 44 base skills with given Composer + ql + proj |
| `src/expand_vocab_with_skills.py` | Stage 2: bake latent vectors into Actor vocab embedding + untie LM head. Supports k>2 via `_suffix_for(j)` (a..h). |
| `src/eval_latent_alfworld.py` | Latent-skill ALFWorld eval (used after Stage 2 + Stage 3) |
| `src/eval_text_alfworld.py` | Text-skill ALFWorld eval (baseline + SkillRL ckpts) |
| `SkillRL/verl/...` | GRPO RL stack (modified for Composer dynamic encode in X2; see "Modified verl Files" below) |
| `SkillRL/agent_system/memory/skill_updater.py` | OpenAI o3-mini-driven dynamic skill text generation |
| `scripts/composer_sft_clean.sbatch` | Path 1 Stage 1 (k=2, full FT) launcher |
| `scripts/composer_sft_lora.sbatch` & `_k8.sbatch` | Path 1 LoRA variants (k=2 / k=8) |
| `scripts/joint_path2_trained.sbatch` & `joint_path2_untrained.sbatch` | Path 2 launchers (DeepSpeed ZeRO-2) |
| `scripts/joint_sft_poc.sbatch` | Path 2 smoke (40 sample × 1 epoch) |
| `scripts/encode_lib_trained.sbatch` & `_untrained.sbatch` | Stage 2 launchers (Path 1) |
| `scripts/encode_lib_path2_trained.sbatch` & `_untrained.sbatch` | Stage 2 launchers (Path 2) — bakes into Path-2-trained Actor |
| `scripts/eval_stage2_trained_clean.sbatch` etc | Stage-2-only eval (Path 1) |
| `scripts/eval_stage2_path2_trained.sbatch` & `_untrained.sbatch` | Stage-2-only eval (Path 2) |
| `scripts/rl_trained.sbatch` & `rl_untrained.sbatch` | Stage 3 launchers |
| `scripts/ds_config_zero2.json` | DeepSpeed ZeRO-2 config (Path 2 + future) |
| `src/legacy/train_joint_sft_fsdp.py` | DEPRECATED — old joint-SFT trainer (FSDP + hook hack, broken) |

---

## Hyperparameters

For Stage 1 SFT (Path 1, full FT):

| HP | Value | Source |
|---|:-:|---|
| Composer encoder lr | 1e-5 | verl SFT default |
| query_latents lr | 5e-3 | tuned for tiny param count |
| latent_proj lr | 5e-4 | LLaVA-style projection mid-range |
| betas | (0.9, 0.95) | verl SFT default |
| weight_decay (Composer) | 0.01 | verl SFT default |
| weight_decay (ql / proj) | 0.0 | tiny params, no regularization |
| warmup_steps_ratio | 0.1 | verl SFT default |
| lr_scheduler | cosine | verl SFT default |
| total_epochs | 4 | verl SFT default |
| clip_grad | 1.0 | verl SFT default |
| max_length | 4096 | ALFworld prompts are long |
| mixed_precision | bf16 (composer fp32) | bf16 update-rounds-to-zero around |w|=74 |
| latents_per_skill (K) | 2 (best), 8 (collapse) | k=2 is sweet spot for Path 1 |

For Stage 1 SFT (Path 2, joint):
- Same as Path 1 plus: `actor_lr = 1e-5` (4 LR groups: composer/actor/ql/proj)
- DeepSpeed ZeRO-2 via `scripts/ds_config_zero2.json`

For Stage 1 SFT (Path 1, LoRA):
- `composer_lr = 1e-4` (LoRA standard, scaled vs full FT 1e-5)
- LoRA: r=16, alpha=32, dropout=0.05, targets q/k/v/o/gate/up/down_proj
- LoRA adapter explicitly cast to bf16 (avoid PEFT fp32 default → matmul dtype mismatch)

For Stage 3 RL (matches SkillRL exactly):

| HP | Value |
|---|:-:|
| actor lr | 1e-6 |
| train_batch_size | 16 |
| val_batch_size | 64 |
| group_size (rollout.n) | 8 |
| max_prompt_length | 4096 |
| max_response_length | 512 |
| epochs | 150 |
| kl_loss_coef | 0.01 |
| val_kwargs.temperature | 0.4 |
| val_kwargs.do_sample | True |
| top_k retrieval | 6 |
| test_freq | 5 |

For eval (matches teammate's protocol exactly — validated 123/140 on RL ckpt):
- `do_sample=True, temperature=0.4, top_p=1.0`
- `history_length=10`
- `eos_token_id=tokenizer.eos_token_id`
- 140 episodes valid_in_distribution, max_steps=50

---

## Memory Budget (4 × GH200 120GB)

### Path 1 SFT (DDP Composer + Actor frozen replica)

| Component | Per GPU |
|---|:-:|
| Composer 3B fp32 + grads + Adam | ~48 GB |
| Actor 7B bf16 frozen replica | 14 GB |
| ql + latent_proj | <1 GB |
| Activations (bs=1, seq=4096, bf16, grad ckpt) | ~25 GB |
| **Total** | **~87 GB / 120 GB** ✓ |

### Path 2 SFT (DeepSpeed ZeRO-2, both Composer + Actor trainable)

| Component | Per GPU |
|---|:-:|
| Composer 3B params (bf16 replicated) + grads + Adam (sharded /4) | ~20 GB |
| Actor 7B params (bf16 replicated) + grads + Adam (sharded /4) | ~45 GB |
| ql + latent_proj | <1 GB |
| Activations (bs=1, seq=4096, bf16, grad ckpt on both) | ~40 GB |
| **Total** | **~105 GB / 120 GB** ✓ tight but fits |

### Stage 3 RL (Actor FSDP + Composer frozen replica + vLLM)

| Component | Per GPU |
|---|:-:|
| Actor FSDP shard (params + grads + Adam) | ~17 GB |
| Composer frozen per-rank replica | 6 GB |
| vLLM (gpu_memory_utilization=0.5) | ~60 GB |
| Activations | ~30 GB |
| **Total** | **~113 GB / 120 GB** ⚠ tight but fits |

If tight: drop Composer to rank 0 only + broadcast results, saves 3×6=18GB.

---

## Infrastructure (DeltaAI)

- Partition: `ghx4` (GH200 120GB GPUs)
- Account: **`bfdz-dtai-gh`** (old `bdns-dtai-gh` exhausted)
- Storage:
  - `/work/nvme/bdns/xzhou10/`: **FULL** (read-only in practice — only the SFT parquet lives here)
  - `/work/nvme/bfdz/xzhou10/`: shared with other group members; aim to keep <100GB
  - HF cache + checkpoints → `bfdz`
  - Other project HF cache: `/work/hdd/bfdz/xzhou10/.cache/huggingface/` (~125GB, Qwen3 base models only — MUA-RL-32B/14B/8B + LoopTool-32B/8B caches deleted Jun 6 to free ~181GB; all re-downloadable from HF hub)

## Modal RL Migration (Jun 17) — Stage 3 GRPO on cloud (DeltaAI quota exhausted)

Stage 3 RL moved to **Modal** (profile `glad-lab`, shared w/ luozheng; Volume
`latentskill` mounted at `/vol`). All RL assets staged on the Volume:
`/vol/rl_assets/{actor_k8_expanded_vocab, rl_parquet/{train,test}.parquet, lib_k8.pt, composer_k8_epoch4}`.
Code (SkillRL+verl fork + src) baked from `.modal_stage_rl/` (45MB curated subset).
`modal_rl.py` = the GRPO launcher (`import_check` / `diag` CPU probes + `train_rl` H200:4).

**Image = `vllm/vllm-openai:v0.8.4` + 7 bring-up fixes (each a real, debugged坑)**:
1. `.entrypoint([])` — base image's vllm-api-server ENTRYPOINT crash-loops the container otherwise.
2. NO `add_python` — image's native python3.12 HAS vllm; a fresh add_python would not.
3. `ln -sf $(command -v python3) /usr/local/bin/python` BEFORE pip — image ships only `python3`; Modal's pip step calls `python`.
4. README.md copied into `.modal_stage_rl/SkillRL/` — verl `setup.py` reads it; `pip install -e .` crashes without it.
5. `gigpo/core_gigpo.py` copied into staging — `ray_trainer.py` does `from gigpo import core_gigpo`; was missing from the 45MB subset.
6. **flash_attn** prebuilt wheel `2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312` — image has only vLLM's internal kernels, not the standalone `flash_attn` pkg verl/HF need (flash_attention_2 + use_remove_padding varlen). ABI probed via `diag` (py3.12/torch2.6+cu124/**cxx11abi=False**); wrong ABI = import-ok-but-segfault.
7. `WANDB_MODE=offline` (+ `WANDB_DIR=/vol/rl_assets/wandb`) — no api key needed; `wandb.init` would else raise UsageError. For live curves: create Modal secret `wandb-secret`, add to `train_rl` secrets, flip to online + `WANDB_ENTITY=medagent`.

Also: `ppo_mini_batch_size = train_size*group_size` (was hardcoded 128 → smoke's 64-traj batch asserted).
Secrets: `openai-secret` (X2, wired into `train_rl`; X2 still OFF in v1), `huggingface-secret`.

**Smoke validated end-to-end (Jun 17)**: env init → FSDP load 7B → vLLM TP=4 gen →
latent skill injection (`SKILL_<cat>_NNN_a..h` per-task retrieval) → `<think>/<action>` →
alfworld scoring → robust parser. `val_before_train` runs the FULL val set (≈140 ep ×
≤50 steps → ~slow; `--val-size` is only batch size, not episode count).

### ✅ FIRST RL RESULT (v1, k=8 static latent, X2 off) — RL lifts latent skills

Run `rl_k8_v1_out` (40-epoch target, online wandb `medagent/latentskill/runs/w1qapfrw`).
Train set is TINY (~1 step/epoch → train.parquet ≈16 prompts; val = held-out ~140 ep).
val/success_rate over GRPO steps (test_freq=5):

| step | epoch | val/success_rate |
|:-:|:-:|:-:|
| 0 (baseline=baked actor) | 0 | 48.4% (matches Stage-2 offline ~50% ✓) |
| 5 | 4 | 43.8% |
| 10 | 9 | 39.1% |
| **15** | **14** | **65.6%** 🏆 best |

**Dip-then-jump** (explore/drift → breakthrough): **+17pp over baseline, +27pp over trough**.
→ core thesis confirmed: latent skills ARE RL-optimizable. (vs text-RL ceiling 88.6%.)
Run died at step ~16/epoch 15 (detached app ran ~2.7h then Modal-side terminated; logs
expired). Per-step ~9-14min (val step +~3.5min).

### ⚠️ CHECKPOINT DURABILITY BUG (Jun 22) — fixed

v1's checkpoints were ALL CORRUPT and unresumable: `torch.load` → `RuntimeError:
PytorchStreamReader failed reading zip archive: failed finding central directory`.
Root cause: **Modal volume writes aren't durable until `vol.commit()`**; train_rl only
committed at the very end, so the hard mid-run kill left the big shard files with valid
SIZES (metadata flushed) but truncated zip tails (data not flushed). `global_step_10`
even lost whole shards; `best` looked complete by size but its optim shard was truncated.
**Fix**: `train_rl` now runs a background thread calling `vol.commit()` every 180s →
every saved checkpoint becomes durable on its own. Lesson: ALWAYS periodic-commit Modal
volumes during long jobs. → restarted clean (v2) from the baked actor.

### RL training data (verified Jun 22) — IDENTICAL to SkillRL

- **Game pool (the real "training set", what AlfredTWEnv samples)**: train **6374** trials
  (2435 task configs × ~2-3 trials, 7 task types); valid_seen 251, valid_unseen 255.
  In `SkillRL/agent_system/environments/env_package/alfworld/json_2.1.1/`. is_train →
  `train_eval='train'` loads the full 6374 pool; each `env.reset()` draws the next (seeded).
- **rl_parquet is NOT the dataset** — it's verl's per-step batch driver: train.parquet=16
  rows (=train_batch_size), test.parquet=64 rows (=val_batch_size). Each row = one env slot;
  per step 16 games sampled from the 6374 pool × group_size 8 GRPO rollouts = 128 traj/step.
- **SkillRL's own alfworld config** (`examples/{grpo,gigpo}_trainer/run_alfworld.sh`):
  train_data_size=16, val_data_size=128, group_size=8, total_epochs=150, max_steps=50
  → our train batch (16×8) is IDENTICAL; we use val=64 (in-loop only; headline = 140-ep
  offline eval) and matched total_epochs=150.

### v2 run (Jun 22) — the paper run: clean, durable, SkillRL-matched

`rl_k8_v2_out`, from baked actor, total_epochs=**150** (set from start so cosine LR matches
SkillRL — must NOT do 40-then-extend, the LR horizon would differ), val=64, save_freq=10,
periodic vol.commit active. ~26h on 4×H200, run in legs (resume_mode=auto from latest
COMPLETE checkpoint). wandb online medagent/latentskill.

### ⚠️ LAUNCH BUG (Jun 23) — the real cause of every "~2.7h death": client cancellation

`modal run --detach` launched from a killable wrapper (a backgrounded shell that the
agent harness later cleans up) gets **CANCELED** when that client is SIGTERM'd — logs show
`[modal-client] Received a cancellation signal ... Successfully canceled input`. `--detach`
did NOT protect it (it still streams/blocks), and **retries do NOT re-run a CANCELED call**
(cancellation is deliberate). So the deaths were never preemption — they were the launcher
being killed. **FIX = true server-side fire-and-forget**:
```
modal deploy modal_rl.py                       # persistent app (NOT ephemeral)
PY=$(head -1 $(command -v modal) | sed 's/^#!//')   # modal's python (/usr/bin/python3.12);
$PY -c "import modal; modal.Function.from_name('latentskill-rl','train_rl').spawn(150,16,64,8,5)"
```
spawn() returns immediately, the function runs on the DEPLOYED app independent of any client
→ killing the shell / logging off the cluster can't cancel it. Monitor: `modal app logs
latentskill-rl`. Resume after any stop: re-spawn (resume_mode=auto + durable ckpts continue).
NOTE: `modal run … .spawn()` does NOT work — ephemeral app is torn down on client exit →
the spawned call dies. Must be `modal deploy` first.

### 🏆 v2 RESULT (Jun 23) — latent-skill RL MATCHES/BEATS text-skill RL

Deployed+spawned run survived unattended to step 131/150 (no deaths). In-loop val
(64 ep, test_freq=5) success_rate over GRPO steps:

| step | val | step | val | step | val |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 (baseline) | 48.4% | 50 | 84.4% | 100 | 85.9% |
| 5 | 62.5% | 65 | 84.4% | 110 | 89.1% |
| 25 | 78.1% | 75 | 87.5% | 115 | 87.5% |
| 45 | 82.8% | **85** | **92.2%** 🏆 | 120 | 92.2% |
| | | 90 | 87.5% | 130 | 87.5% |

**baseline 48.4% → best ckpt 92.2% (step 85); late phase stable ~85-92%.** This
**reaches/exceeds the text-RL ceiling (88.6%)** at ~half the token cost → core thesis
proven: latent skills are RL-optimizable to text-skill-RL parity. CAVEAT: this is the
in-loop val (64 ep); the headline number must be the standard **140-ep offline eval**
(`eval_latent_alfworld.py`) on `rl_k8_v2_out/best` — TODO. Val noise ±, curve bounces
79.7-92.2% late, so report best-ckpt + a 3-seed offline eval, not the single peak.

## Critical Version Pins

| Package | Version | Why |
|---------|---------|-----|
| vLLM | 0.19.0 | system-provided aarch64 wheel |
| transformers | 4.57.1 | needs `ALLOWED_LAYER_TYPES` for vLLM 0.19 |
| **huggingface-hub** | **0.35.3** | **transformers 4.57 hard-requires <1.0; do NOT upgrade to 1.x** |
| tokenizers | 0.22.0 | required by transformers 4.57.1 |
| accelerate | 1.11.0 | for FSDP/DeepSpeed launching |
| peft | 0.17.1 | LoRA support |
| deepspeed | 0.18.9 | for ZeRO-2 (Path 2) |

---

## Modified verl Files (used in Stage 3)

- `verl/utils/vllm_utils.py` — try/except for LoRA imports
- `verl/workers/sharding_manager/fsdp_vllm.py` — Composer dyn-skill sync to vLLM (X2)
- `verl/workers/critic/dp_critic.py` — tensordict `.keys()` fix
- `verl/trainer/ppo/ray_trainer.py` — best-ckpt tracking + X2 dyn skill encoding hook
- `verl/workers/fsdp_workers.py` — Composer init + `write_dyn_skill_embeds` worker method
- `agent_system/memory/skill_updater.py` — OpenAI direct (o3-mini, reasoning_effort=medium, max_completion_tokens=4096)

Note: `verl/workers/actor/dp_actor.py` per-micro-batch Composer re-encode is **NOT
needed** in current design — Composer is frozen during RL, latents are static
(except for new dynamically-added skills via X2 path).

---

## Known Issues / Watchouts

1. **`/work/nvme/bdns/xzhou10` was PURGED entirely (discovered Jun 5)** — SFT parquet re-downloaded to bfdz HF cache (same snapshot bd3996c4); Jianwen--Alfworld-7B-SFT snapshot also lost (re-downloadable from HF hub if needed). Legacy v2 ckpts (`latent_skills_token_v2` etc) are gone for good — legacy scripts referencing them are dead.
2. **`/work/nvme/bfdz` shared with other users** — periodically free unused ckpts (k=8 ablation deleted, eval JSONs preserved)
3. **wandb entity**: `medagent` (personal entity disabled). Project: `latentskill`
4. **Ray OOM monitor**: disabled via `RAY_memory_monitor_refresh_ms=0`
5. **Dataset cache**: verl's `rl_dataset.py` does `shutil.disk_usage` on cache dir; bdns reports 0 → must use bfdz
6. **FSDP + embed_tokens direct access** (lesson from old joint-SFT): NEVER call `model.get_input_embeddings()(ids)` outside `model.forward(...)` under FSDP — embed.weight is 1-D flat_param when accessed directly. **Use ZeRO-2 (Path 2) or DDP (Path 1) instead** to keep params replicated 2-D.
7. **`tie_word_embeddings` trap**: Qwen2.5-3B has tie=True. Freezing lm_head silently freezes embed_tokens too. **Always untie before freeze**: `composer.config.tie_word_embeddings = False; composer.lm_head.weight = nn.Parameter(composer.lm_head.weight.data.clone())`.
8. **bf16 update-round-to-zero**: At weight magnitude ~74, bf16 precision is ~0.585 → updates of magnitude 1e-7 (lr×grad for 1e-5 × 1e-2) round to zero. Always use **fp32 master copies** for trainable Composer body (or use mixed_precision via DeepSpeed/FSDP which handles this).
9. **PEFT LoRA dtype mismatch (two places)**: (a) PEFT defaults adapters to fp32 but base model is bf16 → matmul errors. Cast LoRA params to bf16 explicitly: `for n, p in model.named_parameters(): if 'lora_' in n: p.data = p.data.to(torch.bfloat16)`. (b) **`latent_proj` boundary**: `latent_proj` is fp32 by design (precision for trainable param) but under LoRA the composer body is bf16 → its hidden output is bf16 → `F.linear(bf16, fp32)` crashes with "expected mat1 and mat2 to have the same dtype". Fix: upcast input to fp32 at the boundary: `latent_proj(latent_reprs_composer.float())`. Lossless and keeps the fp32 precision design.
10. **DeepSpeed ZeRO-2 single-prepare requirement**: model + optimizer must be passed to `accelerator.prepare()` in ONE call (separate calls give "zero stage 2 requires an optimizer" assertion). Use a `JointWrapper(nn.Module)` that holds both Composer + Actor + ql + proj as submodules → wraps as one DeepSpeed engine.
11. **Encoder-output collapse under frozen-Actor SFT** (Path 1): with Composer trainable but Actor frozen, optimal solution is "constant latent that Actor learns to ignore". Cross-skill cossim → 1.0. k>2 makes it worse. **Path 2 should resist this** (Actor's gradient provides discriminative signal). LatentMem uses LoRA + joint training to avoid.
12. **Disk quota**: Path 2 ckpts ≈ 6 GB Composer + 14 GB Actor + 4 epoch ql files ≈ 25 GB per run. Keep total project under 100 GB on bfdz nvme.

## OpenAI Key (for Stage 3 X2 dynamic skill bank)

`OPENAI_API_KEY` set in `scripts/setup_env.sh` (gitignored). Backend: o3-mini,
max_completion_tokens 4096, reasoning_effort="medium". Used by
`SkillRL/agent_system/memory/skill_updater.py` to analyze failed trajectories
and propose new skill text → Composer encodes → vocab embedding row.

## Paper Story

**Problem**: Agent RL with text skills (SkillRL) has three limits:
1. High token cost (~1600 tokens per step → ~700 for skill section)
2. Non-differentiable (RL can't optimize skill content directly via reward)
3. Skill bank not modular (skills baked into model weights after RL)

**Contribution**:
1. **Latent skills via small Composer**: a 3B Composer SFT-trained to compress
   each skill into K=2 latent vectors. ~750 tokens of text skills → ~88 latent
   tokens (8.5×) at the cost of one extra small forward.
2. **Per-step token efficiency**: ~37K tokens/episode vs 76K text-skill SFT, **-51%**.
3. **Composer-training value (TRAINED vs UNTRAINED ablation)**: Path 1 shows
   trained Composer 38.6% vs untrained 23.6% (+15%) → SFT training the Composer
   carries discriminative skill information beyond the prompt scaffold.
4. **End-to-end RL on latent vocab rows** (Stage 3 pending): latent skills are
   differentiable — the SKILL_N vocab embedding rows update via GRPO reward.
5. **Dynamic latent skill bank (X2)**: failed trajectories during RL → o3-mini
   proposes new skill text → Composer encodes on the fly → new SKILL token
   added to vLLM vocab. Skill set grows during RL, all in latent form.

**Compare against**:
- SkillRL (text skills + full-model RL, 89.9% on ALFWorld; we measured 123/140 sampled)
- SkillZero (skill internalization via curriculum, different approach)
- LatentMem / TokMem (similar latent infrastructure but SFT only, no RL, no dynamic bank)

## Related Work References

- SkillRL paper: `related work/SkillRL.pdf`
- LatentMem repo: `LatentMem/` (cloned locally)
- TokMem repo: `TokMem/` (cloned locally)
