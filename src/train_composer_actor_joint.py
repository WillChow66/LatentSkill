"""Path 2 — Joint Composer + Actor SFT (full parameter training, DeepSpeed ZeRO-2).

Two SEPARATE models (NOT one model with multiple roles):
  Composer = Qwen2.5-3B-Instruct  (full FT, ZeRO-2 sharded)
  Actor    = Qwen2.5-7B-Instruct  (full FT, ZeRO-2 sharded)

Both trainable. Hidden-dim projection (LLaVA-style) bridges Composer (D=2048)
to Actor (D=3584).

Why ZeRO-2 not FSDP:
  - We must call composer.embed(skill_ids) and actor.embed(before_ids/after_ids)
    OUTSIDE model.forward to compose inputs_embeds (latent skill paradigm).
  - FSDP shards params → embed.weight is 1-D flat_param outside forward → crash.
  - ZeRO-2 keeps params replicated per GPU (only grads + optimizer states sharded)
    → embed.weight always 2-D, direct access works.
  - Mathematically equivalent to DDP full FT, just with sharded optimizer state
    (LatentMem uses this exact stack via accelerate zero2.yaml).

Two run modes:
  default                 TRAINED: both Composer and Actor trainable
  --freeze-composer       UNTRAINED: Composer frozen at random init,
                          only Actor + query_latents + latent_proj train
                          (ablation control to test if Composer training matters
                          beyond what Actor adaptation alone achieves)

Launch (must include --use_deepspeed):
  accelerate launch --num_processes 4 --use_deepspeed \
      --deepspeed_config_file scripts/ds_config_zero2.json \
      --mixed_precision bf16 \
      -m src.train_composer_actor_joint \
        --composer-path Qwen/Qwen2.5-3B-Instruct \
        --actor-path Qwen/Qwen2.5-7B-Instruct \
        --sft-data /work/.../alfworld/train-00000-of-00001.parquet \
        --output-dir /work/.../joint_path2 \
        --epochs 4 --latents-per-skill 8
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

SKILLS_START_MARKER = "## Retrieved Relevant Experience"
SKILLS_END_MARKER = "## Current Progress"
PLACEHOLDER = "[LATENT_SKILLS]"
ASSISTANT_MARKER = "<|im_start|>assistant\n"


# ----------------------------------------------------------------------- #
# Data parsing (same as Path 1)
# ----------------------------------------------------------------------- #

def parse_skills_from_content(content: str):
    start = content.find(SKILLS_START_MARKER)
    end = content.find(SKILLS_END_MARKER)
    if start == -1 or end == -1:
        return content, [], ""
    before = content[:start + len(SKILLS_START_MARKER)]
    section = content[start + len(SKILLS_START_MARKER):end]
    after = content[end:]
    skills = [ln.strip()[2:] for ln in section.strip().split("\n") if ln.strip().startswith("- **")]
    return before, skills, after


class SkillRLSFTDataset(Dataset):
    def __init__(self, parquet_path, max_samples=None):
        df = pd.read_parquet(parquet_path)
        if max_samples:
            df = df.head(max_samples)
        self.samples = []
        for _, row in df.iterrows():
            ins = str(row.get("instruction", row.get("prompt", "")))
            out = str(row.get("output", row.get("answer", "")))
            if not ins or not out:
                continue
            if SKILLS_START_MARKER not in ins:
                continue
            self.samples.append((ins, out))
        logger.info(f"Loaded {len(self.samples)} samples from {parquet_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ins, out = self.samples[idx]
        before, skills, after = parse_skills_from_content(ins)
        return {"output": out, "before": before, "skill_texts": skills, "after": after}


def make_batch(sample, composer_tok, actor_tok, max_length, device):
    modified = sample["before"] + "\n" + PLACEHOLDER + "\n" + sample["after"]
    messages = [
        {"role": "user", "content": modified},
        {"role": "assistant", "content": sample["output"]},
    ]
    full_text = actor_tok.apply_chat_template(messages, tokenize=False)
    pos = full_text.find(PLACEHOLDER)
    if pos < 0:
        return None
    before_text = full_text[:pos]
    after_text = full_text[pos + len(PLACEHOLDER):]

    before_ids = actor_tok(before_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    after_ids = actor_tok(after_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    skill_ids_list = []
    for sk_text in sample["skill_texts"]:
        sid = composer_tok(sk_text, add_special_tokens=False, return_tensors="pt",
                           truncation=True, max_length=128)["input_ids"].to(device)
        skill_ids_list.append(sid)
    if not skill_ids_list:
        return None

    pre_resp_text = ""
    if ASSISTANT_MARKER in after_text:
        pre_resp_text = after_text[:after_text.find(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)]
    n_after_prompt = (
        len(actor_tok(pre_resp_text, add_special_tokens=False)["input_ids"])
        if pre_resp_text else 0
    )

    return {
        "before_ids": before_ids,
        "skill_input_ids_list": skill_ids_list,
        "after_ids": after_ids,
        "n_after_prompt_ids": n_after_prompt,
    }


# ----------------------------------------------------------------------- #
# Forward functions (Composer encode + Actor forward+loss)
# ----------------------------------------------------------------------- #

def composer_encode_skills(composer_inner, query_latents, k, skill_input_ids_list,
                           pad_token_id, hidden_size):
    """Encode each skill into K latent vectors via Composer's Qwen2Model.

    composer_inner: unwrapped Qwen2ForCausalLM (NOT DeepSpeed wrapped). embed.weight
    is full 2-D under ZeRO-2 (params not sharded), so direct access is safe.
    """
    device = query_latents.device
    n_skills = len(skill_input_ids_list)
    if n_skills == 0:
        return torch.zeros(0, hidden_size, device=device, dtype=query_latents.dtype)

    embed = composer_inner.get_input_embeddings()
    max_text_len = max(ids.shape[1] for ids in skill_input_ids_list)
    padded_ids = torch.full((n_skills, max_text_len), pad_token_id, dtype=torch.long, device=device)
    attn_mask_text = torch.zeros((n_skills, max_text_len), dtype=torch.long, device=device)
    for i, ids in enumerate(skill_input_ids_list):
        L = ids.shape[1]
        padded_ids[i, :L] = ids[0]
        attn_mask_text[i, :L] = 1

    text_emb = embed(padded_ids)
    queries = query_latents.to(text_emb.dtype).unsqueeze(0).expand(n_skills, -1, -1)
    combined = torch.cat([text_emb, queries], dim=1)
    attn_mask = torch.cat(
        [attn_mask_text, torch.ones((n_skills, k), dtype=torch.long, device=device)],
        dim=1,
    )

    # Use composer.model (Qwen2Model, no lm_head) — saves a small amount of compute
    out = composer_inner.model(
        inputs_embeds=combined, attention_mask=attn_mask,
        use_cache=False, return_dict=True,
    )
    last_hidden = out.last_hidden_state  # (n_skills, max_text_len + k, D_composer)
    return last_hidden[:, -k:, :].reshape(n_skills * k, -1)


def actor_forward_loss(actor_inner, before_ids, after_ids,
                       latent_reprs_actor_dim, n_after_prompt):
    """Actor forward with [before_emb | latent_reprs | after_emb] inputs_embeds.

    actor_inner: unwrapped Qwen2ForCausalLM. embed.weight is full 2-D under
    ZeRO-2, so direct access works.
    Loss is CE on response tokens (after the assistant marker).
    """
    device = before_ids.device
    actor_embed = actor_inner.get_input_embeddings()
    before_emb = actor_embed(before_ids)
    after_emb = actor_embed(after_ids)
    skill_emb = latent_reprs_actor_dim.unsqueeze(0).to(before_emb.dtype)
    full_emb = torch.cat([before_emb, skill_emb, after_emb], dim=1)
    attn_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=device)

    out = actor_inner(inputs_embeds=full_emb, attention_mask=attn_mask, use_cache=False)
    logits = out.logits

    n_lat = latent_reprs_actor_dim.shape[0]
    before_len = before_ids.shape[1]
    labels = torch.full(full_emb.shape[:2], -100, dtype=torch.long, device=device)
    resp_start = before_len + n_lat + n_after_prompt
    if resp_start < full_emb.shape[1]:
        resp_token_ids = after_ids[0, n_after_prompt:]
        n_resp = min(resp_token_ids.shape[0], full_emb.shape[1] - resp_start)
        labels[0, resp_start:resp_start + n_resp] = resp_token_ids[:n_resp]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ----------------------------------------------------------------------- #
# Save (params replicated under ZeRO-2 → simple unwrap+save_pretrained)
# ----------------------------------------------------------------------- #

def save_checkpoint(accelerator, composer, actor, query_latents, latent_proj,
                    latents_per_skill, composer_tok, actor_tok, output_dir,
                    global_step, ql_only=False, epoch=None,
                    composer_qproj_ref=None, actor_qproj_ref=None,
                    freeze_composer=False):
    accelerator.wait_for_everyone()
    out = Path(output_dir)

    # query_latents + latent_proj (standalone, replicated)
    if accelerator.is_main_process:
        full_ql = query_latents.detach().float().cpu().clone()
        save_dict = {
            "query_latents": full_ql, "step": global_step, "epoch": epoch,
            "latents_per_skill": latents_per_skill,
        }
        if isinstance(latent_proj, nn.Linear):
            save_dict["latent_proj_state"] = latent_proj.state_dict()
            save_dict["proj_in_dim"] = latent_proj.in_features
            save_dict["proj_out_dim"] = latent_proj.out_features
        torch.save(save_dict, out / f"query_latents_step{global_step}.pt")
        logger.info(f"Saved query_latents (norm={full_ql.norm().item():.3f}) "
                    f"+ latent_proj")

    if ql_only:
        accelerator.wait_for_everyone()
        return

    # Composer save (only if trainable)
    composer_inner = accelerator.unwrap_model(composer)
    actor_inner = accelerator.unwrap_model(actor)
    suffix = f"epoch{epoch}" if epoch else f"step{global_step}"

    if not freeze_composer:
        composer_dir = out / f"composer_{suffix}"
        if accelerator.is_main_process:
            full_state = composer_inner.state_dict()
            save_state = {k: v.detach().cpu().to(torch.bfloat16) for k, v in full_state.items()}
            sample = "model.layers.0.self_attn.q_proj.weight"
            sample_norm = save_state[sample].float().norm().item()
            diff = abs(sample_norm - composer_qproj_ref) if composer_qproj_ref else -1.0
            logger.info(f"Composer save: q_proj.norm={sample_norm:.6f} "
                        f"ref={composer_qproj_ref:.6f} abs_diff={diff:.6e}")
            if composer_qproj_ref is not None and diff < 1e-6:
                logger.error("!!! Composer q_proj UNCHANGED — check trainability!")
            composer_inner.save_pretrained(composer_dir, state_dict=save_state,
                                           safe_serialization=True, max_shard_size="10GB")
            composer_tok.save_pretrained(composer_dir)
            logger.info(f"Saved Composer ckpt → {composer_dir}")

    # Actor save (always trainable in Path 2)
    actor_dir = out / f"actor_{suffix}"
    if accelerator.is_main_process:
        full_state = actor_inner.state_dict()
        save_state = {k: v.detach().cpu().to(torch.bfloat16) for k, v in full_state.items()}
        sample = "model.layers.0.self_attn.q_proj.weight"
        sample_norm = save_state[sample].float().norm().item()
        diff = abs(sample_norm - actor_qproj_ref) if actor_qproj_ref else -1.0
        logger.info(f"Actor save: q_proj.norm={sample_norm:.6f} "
                    f"ref={actor_qproj_ref:.6f} abs_diff={diff:.6e} (>1e-3 expected)")
        if actor_qproj_ref is not None and diff < 1e-6:
            logger.error("!!! Actor q_proj UNCHANGED — training did not propagate!")
        actor_inner.save_pretrained(actor_dir, state_dict=save_state,
                                    safe_serialization=True, max_shard_size="10GB")
        actor_tok.save_pretrained(actor_dir)
        logger.info(f"Saved Actor ckpt → {actor_dir}")
    accelerator.wait_for_everyone()


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composer-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--actor-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--composer-lr", type=float, default=1e-5)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--latent-lr", type=float, default=5e-3)
    parser.add_argument("--proj-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--latents-per-skill", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--freeze-composer", action="store_true",
                        help="UNTRAINED ablation: Composer frozen at random init, "
                             "only Actor + query_latents + latent_proj train")
    parser.add_argument("--wandb-project", type=str, default="latentskill")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Args: {vars(args)}")
        logger.info(f"World size: {accelerator.num_processes}")
        mode = "UNTRAINED (composer frozen)" if args.freeze_composer else "TRAINED (joint)"
        logger.info(f"Mode: {mode}")

        if not args.no_wandb:
            try:
                import wandb
                run_name = args.wandb_run_name or Path(args.output_dir).name
                wandb.init(project=args.wandb_project, name=run_name,
                           config=vars(args), resume="allow")
                logger.info(f"wandb: project={args.wandb_project} run={run_name}")
            except Exception as e:
                logger.warning(f"wandb init failed ({e}); continuing without")
                args.no_wandb = True

    # Tokenizers
    composer_tok = AutoTokenizer.from_pretrained(args.composer_path, trust_remote_code=True)
    actor_tok = AutoTokenizer.from_pretrained(args.actor_path, trust_remote_code=True)
    if composer_tok.pad_token is None:
        composer_tok.pad_token = composer_tok.eos_token
    if actor_tok.pad_token is None:
        actor_tok.pad_token = actor_tok.eos_token

    # Composer (Qwen2.5-3B). Untie tied embeddings so freeze of unused parts
    # doesn't accidentally freeze used parts later.
    composer = AutoModelForCausalLM.from_pretrained(
        args.composer_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if composer.config.tie_word_embeddings:
        composer.config.tie_word_embeddings = False
        composer.lm_head.weight = nn.Parameter(composer.lm_head.weight.data.clone())
    # Composer.lm_head NOT used (we take hidden states), freeze to avoid DDP unused-param warnings
    for p in composer.lm_head.parameters():
        p.requires_grad = False
    composer.gradient_checkpointing_enable()

    if args.freeze_composer:
        # UNTRAINED ablation: freeze the entire Composer SYSTEM (body + ql + proj)
        # at random/pretrained init. Only Actor adapts. This is the apples-to-apples
        # control to Path 1 UNTRAINED (where Composer + ql + proj were all random/frozen)
        # — answers "does Composer system training matter beyond Actor adaptation?"
        for p in composer.parameters():
            p.requires_grad = False
        composer.eval()
        # query_latents and latent_proj also frozen (set after they're created below)

    # Snapshot Composer ref BEFORE any wrap
    composer_qproj_ref = composer.model.layers[0].self_attn.q_proj.weight.detach().float().norm().item()

    # Actor (Qwen2.5-7B), full FT, trainable in both TRAINED and UNTRAINED
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if actor.config.tie_word_embeddings:
        actor.config.tie_word_embeddings = False
        actor.lm_head.weight = nn.Parameter(actor.lm_head.weight.data.clone())
    actor.gradient_checkpointing_enable()
    actor_qproj_ref = actor.model.layers[0].self_attn.q_proj.weight.detach().float().norm().item()

    # Hidden-dim projection + query_latents will live INSIDE JointWrapper below
    # (not standalone). This way DeepSpeed wraps everything once via prepare().
    D_composer = composer.config.hidden_size
    D_actor = actor.config.hidden_size

    # Dataset
    dataset = SkillRLSFTDataset(args.sft_data, max_samples=args.max_samples)
    rank = accelerator.process_index
    world = accelerator.num_processes

    if accelerator.is_main_process:
        logger.info("Pre-validating dataset...")
        valid_indices = []
        for i in range(len(dataset)):
            test = make_batch(dataset[i], composer_tok, actor_tok, args.max_length, "cpu")
            if test is not None:
                valid_indices.append(i)
        n_drop = len(valid_indices) % world
        if n_drop:
            valid_indices = valid_indices[:-n_drop]
        valid_indices_t = torch.tensor(valid_indices, dtype=torch.long)
        logger.info(f"Valid samples: {len(valid_indices)} (dropped {n_drop})")
    else:
        valid_indices_t = torch.tensor([], dtype=torch.long)

    accelerator.wait_for_everyone()
    obj_list = [valid_indices_t if accelerator.is_main_process else None]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    valid_indices = obj_list[0].tolist()
    rank_indices = valid_indices[rank::world]
    if accelerator.is_main_process:
        logger.info(f"Per-rank: {len(rank_indices)} samples × {world} ranks = {len(valid_indices)} total")

    # Wrap Composer + Actor + query_latents + latent_proj in ONE nn.Module so
    # DeepSpeed sees a single model. Required for SINGLE accelerator.prepare()
    # call (see "zero stage 2 requires an optimizer" error if split).
    #
    # Trainable matrix:
    #             | composer body | ql + latent_proj | actor body
    #   TRAINED   |    ✓ train    |    ✓ train       |  ✓ train
    #   UNTRAINED |    ✗ frozen   |    ✗ frozen      |  ✓ train  (only actor adapts)
    class JointWrapper(nn.Module):
        def __init__(self, composer, actor, latents_per_skill, D_composer, D_actor,
                     freeze_composer):
            super().__init__()
            self.composer = composer
            self.actor = actor
            # query_latents — random init. Frozen iff freeze_composer (UNTRAINED)
            ql_init = (torch.randn(latents_per_skill, D_composer) * 0.02).to(torch.float32)
            self.query_latents = nn.Parameter(ql_init, requires_grad=not freeze_composer)
            # latent_proj — random init. Frozen iff freeze_composer (UNTRAINED)
            if D_composer != D_actor:
                self.latent_proj = nn.Linear(D_composer, D_actor, bias=False).to(dtype=torch.float32)
                with torch.no_grad():
                    nn.init.normal_(self.latent_proj.weight, std=0.02)
                if freeze_composer:
                    for p in self.latent_proj.parameters():
                        p.requires_grad = False
            else:
                self.latent_proj = nn.Identity()
        def forward(self, *args, **kwargs):
            raise RuntimeError("JointWrapper.forward is not used")

    joint = JointWrapper(
        composer, actor, args.latents_per_skill, D_composer, D_actor,
        args.freeze_composer,
    )
    if accelerator.is_main_process:
        if args.freeze_composer:
            logger.info("UNTRAINED variant: Composer body + query_latents + latent_proj all frozen, only Actor trains")
        else:
            logger.info("TRAINED variant: Composer + query_latents + latent_proj + Actor all trainable")

    # Build optimizer with up to 4 LR groups
    composer_params = [p for _, p in joint.composer.named_parameters() if p.requires_grad]
    actor_params = [p for _, p in joint.actor.named_parameters() if p.requires_grad]
    proj_params = [p for p in joint.latent_proj.parameters() if p.requires_grad] \
                  if isinstance(joint.latent_proj, nn.Linear) else []

    optimizer_groups = []
    if composer_params:
        optimizer_groups.append(
            {"params": composer_params, "lr": args.composer_lr, "weight_decay": args.weight_decay}
        )
    if actor_params:
        optimizer_groups.append(
            {"params": actor_params, "lr": args.actor_lr, "weight_decay": args.weight_decay}
        )
    if joint.query_latents.requires_grad:
        optimizer_groups.append(
            {"params": [joint.query_latents], "lr": args.latent_lr, "weight_decay": 0.0}
        )
    if proj_params:
        optimizer_groups.append(
            {"params": proj_params, "lr": args.proj_lr, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(args.adam_beta1, args.adam_beta2))

    total_steps = len(rank_indices) * args.epochs * accelerator.num_processes
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # SINGLE prepare call — required for DeepSpeed plugin.
    joint, optimizer, scheduler = accelerator.prepare(joint, optimizer, scheduler)
    joint_inner = accelerator.unwrap_model(joint)
    composer_inner = joint_inner.composer
    actor_inner = joint_inner.actor
    # Aliases for code that uses these names downstream
    query_latents = joint_inner.query_latents
    latent_proj = joint_inner.latent_proj

    if accelerator.is_main_process:
        logger.info(f"Trainable: composer={sum(p.numel() for p in composer_params):,} "
                    f"actor={sum(p.numel() for p in actor_params):,} "
                    f"+ ql={query_latents.numel():,} + proj")
        logger.info(f"Composer ref q_proj.norm = {composer_qproj_ref:.6f}")
        logger.info(f"Actor    ref q_proj.norm = {actor_qproj_ref:.6f}")
        logger.info(f"Scheduler: total_steps={total_steps} warmup={warmup_steps}")

    pad_id = composer_tok.pad_token_id or 0

    global_step = 0
    for epoch in range(args.epochs):
        composer_inner.train() if not args.freeze_composer else composer_inner.eval()
        actor_inner.train()
        ep_loss = 0.0
        n_ok = 0

        for dataset_idx in rank_indices:
            sample = dataset[dataset_idx]
            batch = make_batch(sample, composer_tok, actor_tok, args.max_length,
                               accelerator.device)
            if batch is None:
                continue

            with accelerator.accumulate(joint):
                # 1) Encode skills (Composer forward; gradients depend on freeze flag)
                if args.freeze_composer:
                    with torch.no_grad():
                        latent_reprs_composer = composer_encode_skills(
                            composer_inner, query_latents, args.latents_per_skill,
                            batch["skill_input_ids_list"], pad_id, D_composer,
                        )
                    # Re-attach grad to query_latents path so it can train via proj
                    # (query_latents was used inside composer.forward; with no_grad,
                    # the graph is detached. We'll let proj+actor still get grad via
                    # latent_reprs.requires_grad_().)
                    latent_reprs_composer = latent_reprs_composer.detach()
                else:
                    latent_reprs_composer = composer_encode_skills(
                        composer_inner, query_latents, args.latents_per_skill,
                        batch["skill_input_ids_list"], pad_id, D_composer,
                    )

                # 2) Project to Actor hidden space
                latent_reprs_actor = latent_proj(latent_reprs_composer)

                # 3) Actor forward + CE loss
                loss = actor_forward_loss(
                    actor_inner,
                    batch["before_ids"], batch["after_ids"],
                    latent_reprs_actor, batch["n_after_prompt_ids"],
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip joint (composer + actor) trainable params via accelerator
                    accelerator.clip_grad_norm_(
                        [p for p in joint.parameters() if p.requires_grad], 1.0,
                    )
                    if query_latents.grad is not None and query_latents.requires_grad:
                        torch.nn.utils.clip_grad_norm_([query_latents], 1.0)
                    if isinstance(latent_proj, nn.Linear) and any(p.requires_grad for p in latent_proj.parameters()):
                        torch.nn.utils.clip_grad_norm_(latent_proj.parameters(), 1.0)

                do_log = (global_step + 1) % args.log_every == 0
                if do_log:
                    ql_norm = query_latents.detach().float().norm().item()
                    aqp = actor_inner.model.layers[0].self_attn.q_proj.weight
                    actor_qproj_now = aqp.detach().float().norm().item()
                    cqp = composer_inner.model.layers[0].self_attn.q_proj.weight
                    composer_qproj_now = cqp.detach().float().norm().item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            ep_loss += loss.item()
            n_ok += 1
            global_step += 1

            if do_log and accelerator.is_main_process:
                lrs = [g["lr"] for g in optimizer.param_groups]
                logger.info(
                    f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} ql_norm={ql_norm:.3f} "
                    f"composer_qproj={composer_qproj_now:.4f} actor_qproj={actor_qproj_now:.4f} "
                    f"lrs={['%.2e' % lr for lr in lrs]}"
                )
                if not args.no_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/ql_norm": ql_norm,
                        "train/composer_qproj_norm": composer_qproj_now,
                        "train/actor_qproj_norm": actor_qproj_now,
                        "train/lr_0": lrs[0] if lrs else 0,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

        avg = ep_loss / max(n_ok, 1)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} avg_loss={avg:.4f} ({n_ok} per rank)")
            if not args.no_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg, "epoch/epoch": epoch + 1}, step=global_step)

        is_last = (epoch + 1 == args.epochs)
        save_checkpoint(
            accelerator, composer, actor, query_latents, latent_proj,
            args.latents_per_skill, composer_tok, actor_tok, args.output_dir,
            global_step, ql_only=not is_last, epoch=epoch + 1,
            composer_qproj_ref=composer_qproj_ref, actor_qproj_ref=actor_qproj_ref,
            freeze_composer=args.freeze_composer,
        )


if __name__ == "__main__":
    main()
