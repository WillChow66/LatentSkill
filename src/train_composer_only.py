"""Stage 1 — Composer SFT (Composer trained, Actor frozen).

Two models:
  Composer = Qwen2.5-3B-Instruct  (FSDP-wrapped, trainable)
  Actor    = Qwen2.5-7B-Instruct  (per-rank full replica, frozen, eval mode)

Per training sample:
  1. Parse SkillRL parquet row → before_text, [skill_text_1..N], after_text, response.
  2. For each skill, run Composer:
       composer.embed(skill_ids ⊕ query_latent_placeholders)
         → composer.transformer
         → take last K hidden states = latent_repr (with grad)
  3. Build Actor inputs_embeds:
       [actor.embed(before_ids) | latent_reprs (k×N) | actor.embed(after_ids)]
  4. actor.forward(inputs_embeds=...) → logits  (Actor frozen, eval, no_grad)
  5. CE loss on response tokens

Backward:
  loss → through Actor (frozen, no param updates, but grad still flows back through
         inputs_embeds → latent_reprs)
       → through Composer encoder
       → query_latents and Composer encoder weights ← UPDATE

Why this design avoids the FSDP hook hell:
  - Actor is frozen + per-rank-replica → never sees gradient routing or FSDP
    edge cases. We never call actor.embed(ids) outside actor.forward EXCEPT
    when the actor is frozen and not FSDP-wrapped, where the embed.weight is
    a normal 2-D tensor.
  - Composer is FSDP-wrapped and only ever called as composer(input_ids=...) →
    standard FSDP forward path, no embed hooks, no edge cases.
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
# Data parsing
# ----------------------------------------------------------------------- #

def parse_skills_from_content(content: str):
    """SkillRL prompt has a section between ## Retrieved Relevant Experience and
    ## Current Progress; each retrieved skill starts with '- **'. Returns
    (before_text_inclusive, [skill_lines], after_text_inclusive)."""
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
    """Tokenize one sample for both Composer (per-skill) and Actor (full prompt
    with [LATENT_SKILLS] placeholder).

    Returns dict or None if malformed.
    """
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

    # Tokenize each skill with Composer's tokenizer (could differ from Actor's,
    # but Qwen2.5-3B and Qwen2.5-7B share tokenizer in practice — verified).
    skill_ids_list = []
    for sk_text in sample["skill_texts"]:
        sid = composer_tok(sk_text, add_special_tokens=False, return_tensors="pt",
                           truncation=True, max_length=128)["input_ids"].to(device)
        skill_ids_list.append(sid)
    if not skill_ids_list:
        return None

    # Compute n_prompt_tokens (where the assistant response begins)
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
# Composer: encode skills via standard composer.forward (no hooks)
# ----------------------------------------------------------------------- #

def _get_qwen2model(composer_inner):
    """Resolve the inner Qwen2Model regardless of PEFT wrapping.
    PeftModelForCausalLM.model = LoraModel, .base_model = LoraModel,
    .get_base_model() = Qwen2ForCausalLM. Plain HF: composer_inner.model = Qwen2Model.
    """
    try:
        from peft import PeftModel
        if isinstance(composer_inner, PeftModel):
            return composer_inner.get_base_model().model  # Qwen2ForCausalLM.model = Qwen2Model
    except ImportError:
        pass
    return composer_inner.model


def _get_qwen2_qproj(composer_inner, layer_idx=0):
    """q_proj.weight at given layer, regardless of PEFT (returns the base Linear's weight,
    NOT the LoRA-modulated version — for monitoring training signal only)."""
    return _get_qwen2model(composer_inner).layers[layer_idx].self_attn.q_proj.weight


def composer_encode_skills(composer_inner, query_latents, k, skill_input_ids_list,
                           pad_token_id, hidden_size):
    """Encode each skill into K latent vectors via Composer's inner Qwen2 model.

    DESIGN: Composer is DDP-wrapped, so embed.weight is a normal 2-D tensor on
    every rank. We use the UNWRAPPED inner module. For PEFT-wrapped composer,
    we use the full forward via outer wrapper (output_hidden_states=True) so
    LoRA adapters fire correctly.
    """
    device = query_latents.device
    n_skills = len(skill_input_ids_list)
    if n_skills == 0:
        return torch.zeros(0, hidden_size, device=device, dtype=query_latents.dtype)

    embed = composer_inner.get_input_embeddings()  # works under both PEFT and plain HF

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

    # Use the inner Qwen2Model directly (works for both plain HF and PEFT — PEFT
    # patches the linear layers in-place, so calling Qwen2Model.forward still
    # invokes the LoRA-wrapped projections).
    qwen2_model = _get_qwen2model(composer_inner)
    out = qwen2_model(
        inputs_embeds=combined, attention_mask=attn_mask,
        use_cache=False, return_dict=True,
    )
    last_hidden = out.last_hidden_state  # (n_skills, max_text_len + k, D)
    return last_hidden[:, -k:, :].reshape(n_skills * k, -1)


# ----------------------------------------------------------------------- #
# Actor: forward with inputs_embeds=spliced (frozen, no hook needed)
# ----------------------------------------------------------------------- #

def actor_forward_loss(actor, actor_embed_layer, before_ids, after_ids,
                       latent_reprs, n_after_prompt):
    """Build inputs_embeds = [actor.embed(before) | latent_reprs | actor.embed(after)]
    and run actor.forward to get logits. CE loss on response tokens only.

    Actor is frozen (eval mode, no_grad on actor params). However, actor.forward
    itself must compute grads on inputs_embeds because latent_reprs has grad
    (we want it to flow back to Composer + query_latents + projection).

    Actor here is NOT FSDP-wrapped — it's a per-rank full replica. So
    actor.embed(ids) is safe to call directly (embed.weight is normal 2-D).

    NOTE: latent_reprs MUST already be in actor's hidden_size (apply projection
    before calling this function).
    """
    device = before_ids.device
    before_emb = actor_embed_layer(before_ids)  # (1, B, D_actor)
    after_emb = actor_embed_layer(after_ids)
    skill_emb = latent_reprs.unsqueeze(0).to(before_emb.dtype)  # (1, n_lat, D_actor)
    full_emb = torch.cat([before_emb, skill_emb, after_emb], dim=1)  # (1, T, D_actor)
    attn_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=device)

    out = actor(inputs_embeds=full_emb, attention_mask=attn_mask, use_cache=False)
    logits = out.logits  # (1, T, V)

    # Build labels: -100 for prompt portion (incl. before, latents, and prompt
    # tokens of after up to assistant marker); valid IDs for response tokens.
    # The "valid" labels are the after_ids[:, n_after_prompt:].
    n_lat = latent_reprs.shape[0]
    before_len = before_ids.shape[1]

    labels = torch.full(full_emb.shape[:2], -100, dtype=torch.long, device=device)
    # Response starts at: before_len + n_lat + n_after_prompt
    resp_start = before_len + n_lat + n_after_prompt
    if resp_start < full_emb.shape[1]:
        # response tokens come from after_ids starting at index n_after_prompt
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
    parser.add_argument("--latent-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb-project", type=str, default="latentskill")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--use-lora", action="store_true",
                        help="Train Composer with LoRA adapter (LatentMem default). "
                             "Implicit low-rank regularization prevents the encoder-output "
                             "collapse seen in full fine-tuning + frozen Actor setups.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Args: {vars(args)}")
        logger.info(f"World size: {accelerator.num_processes}")

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

    # Composer (TRAINABLE, DDP-wrapped). Load in fp32 — accelerate autocast does
    # forward in bf16, but params + Adam state stay fp32 so small updates aren't
    # rounded to zero (bf16 precision around |w|=74 is ~0.585; lr×grad ~ 1e-7
    # would round to zero in bf16). Memory cost: 2x params/grads/Adam vs bf16.
    # lm_head is NOT used in our encode path (we take hidden states from
    # composer.model, not logits) so freeze it — otherwise DDP
    # find_unused_parameters=False errors at backward.
    # bf16 for LoRA mode (LoRA adapter is fp32 trainable but base bf16 saves memory)
    # fp32 for full FT mode (avoid bf16 update-rounds-to-zero around weight magnitude 74)
    composer_dtype = torch.bfloat16 if args.use_lora else torch.float32
    composer = AutoModelForCausalLM.from_pretrained(
        args.composer_path, torch_dtype=composer_dtype, trust_remote_code=True,
    )
    # Untie lm_head BEFORE freezing — Qwen2.5-3B has tie_word_embeddings=True
    # by default (lm_head.weight IS embed_tokens.weight, same memory). If we
    # freeze lm_head without untying first, embed_tokens silently freezes too.
    if composer.config.tie_word_embeddings:
        composer.config.tie_word_embeddings = False
        composer.lm_head.weight = nn.Parameter(composer.lm_head.weight.data.clone())
    for p in composer.lm_head.parameters():
        p.requires_grad = False
    composer.gradient_checkpointing_enable()

    # Snapshot pre-training fingerprint NOW (before any PEFT wrap, so the path
    # composer.model.layers[0]... is valid).
    composer_qproj_ref = composer.model.layers[0].self_attn.q_proj.weight.detach().float().norm().item()

    if args.use_lora:
        # LoRA training (LatentMem-style): implicit low-rank regularization
        # prevents the encoder-output collapse seen with full FT + frozen Actor.
        # Only LoRA adapter weights are trainable (~1% of params).
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        composer = get_peft_model(composer, peft_cfg)
        # PEFT defaults LoRA adapters to fp32, but base composer is bf16
        # (composer_dtype = torch.bfloat16 in LoRA branch above). Mixed dtypes
        # cause "mat1 and mat2 to have the same dtype" matmul errors. Cast
        # adapter weights to bf16 to match base.
        for n, p in composer.named_parameters():
            if "lora_" in n and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        if accelerator.is_main_process:
            composer.print_trainable_parameters()

    # Actor (FROZEN, per-rank full replica, eval). Stays bf16 to save memory —
    # frozen params don't get updated so bf16 precision is fine.
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    for p in actor.parameters():
        p.requires_grad = False
    actor.eval()
    actor = actor.to(accelerator.device)

    # Hidden-dim projection: Composer (small) → Actor (big) hidden space.
    # E.g. Qwen2.5-3B (D=2048) → Qwen2.5-7B (D=3584). Standard LLaVA-style
    # projector. Trainable. Init: small Gaussian.
    D_composer = composer.config.hidden_size
    D_actor = actor.config.hidden_size
    if D_composer != D_actor:
        # fp32 for the projection weight too (same precision-overflow logic).
        latent_proj = nn.Linear(D_composer, D_actor, bias=False).to(
            accelerator.device, dtype=torch.float32,
        )
        with torch.no_grad():
            nn.init.normal_(latent_proj.weight, std=0.02)
        if accelerator.is_main_process:
            logger.info(f"latent_proj: {D_composer} → {D_actor} "
                        f"({D_composer*D_actor:,} params, fp32)")
    else:
        latent_proj = nn.Identity()
        if accelerator.is_main_process:
            logger.info(f"D_composer == D_actor == {D_actor}, no projection needed")

    # query_latents — STANDALONE Parameter (fp32 for update precision).
    ql_init = (torch.randn(args.latents_per_skill, D_composer) * 0.02).to(torch.float32).to(accelerator.device)
    query_latents = nn.Parameter(ql_init, requires_grad=True)

    # composer_qproj_ref already captured above (before LoRA wrap)
    actor_qproj_ref = actor.model.layers[0].self_attn.q_proj.weight.detach().float().norm().item()

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

    # DDP wrap Composer (NO FSDP — accelerate uses DDP by default with no --use_fsdp).
    # Composer 3B fully replicated per rank, grads averaged across ranks via DDP.
    composer = accelerator.prepare(composer)
    # Inner reference for the encode helper (DDP wrapper doesn't proxy
    # `.get_input_embeddings()` / `.model`, so we use the unwrapped inner.
    # Params are SHARED — calling inner.forward still triggers DDP grad sync via
    # the param.backward hooks DDP installed during prepare.
    composer_inner = accelerator.unwrap_model(composer)

    # Optimizer (DDP doesn't change param identities, but follow the safe pattern)
    composer_params = [p for _, p in composer.named_parameters() if p.requires_grad]
    optimizer_groups = [
        {"params": composer_params, "lr": args.composer_lr, "weight_decay": args.weight_decay},
        {"params": [query_latents], "lr": args.latent_lr, "weight_decay": 0.0},
    ]
    proj_params = [p for p in latent_proj.parameters() if p.requires_grad]
    if proj_params:
        # Projection LR: small param, can use latent_lr / 10 (between composer_lr and latent_lr)
        optimizer_groups.append(
            {"params": proj_params, "lr": args.latent_lr / 10, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(args.adam_beta1, args.adam_beta2))

    total_steps = len(rank_indices) * args.epochs * accelerator.num_processes
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    if accelerator.is_main_process:
        logger.info(f"Trainable: composer={sum(p.numel() for p in composer_params):,} "
                    f"+ query_latents={query_latents.numel():,}")
        logger.info(f"Composer ref q_proj.norm = {composer_qproj_ref:.6f}")
        logger.info(f"Actor    ref q_proj.norm = {actor_qproj_ref:.6f} (must NOT change)")

    pad_id = composer_tok.pad_token_id or 0
    actor_embed_layer = actor.get_input_embeddings()  # safe: actor not FSDP'd

    global_step = 0
    for epoch in range(args.epochs):
        composer.train()
        ep_loss = 0.0
        n_ok = 0

        for dataset_idx in rank_indices:
            sample = dataset[dataset_idx]
            batch = make_batch(sample, composer_tok, actor_tok, args.max_length,
                               accelerator.device)
            if batch is None:
                continue  # pre-validated; should not happen

            with accelerator.accumulate(composer):
                # 1) Encode skills via Composer (with grad). Pass UNWRAPPED inner —
                # DDP wrapper doesn't proxy `.get_input_embeddings()` / `.model`.
                # Grad sync still happens via DDP backward hooks on params.
                latent_reprs_composer = composer_encode_skills(
                    composer_inner, query_latents, args.latents_per_skill,
                    batch["skill_input_ids_list"], pad_id, D_composer,
                )

                # 2) Project Composer hidden → Actor hidden (no-op if same dim).
                # latent_proj is fp32 by design (precision); under LoRA the
                # composer body is bf16 so its hidden states come out bf16 →
                # cast input to fp32 to match. Under full-FT both are already fp32.
                latent_reprs_actor = latent_proj(latent_reprs_composer.float())

                # 3) Actor forward (frozen) + loss
                loss = actor_forward_loss(
                    actor, actor_embed_layer,
                    batch["before_ids"], batch["after_ids"],
                    latent_reprs_actor, batch["n_after_prompt_ids"],
                )

                # 3) Backward → grads to Composer + query_latents
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(composer.parameters(), 1.0)
                    if query_latents.grad is not None:
                        torch.nn.utils.clip_grad_norm_([query_latents], 1.0)

                do_log = (global_step + 1) % args.log_every == 0
                if do_log:
                    ql_norm = query_latents.detach().float().norm().item()
                    ql_grad = (query_latents.grad.detach().float().norm().item()
                               if query_latents.grad is not None else 0.0)
                    inner_c = accelerator.unwrap_model(composer)
                    cqp = _get_qwen2_qproj(inner_c)
                    composer_qproj_now = cqp.detach().float().norm().item()
                    composer_qproj_grad = (cqp.grad.detach().float().norm().item()
                                           if cqp.grad is not None else -1.0)
                else:
                    ql_norm = ql_grad = composer_qproj_now = composer_qproj_grad = 0.0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            ep_loss += loss.item()
            n_ok += 1
            global_step += 1

            if do_log and accelerator.is_main_process:
                model_lr = optimizer.param_groups[0]["lr"]
                latent_lr = optimizer.param_groups[1]["lr"]
                logger.info(
                    f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} ql_norm={ql_norm:.3f} ql_grad={ql_grad:.3e} "
                    f"composer_qproj_norm_local={composer_qproj_now:.4f} "
                    f"composer_qproj_grad_local={composer_qproj_grad:.3e} "
                    f"composer_lr={model_lr:.2e} latent_lr={latent_lr:.2e}"
                )
                if not args.no_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/ql_norm": ql_norm,
                        "train/ql_grad_norm": ql_grad,
                        "train/composer_qproj_norm_local": composer_qproj_now,
                        "train/composer_qproj_grad_local": composer_qproj_grad,
                        "train/composer_lr": model_lr,
                        "train/latent_lr": latent_lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

            if global_step % args.save_every == 0 and global_step > 0:
                save_checkpoint(accelerator, composer, query_latents, args.latents_per_skill,
                                composer_tok, args.output_dir, global_step,
                                ql_only=False, composer_qproj_ref=composer_qproj_ref,
                                actor=actor, actor_qproj_ref=actor_qproj_ref,
                                latent_proj=latent_proj)

        avg = ep_loss / max(n_ok, 1)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} avg_loss={avg:.4f} ({n_ok} per rank)")
            if not args.no_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg, "epoch/epoch": epoch + 1}, step=global_step)

        is_last = (epoch + 1 == args.epochs)
        save_checkpoint(accelerator, composer, query_latents, args.latents_per_skill,
                        composer_tok, args.output_dir, global_step,
                        ql_only=not is_last, epoch=epoch + 1,
                        composer_qproj_ref=composer_qproj_ref,
                        actor=actor, actor_qproj_ref=actor_qproj_ref,
                        latent_proj=latent_proj)


def save_checkpoint(accelerator, composer, query_latents, latents_per_skill,
                    composer_tok, output_dir, global_step,
                    ql_only=False, epoch=None, composer_qproj_ref=None,
                    actor=None, actor_qproj_ref=None, latent_proj=None):
    """Save Composer (DDP-wrapped → params replicated, simple unwrap) +
    query_latents + latent_proj."""
    accelerator.wait_for_everyone()
    out = Path(output_dir)

    # query_latents + latent_proj — standalone, replicated → just clone on rank 0
    if accelerator.is_main_process:
        full_ql = query_latents.detach().float().cpu().clone()
        save_dict = {
            "query_latents": full_ql, "step": global_step, "epoch": epoch,
            "latents_per_skill": latents_per_skill,
        }
        if latent_proj is not None and isinstance(latent_proj, nn.Linear):
            save_dict["latent_proj_state"] = latent_proj.state_dict()
            save_dict["proj_in_dim"] = latent_proj.in_features
            save_dict["proj_out_dim"] = latent_proj.out_features
        torch.save(save_dict, out / f"query_latents_step{global_step}.pt")
        logger.info(f"Saved query_latents (norm={full_ql.norm().item():.3f}) "
                    f"+ latent_proj (Linear={isinstance(latent_proj, nn.Linear)})")

    if ql_only:
        accelerator.wait_for_everyone()
        return

    # Composer DDP-wrapped → full params on every rank → unwrap_model().state_dict() works
    inner = accelerator.unwrap_model(composer)
    suffix = f"epoch{epoch}" if epoch else f"step{global_step}"
    model_dir = out / f"composer_{suffix}"

    if accelerator.is_main_process:
        # If LoRA-wrapped, merge adapter into base before extracting state_dict —
        # encode_latent_library expects standard HF model files.
        from peft import PeftModel
        if isinstance(inner, PeftModel):
            logger.info("LoRA detected → merging adapter into base for save")
            merged = inner.merge_and_unload()  # returns base model with LoRA folded in
            full_state = merged.state_dict()
            save_target = merged
        else:
            full_state = inner.state_dict()
            save_target = inner

        save_state = {k: v.detach().cpu().to(torch.bfloat16) for k, v in full_state.items()}
        sample_norm = save_state["model.layers.0.self_attn.q_proj.weight"].float().norm().item()
        diff = abs(sample_norm - composer_qproj_ref) if composer_qproj_ref else -1.0
        logger.info(f"Composer save: q_proj.norm={sample_norm:.6f} ref={composer_qproj_ref:.6f} "
                    f"abs_diff={diff:.6e} (>1e-3 expected after training)")
        if composer_qproj_ref is not None and diff < 1e-6:
            logger.error("!!! Composer q_proj UNCHANGED — training did not propagate!")

        if actor is not None:
            actor_qproj_now = actor.model.layers[0].self_attn.q_proj.weight.detach().float().norm().item()
            actor_diff = abs(actor_qproj_now - actor_qproj_ref) if actor_qproj_ref else -1.0
            logger.info(f"Actor sanity: q_proj.norm={actor_qproj_now:.6f} ref={actor_qproj_ref:.6f} "
                        f"abs_diff={actor_diff:.6e} (MUST be 0 — actor frozen)")
            if actor_diff > 1e-6:
                logger.error("!!! Actor q_proj CHANGED — frozen Actor was modified!")

        save_target.save_pretrained(model_dir, state_dict=save_state,
                                    safe_serialization=True, max_shard_size="10GB")
        composer_tok.save_pretrained(model_dir)
        logger.info(f"Saved Composer ckpt → {model_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
