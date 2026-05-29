"""Plan A: End-to-end joint SFT (Composer + 7B model) with FSDP multi-GPU.

base_model + query_latents trained jointly via standard next-token SFT loss.
query_latents is attached as nn.Parameter on base_model (NOT in an outer wrapper)
so FSDP wraps base_model directly — avoids pytorch#122663 (state_dict on a
non-FSDP outer wrapper silently returns ungathered initial weights).

Three run modes (all via the same script, different flags):
  default                            Plan A: train query_latents + model jointly
  --freeze-latents                   Plan A B (ablation): random ql fixed, train model
  --freeze-latents --init-latent-from <path>  Plan A C: pretrained-frozen ql, train model

Launch via accelerate (must include --fsdp_use_orig_params true for multi-LR):
  accelerate launch --num_processes 4 --use_fsdp --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
    --fsdp_use_orig_params true --fsdp_state_dict_type FULL_STATE_DICT \
    --mixed_precision bf16 \
    -m src.train_joint_sft_fsdp \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --sft-data /work/.../alfworld/train-00000-of-00001.parquet \
      --output-dir /work/.../joint_sft_A \
      --epochs 4
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

SKILLS_START_MARKER = "## Retrieved Relevant Experience"
SKILLS_END_MARKER = "## Current Progress"


def parse_skills_from_content(content: str):
    start = content.find(SKILLS_START_MARKER)
    end = content.find(SKILLS_END_MARKER)
    if start == -1 or end == -1:
        return content, [], ""
    before = content[:start + len(SKILLS_START_MARKER)]
    skills_section = content[start + len(SKILLS_START_MARKER):end]
    after = content[end:]
    individual = []
    for line in skills_section.strip().split("\n"):
        line = line.strip()
        if line.startswith("- **"):
            individual.append(line[2:])
    return before, individual, after


# ----------------------------------------------------------------------- #
# Model wrapper
# ----------------------------------------------------------------------- #

def make_query_latents(base_model: nn.Module, latents_per_skill: int,
                       freeze_latents: bool, device) -> nn.Parameter:
    """Create query_latents as a STANDALONE nn.Parameter (NOT inside base_model).

    Why standalone: putting query_latents as a submodule of base_model means FSDP
    wraps it as part of the root flat_param. With use_orig_params=True, accessing
    it OUTSIDE FSDP forward (e.g. from our embedding-hook closure) gives a 0-dim
    or 1-D shard view → IndexError. As a standalone param, it lives fully on each
    rank, replicated. Memory cost: trivial (k * D * 2 bytes ≈ 14KB).
    """
    D = base_model.config.hidden_size
    ql_dtype = next(base_model.parameters()).dtype
    ql_init = (torch.randn(latents_per_skill, D) * 0.02).to(ql_dtype).to(device)
    return nn.Parameter(ql_init, requires_grad=not freeze_latents)


def _forward_with_embedding_override(base_model, input_ids, attention_mask,
                                     override_vec, override_slices,
                                     output_hidden_states=False):
    """Run base_model.forward, but inject ``override_vec`` at the embedding layer
    at the positions specified by ``override_slices``.

    This avoids the FSDP issue where ``base_model.get_input_embeddings()(ids)``
    crashes outside of base_model.forward (param is sharded as 1-D flat tensor
    when accessed directly). By going through forward, FSDP's all-gather hooks
    fire and embed_tokens.weight is materialized to 2-D.

    Args:
        base_model: FSDP-wrapped model
        input_ids: (B, T) int64
        attention_mask: (B, T) int64
        override_vec: (k, D) shared across batch, OR (B, k_b, D) per-sample
        override_slices: list of length B; each item is a slice or LongTensor of
                         positions in [0, T) to overwrite. May be empty.
        output_hidden_states: forwarded to base_model
    """
    embed = base_model.get_input_embeddings()
    is_per_sample = override_vec.dim() == 3

    def hook(module, inputs, output):
        # output: (B, T, D) — embeddings
        new_out = output.clone()
        for b, sl in enumerate(override_slices):
            n = sl.stop - sl.start if isinstance(sl, slice) else len(sl)
            if n == 0:
                continue
            vec_b = override_vec[b] if is_per_sample else override_vec
            new_out[b, sl, :] = vec_b[:n].to(output.dtype)
        return new_out

    handle = embed.register_forward_hook(hook)
    try:
        out = base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=output_hidden_states, use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    return out


def encode_skills(base_model, query_latents, k, skill_input_ids_list,
                  pad_token_id: int = 0):
    """Encode all skills in ONE batched forward (FSDP-symmetric collective count)."""
    device = query_latents.device
    n_skills = len(skill_input_ids_list)
    if n_skills == 0:
        D = base_model.config.hidden_size
        return torch.zeros(0, D, device=device, dtype=query_latents.dtype)

    max_text_len = max(ids.shape[1] for ids in skill_input_ids_list)
    total_len = max_text_len + k
    input_ids = torch.full((n_skills, total_len), pad_token_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((n_skills, total_len), dtype=torch.long, device=device)
    for i, ids in enumerate(skill_input_ids_list):
        L = ids.shape[1]
        input_ids[i, :L] = ids[0]
        attn_mask[i, :L] = 1
    attn_mask[:, -k:] = 1  # query_latents positions always attended

    override_slices = [slice(total_len - k, total_len) for _ in range(n_skills)]
    out = _forward_with_embedding_override(
        base_model, input_ids, attn_mask,
        override_vec=query_latents,  # (k, D), broadcast across batch
        override_slices=override_slices,
        output_hidden_states=True,
    )
    last_hidden = out.hidden_states[-1]  # (n_skills, total_len, D)
    latents = last_hidden[:, -k:, :]
    return latents.reshape(n_skills * k, -1)


def joint_forward(base_model, query_latents, k, before_ids, skill_input_ids_list,
                  after_ids, n_prompt_tokens, full_token_ids, pad_token_id: int = 0):
    """Joint forward: encode skills, then full forward with hook-injected splices."""
    device = before_ids.device

    skill_latents = encode_skills(
        base_model, query_latents, k, skill_input_ids_list, pad_token_id=pad_token_id,
    )
    n_lat = skill_latents.shape[0]

    skill_placeholder = torch.full((1, n_lat), pad_token_id, dtype=torch.long, device=device)
    full_ids = torch.cat([before_ids, skill_placeholder, after_ids], dim=1)
    attn_mask = torch.ones(full_ids.shape, dtype=torch.long, device=device)

    before_len = before_ids.shape[1]
    override_slice = slice(before_len, before_len + n_lat)
    out = _forward_with_embedding_override(
        base_model, full_ids, attn_mask,
        override_vec=skill_latents,  # (n_lat, D)
        override_slices=[override_slice],
        output_hidden_states=False,
    )
    logits = out.logits

    labels = full_token_ids.clone()
    labels[:, :n_prompt_tokens] = -100
    labels[:, before_len:before_len + n_lat] = -100

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ----------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------- #

class JointSFTDataset(Dataset):
    """Loads SkillRL SFT parquet, exposes raw fields for collator."""

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 4096,
                 max_samples: int = None):
        df = pd.read_parquet(parquet_path)
        if max_samples:
            df = df.head(max_samples)
        self.samples = []
        for _, row in df.iterrows():
            instruction = str(row.get("instruction", row.get("prompt", "")))
            output = str(row.get("output", row.get("answer", "")))
            if not instruction or not output:
                continue
            if SKILLS_START_MARKER not in instruction:
                continue
            self.samples.append((instruction, output))
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        instruction, output = self.samples[idx]
        before, skill_texts, after = parse_skills_from_content(instruction)
        return {
            "instruction": instruction,
            "output": output,
            "before": before,
            "skill_texts": skill_texts,
            "after": after,
        }


def make_batch(sample, tokenizer, max_length, device):
    """Tokenize a single sample into the tensors joint_forward expects.

    Returns dict with: before_ids, skill_input_ids_list, after_ids,
    n_prompt_tokens, full_token_ids — or None if sample malformed.
    """
    placeholder = "[LATENT_SKILLS]"
    modified = sample["before"] + "\n" + placeholder + "\n" + sample["after"]
    messages = [
        {"role": "user", "content": modified},
        {"role": "assistant", "content": sample["output"]},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    pos = full_text.find(placeholder)
    if pos < 0:
        return None

    before_text = full_text[:pos]
    after_text = full_text[pos + len(placeholder):]

    before_ids = tokenizer(before_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    after_ids = tokenizer(after_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    skill_ids_list = []
    for skill_text in sample["skill_texts"]:
        sid = tokenizer(skill_text, add_special_tokens=False,
                        return_tensors="pt", truncation=True, max_length=128
                        )["input_ids"].to(device)
        skill_ids_list.append(sid)

    if not skill_ids_list:
        return None

    # Compute n_prompt_tokens: before + skills + (after up to assistant marker)
    assistant_marker = "<|im_start|>assistant\n"
    pre_resp_text = ""
    if assistant_marker in after_text:
        pre_resp_text = after_text[:after_text.find(assistant_marker) + len(assistant_marker)]
    n_after_prompt = (
        len(tokenizer(pre_resp_text, add_special_tokens=False)["input_ids"])
        if pre_resp_text else 0
    )
    n_lat_total = len(skill_ids_list) * 2  # latents_per_skill=2 baked in
    n_prompt = before_ids.shape[1] + n_lat_total + n_after_prompt

    full_token_ids = torch.cat([
        before_ids,
        torch.zeros(1, n_lat_total, dtype=torch.long, device=device),
        after_ids,
    ], dim=1)
    if full_token_ids.shape[1] > max_length:
        # truncate from the right; latent positions need to be preserved
        full_token_ids = full_token_ids[:, :max_length]

    return {
        "before_ids": before_ids,
        "skill_input_ids_list": skill_ids_list,
        "after_ids": after_ids,
        "n_prompt_tokens": n_prompt,
        "full_token_ids": full_token_ids,
    }


# ----------------------------------------------------------------------- #
# Training
# ----------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4,
                        help="Aligned with verl/SkillRL SFT default")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap dataset size (for testing)")
    parser.add_argument("--model-lr", type=float, default=1e-5)
    parser.add_argument("--latent-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Aligned with verl/SkillRL warmup_steps_ratio=0.1")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95,
                        help="Aligned with verl/SkillRL betas=[0.9, 0.95]")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=500, help="Save ckpt every N steps")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--freeze-latents", action="store_true",
                        help="Ablation: keep query_latents at initialization, no gradient")
    parser.add_argument("--init-latent-from", type=str, default=None,
                        help="Optional .pt path to load query_latents from (e.g. Stage 1 SFT result). "
                             "Use with --freeze-latents for Group C (pretrained-frozen)")
    parser.add_argument("--wandb-project", type=str, default="latentskill")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Defaults to basename of --output-dir")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Args: {vars(args)}")
        logger.info(f"World size: {accelerator.num_processes}")
        if args.freeze_latents:
            mode_str = ("FROZEN-PRETRAINED (ablation C)" if args.init_latent_from
                        else "FROZEN-RANDOM (ablation B)")
        else:
            mode_str = "JOINT (Plan A main)"
        logger.info(f"Mode: {mode_str}")

        # Init wandb (rank 0 only). Picks up WANDB_API_KEY/WANDB_PROJECT/WANDB_ENTITY from env.
        if not args.no_wandb:
            try:
                import wandb
                run_name = args.wandb_run_name or Path(args.output_dir).name
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=vars(args),
                    resume="allow",
                )
                logger.info(f"wandb initialized: project={args.wandb_project}, run={run_name}")
            except Exception as e:
                logger.warning(f"wandb init failed ({e}); continuing without wandb")
                args.no_wandb = True

    # Load tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # Snapshot pre-training reference for save-correctness verification later.
    # Take a small fingerprint (norm of layer 0 q_proj) BEFORE FSDP wrap, on CPU.
    ref_qproj_norm = base_model.model.layers[0].self_attn.q_proj.weight.detach().float().norm().item()

    # Create query_latents as STANDALONE Parameter (NOT inside base_model).
    # See make_query_latents docstring: avoids 0-dim shard view bug when accessed
    # from embed-hook closure outside FSDP forward.
    query_latents = make_query_latents(
        base_model, args.latents_per_skill, args.freeze_latents,
        device=accelerator.device,
    )

    # Optionally load pre-trained query_latents (e.g. Group C: Stage 1 SFT'd ql)
    if args.init_latent_from:
        ql_data = torch.load(args.init_latent_from, map_location="cpu", weights_only=False)
        loaded_ql = ql_data["query_latents"] if isinstance(ql_data, dict) else ql_data
        with torch.no_grad():
            query_latents.data.copy_(loaded_ql.to(query_latents.dtype).to(query_latents.device))
        if accelerator.is_main_process:
            logger.info(f"Loaded query_latents from {args.init_latent_from} "
                        f"(norm={query_latents.norm().item():.3f})")

    # Dataset
    dataset = JointSFTDataset(
        args.sft_data, tokenizer, args.max_length, max_samples=args.max_samples
    )

    # Sample-by-sample loader (batch_size=1 because seqs are huge and variable).
    # For multi-GPU, each rank gets its own slice. Two safety requirements for
    # FSDP collectives: (1) all ranks must hit same number of backwards per epoch,
    # (2) make_batch must never return None during training (otherwise rank skips
    # while others do backward → NCCL timeout after 30 min).
    rank = accelerator.process_index
    world = accelerator.num_processes

    # Pre-validate: walk all samples once on rank 0, broadcast valid index list to all ranks
    if accelerator.is_main_process:
        logger.info("Pre-validating dataset (filtering samples that produce None batches)...")
        valid_indices = []
        for i in range(len(dataset)):
            sample_i = dataset[i]
            test_batch = make_batch(sample_i, tokenizer, args.max_length, "cpu")
            if test_batch is not None:
                valid_indices.append(i)
        # Truncate to multiple of world_size so per-rank counts are EXACTLY equal
        n_drop = len(valid_indices) % world
        if n_drop:
            valid_indices = valid_indices[:-n_drop]
        valid_indices_t = torch.tensor(valid_indices, dtype=torch.long)
        logger.info(f"Valid samples: {len(valid_indices)} (dropped {n_drop} for divisibility, "
                    f"original total {len(dataset)})")
    else:
        valid_indices_t = torch.tensor([], dtype=torch.long)

    # Broadcast the valid index list to all ranks
    accelerator.wait_for_everyone()
    obj_list = [valid_indices_t if accelerator.is_main_process else None]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    valid_indices_t = obj_list[0]
    valid_indices = valid_indices_t.tolist()

    # Per-rank stride — guaranteed equal length across ranks
    rank_indices = valid_indices[rank::world]
    if accelerator.is_main_process:
        logger.info(f"Per-rank samples: {len(rank_indices)} (× {world} ranks = {len(valid_indices)} total)")

    # DEBUG: confirm FSDP plugin config that accelerate received
    if accelerator.is_main_process:
        fsdp_plugin = getattr(accelerator.state, 'fsdp_plugin', None)
        if fsdp_plugin is not None:
            logger.info(f"FSDP plugin: use_orig_params={fsdp_plugin.use_orig_params!r} "
                        f"sharding_strategy={fsdp_plugin.sharding_strategy!r} "
                        f"auto_wrap_policy={type(fsdp_plugin.auto_wrap_policy).__name__ if fsdp_plugin.auto_wrap_policy else None} "
                        f"state_dict_type={fsdp_plugin.state_dict_type!r} "
                        f"mixed_precision_policy={fsdp_plugin.mixed_precision_policy!r}")
        else:
            logger.warning("No FSDP plugin found in accelerator.state!")

    # FSDP wrap base_model FIRST. Optimizer is created AFTER, referencing the
    # FSDP-managed params.
    base_model = accelerator.prepare(base_model)

    # DEBUG: after FSDP wrap, inspect actual param structure
    if accelerator.is_main_process:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
        unwrapped = accelerator.unwrap_model(base_model)
        sample_param = unwrapped.model.layers[0].self_attn.q_proj.weight
        logger.info(f"DEBUG post-FSDP: unwrapped.model.layers[0].self_attn.q_proj.weight: "
                    f"shape={sample_param.shape} numel={sample_param.numel()} "
                    f"is_FSDP_FlatParameter={'FlatParameter' in type(sample_param).__name__} "
                    f"requires_grad={sample_param.requires_grad}")
        logger.info(f"DEBUG: base_model type={type(base_model).__name__} "
                    f"is_FSDP={isinstance(base_model, _FSDP)} "
                    f"unwrapped type={type(unwrapped).__name__}")
        # Count FSDP modules
        n_fsdp = sum(1 for m in base_model.modules() if isinstance(m, _FSDP))
        logger.info(f"DEBUG: total FSDP-wrapped modules in tree = {n_fsdp}")

    # Now optimizer with two LR groups, references POST-FSDP params.
    params_model = [p for _, p in base_model.named_parameters() if p.requires_grad]
    optimizer_groups = [
        {"params": params_model, "lr": args.model_lr, "weight_decay": args.weight_decay},
    ]
    if not args.freeze_latents:
        optimizer_groups.append(
            {"params": [query_latents], "lr": args.latent_lr, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(args.adam_beta1, args.adam_beta2))

    total_steps = len(rank_indices) * args.epochs * accelerator.num_processes
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    if accelerator.is_main_process:
        logger.info(f"Scheduler: total_steps={total_steps} warmup_steps={warmup_steps} "
                    f"(per-rank steps={len(rank_indices)*args.epochs}, accelerate advances {accelerator.num_processes}x per step.step())")
        n_model = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        n_ql = query_latents.numel() if query_latents.requires_grad else 0
        logger.info(f"Trainable params: model={n_model:,} + query_latents={n_ql:,} "
                    f"(ql {'frozen' if not query_latents.requires_grad else 'trainable'})")
        logger.info(f"Pre-training reference: layers[0].q_proj norm = {ref_qproj_norm:.6f} "
                    "(post-training norm in save log must differ)")
        logger.info(f"Optimizer param_group[0] (model) has {len(optimizer_groups[0]['params'])} tensors, "
                    f"first tensor.shape={optimizer_groups[0]['params'][0].shape}, "
                    f"first tensor identity={id(optimizer_groups[0]['params'][0])}, "
                    f"matches base_model.model.embed_tokens.weight identity={id(accelerator.unwrap_model(base_model).model.embed_tokens.weight)}")

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        base_model.train()
        ep_loss = 0.0
        n_ok = 0

        for local_idx, dataset_idx in enumerate(rank_indices):
            sample = dataset[dataset_idx]
            batch = make_batch(sample, tokenizer, args.max_length, accelerator.device)
            if batch is None:
                # Should never happen after pre-validation. If it does, we MUST
                # still call all collectives to keep ranks in sync, otherwise
                # NCCL timeout. Log loudly and proceed with a dummy single-token
                # batch (zero loss).
                if accelerator.is_main_process:
                    logger.error(f"Sample {dataset_idx} unexpectedly returned None batch! "
                                 f"Inserting dummy step to avoid NCCL desync.")
                # Build a minimal dummy batch: single BOS token, label=-100
                bos = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
                dummy_ids = torch.tensor([[bos, bos]], dtype=torch.long, device=accelerator.device)
                batch = {
                    "before_ids": dummy_ids,
                    "skill_input_ids_list": [dummy_ids],
                    "after_ids": dummy_ids,
                    "n_prompt_tokens": 4,  # whole thing masked
                    "full_token_ids": torch.tensor([[bos, bos, bos, bos, bos, bos]],
                                                    dtype=torch.long, device=accelerator.device),
                }

            with accelerator.accumulate(base_model):
                loss = joint_forward(
                    base_model, query_latents, args.latents_per_skill,
                    before_ids=batch["before_ids"],
                    skill_input_ids_list=batch["skill_input_ids_list"],
                    after_ids=batch["after_ids"],
                    n_prompt_tokens=batch["n_prompt_tokens"],
                    full_token_ids=batch["full_token_ids"],
                    pad_token_id=tokenizer.pad_token_id or 0,
                )

                accelerator.backward(loss)

                # DEBUG (BEFORE clip + step + zero): is grad flowing to model params?
                _do_debug = (global_step < 3) or ((global_step + 1) % args.log_every == 0)
                if _do_debug and accelerator.is_main_process:
                    inner_dbg = accelerator.unwrap_model(base_model)
                    qpw = inner_dbg.model.layers[0].self_attn.q_proj.weight
                    embw = inner_dbg.model.embed_tokens.weight
                    qpw_grad_norm = qpw.grad.detach().float().norm().item() if qpw.grad is not None else -999.0
                    qpw_grad_shape = list(qpw.grad.shape) if qpw.grad is not None else None
                    embw_grad_norm = embw.grad.detach().float().norm().item() if embw.grad is not None else -999.0
                    qpw_data_pre_step_norm = qpw.data.detach().float().norm().item()
                    logger.info(f"[DBG step {global_step}] AFTER backward, BEFORE step: "
                                f"q_proj.grad_norm={qpw_grad_norm:.6e} grad_shape={qpw_grad_shape} "
                                f"embed.grad_norm={embw_grad_norm:.6e} "
                                f"q_proj.data_norm_pre={qpw_data_pre_step_norm:.6f}")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(base_model.parameters(), 1.0)
                    if query_latents.requires_grad and query_latents.grad is not None:
                        torch.nn.utils.clip_grad_norm_([query_latents], 1.0)

                do_log = (global_step + 1) % args.log_every == 0
                if do_log:
                    ql_grad_norm_local = (
                        query_latents.grad.detach().float().norm().item()
                        if query_latents.grad is not None else 0.0
                    )
                    ql_norm_local = query_latents.detach().float().norm().item()
                else:
                    ql_grad_norm_local = 0.0
                    ql_norm_local = 0.0

                optimizer.step()
                # DEBUG: data norm AFTER step
                if _do_debug and accelerator.is_main_process:
                    inner_dbg2 = accelerator.unwrap_model(base_model)
                    qpw2 = inner_dbg2.model.layers[0].self_attn.q_proj.weight
                    qpw_data_post_step_norm = qpw2.data.detach().float().norm().item()
                    logger.info(f"[DBG step {global_step}] AFTER step: "
                                f"q_proj.data_norm_post={qpw_data_post_step_norm:.6f}")
                scheduler.step()
                optimizer.zero_grad()

            ep_loss += loss.item()
            n_ok += 1
            global_step += 1

            # Diagnostic: also probe a decoder param's LOCAL norm to verify training
            if do_log:
                inner_d = accelerator.unwrap_model(base_model)
                try:
                    qp = inner_d.model.layers[0].self_attn.q_proj.weight
                    decoder_norm_local = qp.detach().float().norm().item()
                    decoder_grad_local = (qp.grad.detach().float().norm().item()
                                          if qp.grad is not None else -1.0)
                except Exception:
                    decoder_norm_local = -1.0
                    decoder_grad_local = -1.0
            if do_log and accelerator.is_main_process:
                model_lr = optimizer.param_groups[0]["lr"]
                latent_lr = (optimizer.param_groups[1]["lr"]
                             if len(optimizer.param_groups) > 1 else 0.0)
                logger.info(
                    f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} ql_norm_local={ql_norm_local:.3f} "
                    f"ql_grad_local={ql_grad_norm_local:.3e} model_lr={model_lr:.2e} "
                    f"decoder_q_norm_local={decoder_norm_local:.4f} decoder_q_grad_local={decoder_grad_local:.3e}"
                )
                if not args.no_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/ql_norm_local": ql_norm_local,
                        "train/ql_grad_norm_local": ql_grad_norm_local,
                        "train/model_lr": model_lr,
                        "train/latent_lr": latent_lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

            if global_step % args.save_every == 0 and global_step > 0:
                save_checkpoint(accelerator, base_model, query_latents, args.latents_per_skill,
                                tokenizer, args.output_dir, global_step,
                                ql_only=False, ref_qproj_norm=ref_qproj_norm)

        avg = ep_loss / max(n_ok, 1)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} avg_loss={avg:.4f} ({n_ok} samples per rank)")
            if not args.no_wandb:
                import wandb
                wandb.log({
                    "epoch/avg_loss": avg,
                    "epoch/samples_per_rank": n_ok,
                    "epoch/epoch": epoch + 1,
                }, step=global_step)

        # Per-epoch: save query_latents (tiny). Model only at final epoch
        # (saves disk: 1 × 14GB instead of N × 14GB per run).
        is_last_epoch = (epoch + 1 == args.epochs)
        save_checkpoint(accelerator, base_model, query_latents, args.latents_per_skill,
                        tokenizer, args.output_dir, global_step,
                        ql_only=not is_last_epoch, epoch=epoch + 1,
                        ref_qproj_norm=ref_qproj_norm)

    # Final save (already saved above if last_epoch was the loop body, idempotent here)


def save_checkpoint(accelerator, base_model, query_latents, latents_per_skill,
                    tokenizer, output_dir, global_step,
                    ql_only=False, epoch=None, final=False, ref_qproj_norm=None):
    """Save checkpoint. query_latents is standalone (not FSDP-managed) → saved
    directly. base_model is FSDP-wrapped → gather via state_dict_type context.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
    from torch.distributed.fsdp.api import FullStateDictConfig

    accelerator.wait_for_everyone()
    out = Path(output_dir)

    # 1) query_latents — standalone, full on every rank → just clone on rank 0.
    if accelerator.is_main_process:
        full_ql = query_latents.detach().float().cpu().clone()
        ql_path = out / f"query_latents_step{global_step}.pt"
        torch.save({
            "query_latents": full_ql,
            "step": global_step,
            "epoch": epoch,
            "latents_per_skill": latents_per_skill,
        }, ql_path)
        logger.info(f"Saved query_latents to {ql_path} (norm={full_ql.norm().item():.3f})")

    if ql_only:
        accelerator.wait_for_everyone()
        return

    # 2) base_model — gather across FSDP shards (collective on all ranks).
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    inner = accelerator.unwrap_model(base_model)
    if isinstance(base_model, FSDP):
        with FSDP.state_dict_type(base_model, StateDictType.FULL_STATE_DICT, cfg):
            full_state = base_model.state_dict()
    else:
        full_state = inner.state_dict()

    suffix = "final" if final else (f"epoch{epoch}" if epoch else f"step{global_step}")
    model_dir = out / f"model_{suffix}"

    if accelerator.is_main_process:
        save_state = {k: v.detach().cpu().to(torch.bfloat16) for k, v in full_state.items()}
        # Verify decoder weights actually changed (NeurIPS-quality sanity check).
        sample_key = "model.layers.0.self_attn.q_proj.weight"
        sample_norm = save_state[sample_key].float().norm().item() if sample_key in save_state else -1.0
        ref_str = f"{ref_qproj_norm:.6f}" if ref_qproj_norm is not None else "n/a"
        diff_vs_ref = abs(sample_norm - ref_qproj_norm) if ref_qproj_norm is not None else -1.0
        logger.info(f"Saving {len(save_state)} tensors to {model_dir} "
                    f"(includes lm_head: {'lm_head.weight' in save_state}, "
                    f"embed: {'model.embed_tokens.weight' in save_state}, "
                    f"layers.0.q_proj norm={sample_norm:.6f}, "
                    f"ref_pre_train={ref_str}, abs_diff={diff_vs_ref:.6e})")
        if ref_qproj_norm is not None and diff_vs_ref < 1e-6:
            logger.error("!!! q_proj norm IDENTICAL to pre-training reference. "
                         "Save likely captured untrained weights. Check FSDP wrap.")
        inner.save_pretrained(
            model_dir, state_dict=save_state,
            safe_serialization=True, max_shard_size="10GB",
        )
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Saved model+config+tokenizer to {model_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
