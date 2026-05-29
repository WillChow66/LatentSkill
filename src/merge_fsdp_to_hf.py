"""Merge verl FSDP checkpoint shards into HF format.

Custom merger that bypasses verl's model_merger.py — verl's version uses
torch.distributed.tensor.DeviceMesh APIs that break on PyTorch 2.10
(`'DeviceMesh' object has no attribute '_layout'`).

Assumes simple FSDP without TP: each DTensor is Shard(dim) across the
flat fsdp dim, so we just concat _local_tensor across ranks.

Usage:
  python -m src.merge_fsdp_to_hf \
      --actor-dir /path/to/actor \
      --target-dir /path/to/output_hf
"""

import argparse
import logging
import os
import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)


def find_world_size(actor_dir: Path) -> int:
    for fn in os.listdir(actor_dir):
        m = re.match(r"model_world_size_(\d+)_rank_0\.pt", fn)
        if m:
            return int(m.group(1))
    raise FileNotFoundError(f"No model_world_size_*_rank_0.pt found in {actor_dir}")


def merge(actor_dir: str, target_dir: str):
    actor_dir = Path(actor_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    world_size = find_world_size(actor_dir)
    logger.info(f"World size: {world_size}")

    shards = []
    for rank in range(world_size):
        path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        logger.info(f"Loading shard {rank}: {path.name}")
        shards.append(torch.load(path, map_location="cpu", weights_only=False))

    keys = sorted(shards[0].keys())
    logger.info(f"Keys: {len(keys)}")

    merged = {}
    for key in keys:
        tensors_per_rank = [shards[r][key] for r in range(world_size)]
        first = tensors_per_rank[0]
        if isinstance(first, DTensor):
            placements = first.placements
            assert len(placements) == 1, f"Expected 1-D mesh, got {placements}"
            placement = placements[0]
            local_tensors = [t._local_tensor.bfloat16() for t in tensors_per_rank]
            if placement.is_replicate():
                merged[key] = local_tensors[0]
            elif placement.is_shard():
                merged[key] = torch.cat(local_tensors, dim=placement.dim).contiguous()
            else:
                raise NotImplementedError(f"Unsupported placement {placement} for {key}")
        else:
            local_tensors = [t.bfloat16() for t in tensors_per_rank]
            merged[key] = torch.cat(local_tensors, dim=0).contiguous()

        # Drop loaded shard data for this key to save RAM
        for r in range(world_size):
            shards[r].pop(key, None)

    del shards

    # Build HF model on meta + load merged state_dict
    config_path = actor_dir
    logger.info(f"Loading HF config from {config_path}")
    model_config = AutoConfig.from_pretrained(config_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")

    try:
        model.generation_config = GenerationConfig.from_pretrained(config_path)
    except Exception as e:
        logger.warning(f"Skip generation_config: {e}")

    logger.info(f"Saving HF model to {target_dir}")
    model.save_pretrained(target_dir, state_dict=merged)

    tokenizer = AutoTokenizer.from_pretrained(config_path)
    tokenizer.save_pretrained(target_dir)
    logger.info(f"Done. Files: {sorted(os.listdir(target_dir))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    merge(args.actor_dir, args.target_dir)
