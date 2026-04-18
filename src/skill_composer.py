"""
SkillComposer: Encodes text skills into latent tokens using the agent model as encoder.

Architecture (LatentMem-inspired):
  - Encoder = agent model itself (frozen)
  - Learnable query_latents appended to skill text embeddings
  - Forward through encoder → extract last hidden states at query positions
  - These become the latent skill tokens

ComposerEmbeddingWrapper (TokMem-inspired):
  - Monkey-patches the model's embedding layer
  - For SKILL token positions, replaces embeddings with Composer's latent output
  - Maintains gradient flow from loss → Composer query_latents

Usage in end-to-end RL:
  1. Composer encodes skills → latent vectors (with grad)
  2. ComposerEmbeddingWrapper injects latent vectors during actor forward
  3. loss.backward() → gradient flows to Composer query_latents
"""

import json
import hashlib
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def hash_skill(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def extract_individual_skills(skills_json_path: str) -> list[dict]:
    """Load all skills from SkillRL's skill library."""
    with open(skills_json_path) as f:
        data = json.load(f)

    skills = []
    for s in data.get("general_skills", []):
        text = f"**{s['title']}**: {s['principle']}"
        if "when_to_apply" in s:
            text += f" Apply when: {s['when_to_apply']}"
        skills.append({"id": s["skill_id"], "text": text, "type": "general"})

    for task_type, task_skills in data.get("task_specific_skills", {}).items():
        for s in task_skills:
            text = f"**{s['title']}**: {s['principle']}"
            if "when_to_apply" in s:
                text += f" Apply when: {s['when_to_apply']}"
            skills.append({"id": s["skill_id"], "text": text, "type": task_type})

    for i, m in enumerate(data.get("common_mistakes", [])):
        if "title" in m and "principle" in m:
            text = f"**{m['title']}**: {m['principle']}"
            skills.append({"id": f"mistake_{i}", "text": text, "type": "mistakes"})

    return skills


class SkillComposer(nn.Module):
    """Compresses text skills into latent tokens via learned query_latents.

    The encoder is the agent model itself (frozen). Only query_latents are trainable.
    """

    def __init__(self, encoder_model=None, latents_per_skill: int = 2, hidden_size: int = None):
        """
        Args:
            encoder_model: the frozen encoder to use for skill encoding.
                Can be None at init time (set later via self.encoder = ...).
                In Plan C, we load a separate encoder_copy AFTER FSDP wrapping.
            latents_per_skill: number of latent tokens per skill (k)
            hidden_size: hidden dim; required if encoder_model is None
        """
        super().__init__()
        self.encoder = encoder_model
        self.latents_per_skill = latents_per_skill

        if encoder_model is not None:
            self.hidden_size = encoder_model.config.hidden_size
            # Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            assert hidden_size is not None, "hidden_size required when encoder_model is None"
            self.hidden_size = hidden_size

        # Learnable query latents (shared across all skills)
        self.query_latents = nn.Parameter(
            torch.randn(latents_per_skill, self.hidden_size) * 0.02
        )

        # Cache for skill texts and their token IDs
        self._skill_texts = None
        self._skill_token_ids = None
        self._tokenizer = None

    def load_pretrained(self, path: str):
        """Load pre-trained query_latents from Composer training."""
        data = torch.load(path, map_location="cpu")
        self.query_latents.data.copy_(data["query_latents"].to(self.query_latents.dtype))
        logger.info(f"Loaded pre-trained query_latents from {path}")

    def setup_skills(self, skills_json_path: str, tokenizer, skill_token_map: dict):
        """Setup skill texts and their corresponding SKILL token IDs.

        Args:
            skills_json_path: path to claude_style_skills.json
            tokenizer: tokenizer with SKILL tokens added
            skill_token_map: {skill_id: {"token_a": id, "token_b": id}}
        """
        self._tokenizer = tokenizer
        all_skills = extract_individual_skills(skills_json_path)

        self._skill_texts = []
        self._skill_token_ids = []  # list of (token_a_id, token_b_id) per skill

        for skill in all_skills:
            sid = skill["id"]
            if sid in skill_token_map:
                self._skill_texts.append(skill["text"])
                token_ids = skill_token_map[sid]["token_ids"]
                self._skill_token_ids.append((token_ids[0], token_ids[1]))
            else:
                logger.warning(f"Skill {sid} not found in skill_token_map, skipping")

        logger.info(f"SkillComposer setup: {len(self._skill_texts)} skills, "
                    f"{len(self._skill_token_ids)} token pairs")

    def encode_skills(self, skill_texts: list[str]) -> torch.Tensor:
        """Encode a batch of skill texts into latent tokens.

        Following LatentMem's text_to_latent() pattern (simple, standard forward).
        Uses the encoder model directly (which is a non-FSDP frozen copy in our
        Plan C setup, so no FSDP workaround needed).

        Args:
            skill_texts: list of skill text strings

        Returns: (batch, k, D) tensor with gradient through query_latents
        """
        device = self.query_latents.device
        tokens = self._tokenizer(
            skill_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128, add_special_tokens=False
        ).to(device)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Standard embedding lookup from (non-FSDP) encoder
        text_embeds = self.encoder.get_input_embeddings()(input_ids)

        # Append learnable query_latents (gradient flows through here)
        batch_size = input_ids.size(0)
        latents = self.query_latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents = latents.to(text_embeds.dtype)
        latents_mask = torch.ones(
            (batch_size, self.latents_per_skill),
            dtype=attention_mask.dtype, device=device
        )

        inputs_embeds = torch.cat([text_embeds, latents], dim=1)
        full_attn_mask = torch.cat([attention_mask, latents_mask], dim=1)

        # Use inner Qwen2Model (skip lm_head since we only need hidden states)
        inner_model = self.encoder.model if hasattr(self.encoder, 'model') else self.encoder

        outputs = inner_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn_mask,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state
        latent_tokens = hidden_states[:, -self.latents_per_skill:, :]  # (batch, k, D)
        return latent_tokens

    def encode_all_skills(self) -> dict[int, torch.Tensor]:
        """Encode all skills in one batch. Returns {token_id: latent_vector (with grad)}."""
        assert self._skill_texts is not None, "Call setup_skills() first"

        # Batch encode all skills at once
        all_latents = self.encode_skills(self._skill_texts)  # (n_skills, k, D)

        latent_map = {}
        for i, (token_a, token_b) in enumerate(self._skill_token_ids):
            latent_map[token_a] = all_latents[i, 0]  # first latent vector
            latent_map[token_b] = all_latents[i, 1]  # second latent vector

        return latent_map

    def get_all_skill_token_ids(self) -> list[int]:
        """Get flat list of all SKILL token IDs."""
        ids = []
        for token_a, token_b in self._skill_token_ids:
            ids.extend([token_a, token_b])
        return ids


def setup_model_overrides(model, composer: SkillComposer):
    """Monkey-patch model's embed_tokens for SKILL tokens.

    Following TokMem's pattern but only for embed_tokens:
    - embed_tokens.forward: SKILL token embeddings → Composer's latent output (with grad)

    LM head doesn't need monkey-patching because the v2 checkpoint already has
    lm_head.weight[SKILL_token_ids] = 0 (untied + zeroed). The model naturally
    won't generate SKILL tokens.

    Args:
        model: the HF model (e.g., Qwen2ForCausalLM)
        composer: SkillComposer instance with encoded skills
    """
    # Store original embed_tokens forward method
    original_embed_forward = model.model.embed_tokens.forward

    # Mutable state: Composer's latest latent output
    # Updated via composer.update_latent_cache() before each forward pass
    _latent_cache = {}  # {token_id: latent_vector}

    def custom_embed_forward(input_ids):
        """Replace SKILL token embeddings with Composer's latent output.
        Following TokMem's custom_embed_forward pattern.
        """
        embeddings = original_embed_forward(input_ids)

        if _latent_cache:
            # Following TokMem: direct replacement without clone
            for token_id, latent_vec in _latent_cache.items():
                mask = (input_ids == token_id)
                if mask.any():
                    embeddings[mask] = latent_vec.to(embeddings.dtype)

        return embeddings

    # Apply only the embed_tokens monkey-patch (lm_head is handled by checkpoint)
    model.model.embed_tokens.forward = custom_embed_forward

    # Attach cache reference to composer for updating
    composer._latent_cache = _latent_cache
    composer._original_embed_forward = original_embed_forward

    skill_token_ids = composer.get_all_skill_token_ids()
    logger.info(f"Model overrides applied: {len(skill_token_ids)} SKILL tokens "
                f"patched in embed_tokens (lm_head untied + zeroed in checkpoint)")

    def update_latent_cache(latent_map: dict[int, torch.Tensor]):
        """Update the shared latent cache (called before each forward pass)."""
        _latent_cache.clear()
        _latent_cache.update(latent_map)

    def clear_latent_cache():
        """Clear the cache (use original embeddings)."""
        _latent_cache.clear()

    # Attach update functions to composer
    composer.update_latent_cache = update_latent_cache
    composer.clear_latent_cache = clear_latent_cache
