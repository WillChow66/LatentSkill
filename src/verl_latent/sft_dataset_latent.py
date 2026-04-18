# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified for Latent Skills: uses pre-computed latent skill library
# instead of text skills in the prompt.
#
# Changes from SkillRL original (verl/utils/dataset/sft_dataset.py):
# 1. __init__: loads pre-computed latent_skill_library.pt
# 2. __getitem__: removes skills text from prompt, looks up pre-computed latent tokens
# 3. Returns extra fields: before_length, latent_tokens

import hashlib
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


# Skills section markers in SkillRL's instruction format
SKILLS_START_MARKER = "## Retrieved Relevant Experience"
SKILLS_END_MARKER = "## Current Progress"


def hash_skill(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def split_prompt_at_skills(prompt_str):
    """Split prompt at skills boundaries.
    Returns (before_str, skills_str, after_str, individual_skill_texts).
    """
    start = prompt_str.find(SKILLS_START_MARKER)
    end = prompt_str.find(SKILLS_END_MARKER)
    if start == -1 or end == -1:
        return prompt_str, "", "", []

    before = prompt_str[:start]
    skills = prompt_str[start:end]
    after = prompt_str[end:]

    individual_skills = []
    for line in skills.split("\n"):
        line = line.strip()
        if line.startswith("- **"):
            individual_skills.append(line[2:])  # Remove "- " prefix

    return before, skills, after, individual_skills


class SFTDataset(Dataset):
    """
    SFT Dataset with pre-computed latent skill tokens.

    Identical to SkillRL's SFTDataset except:
    - Skills text is removed from the prompt (replaced by latent tokens)
    - Pre-computed latent tokens are looked up from latent_skill_library.pt
    - Returns additional fields: before_length, latent_tokens
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config,
                 latent_library_path: str = None):
        # === SkillRL original init logic (unchanged) ===
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get('use_shm', False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

        # === LATENT SKILL: load pre-computed latent library ===
        self.latent_library = {}
        if latent_library_path:
            ckpt = torch.load(latent_library_path, map_location="cpu")
            self.latent_library = ckpt["latent_library"]  # {hash: (k, D) tensor}
            self.latents_per_skill = ckpt["latents_per_skill"]
            print(f"[LatentSkill] Loaded {len(self.latent_library)} pre-computed skill latents "
                  f"(k={self.latents_per_skill}) from {latent_library_path}")

    # === SkillRL original methods (unchanged) ===
    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # Apply chat template (identical to SkillRL original)
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # === LATENT SKILL: split prompt, remove skills text, look up latent tokens ===
        before_str, skills_str, after_str, individual_skills = split_prompt_at_skills(prompt_chat_str)

        # Look up pre-computed latent tokens for each individual skill
        latent_tokens_list = []
        for skill_text in individual_skills:
            h = hash_skill(skill_text)
            if h in self.latent_library:
                latent_tokens_list.append(self.latent_library[h])  # (k, D)

        # Concatenate and pad latent tokens to fixed size for batching
        # Pad to fixed size for batching. 100 allows future skill library expansion.
        MAX_LATENT_TOKENS = 100
        hidden_size = latent_tokens_list[0].shape[-1] if latent_tokens_list else 3584

        if latent_tokens_list:
            latent_tokens = torch.cat(latent_tokens_list, dim=0)  # (N, D)
            num_latent = latent_tokens.shape[0]
            if num_latent < MAX_LATENT_TOKENS:
                padding = torch.zeros(MAX_LATENT_TOKENS - num_latent, hidden_size, dtype=latent_tokens.dtype)
                latent_tokens = torch.cat([latent_tokens, padding], dim=0)
            elif num_latent > MAX_LATENT_TOKENS:
                latent_tokens = latent_tokens[:MAX_LATENT_TOKENS]
                num_latent = MAX_LATENT_TOKENS
        else:
            latent_tokens = torch.zeros(MAX_LATENT_TOKENS, hidden_size)
            num_latent = 0

        # Tokenize prompt WITHOUT skills text
        prompt_without_skills = before_str + after_str
        prompt_ids_output = tokenizer(prompt_without_skills, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        # Tokenize before_str separately to get insertion point
        before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        before_length = before_ids.shape[0]

        # Tokenize response (identical to SkillRL original)
        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # Padding to max length (identical to SkillRL original)
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        # Position IDs (identical to SkillRL original)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Loss mask (identical to SkillRL original)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            # === SkillRL original fields (unchanged) ===
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            # === LATENT SKILL: extra fields (all fixed-size tensors for TensorDict compatibility) ===
            "before_length": torch.tensor(before_length, dtype=torch.long),
            "latent_tokens": latent_tokens,  # (MAX_LATENT_TOKENS, D) — padded to fixed size
            "num_latent_tokens": torch.tensor(num_latent, dtype=torch.long),  # actual count before padding
        }
