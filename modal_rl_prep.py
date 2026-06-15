"""RL prerequisite on Modal: bake the k=8 latent library into a vocab-expanded
Actor (the Stage-3 RL starting actor). One-shot, ~5 min on 1 GPU.

Reads:  /vol/out/composer_same7b_k8/...  NO — lib is at /vol/libs or we upload it.
Writes: /vol/rl_assets/actor_k8_expanded_vocab/  (the RL starting actor, ~15GB)

Usage:
  modal volume put latentskill <cluster>/latent_lib_same7b_k8/latent_skill_library.pt /rl_assets/lib_k8.pt
  modal run modal_rl_prep.py
"""
import subprocess
import modal

app = modal.App("latentskill-rl-prep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.1", "tokenizers==0.22.0",
        "huggingface-hub==0.35.3", "safetensors", "sentencepiece", "numpy",
    )
    .add_local_dir("src", "/root/app/src")
)
vol = modal.Volume.from_name("latentskill", create_if_missing=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3600)
def bake_k8():
    import os
    os.environ.update({"HF_HOME": "/vol/hf_cache", "HF_HUB_CACHE": "/vol/hf_cache/hub",
                       "PYTHONUNBUFFERED": "1"})
    lib = "/vol/rl_assets/lib_k8.pt"
    assert os.path.exists(lib), f"upload lib first: {lib}"
    out = "/vol/rl_assets/actor_k8_expanded_vocab"
    cmd = [
        "python", "-m", "src.expand_vocab_with_skills", "expand",
        "--sft-checkpoint", "Qwen/Qwen2.5-7B-Instruct",
        "--latent-library", lib,
        "--output-dir", out,
    ]
    print("=== BAKE:", " ".join(cmd))
    subprocess.run(cmd, cwd="/root/app", check=True)
    vol.commit()
    import os as _os
    print("=== baked files:", _os.listdir(out))
    return "bake k8 done -> /vol/rl_assets/actor_k8_expanded_vocab"


@app.local_entrypoint()
def main():
    print(bake_k8.remote())
