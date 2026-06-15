"""Modal port of the B-arm LoRA SFT chain (train + encode in one container).

Why: DeltaAI group allocation exhausted (QOSGrpBillingMinutes) — LoRA k=6/8/16/32
chains stuck. This runs them on Modal 4×A100-80GB (same code, x86 wheels).

Pipeline per k (mirrors cluster scripts/composer_same7b_lora_kN.sbatch +
encode_lib_same7b_lora_kN.sbatch, LatentMem-exact LoRA recipe):
  1. accelerate launch (4 GPU DDP) src.train_composer_only
     --use-lora r16 a32 dropout0.1 targets q_proj,v_proj, seed 42, 4 epochs
  2. encode 44 skills from the merged ckpt -> latent_skill_library.pt
  3. persist to Volume: ckpt (15GB), ql, lib.pt

Usage:
  modal run modal_lora_sft.py --k 8 --smoke true   # 40-sample smoke first!
  modal run modal_lora_sft.py --k 6                # full run
  modal run modal_lora_sft.py --all-ks true        # k=6,8,16,32 in parallel

Outputs land in Volume 'latentskill' under /out/composer_same7b_lora_k{K}/.
Fetch lib back:  modal volume get latentskill out/composer_same7b_lora_k8/latent_skill_library.pt .
"""

import subprocess
import modal

app = modal.App("latentskill-lora-sft")

# x86 wheels — pinned to match the cluster stack where it matters.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "tokenizers==0.22.0",
        "accelerate==1.11.0",
        "peft==0.17.1",
        "huggingface-hub==0.35.3",
        "pandas",
        "pyarrow",
        "safetensors",
        "sentencepiece",
    )
    .add_local_dir("src", "/root/app/src")
    .add_local_file(
        "SkillRL/memory_data/alfworld/claude_style_skills.json",
        "/root/app/SkillRL/memory_data/alfworld/claude_style_skills.json",
    )
)

vol = modal.Volume.from_name("latentskill", create_if_missing=True)

TRAIN_ENV = {
    "HF_HOME": "/vol/hf_cache",
    "HF_HUB_CACHE": "/vol/hf_cache/hub",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/vol": vol},
    timeout=5 * 3600,
)
def train_and_encode(k: int, smoke: bool = False):
    import os
    os.environ.update(TRAIN_ENV)
    os.makedirs("/vol/out", exist_ok=True)

    out_dir = f"/vol/out/composer_same7b_lora_k{k}" + ("_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    parquet = "/vol/data/train-00000-of-00001.parquet"
    assert os.path.exists(parquet), f"upload the SFT parquet to the volume first: {parquet}"

    train_cmd = [
        "accelerate", "launch", "--num_processes", "4", "--mixed_precision", "bf16",
        "-m", "src.train_composer_only",
        "--composer-path", "Qwen/Qwen2.5-7B-Instruct",
        "--actor-path", "Qwen/Qwen2.5-7B-Instruct",
        "--sft-data", parquet,
        "--output-dir", out_dir,
        "--epochs", "1" if smoke else "4",
        "--composer-lr", "1e-4", "--latent-lr", "5e-3",
        "--weight-decay", "0.01", "--warmup-ratio", "0.1",
        "--adam-beta1", "0.9", "--adam-beta2", "0.95",
        "--max-length", "4096", "--latents-per-skill", str(k), "--seed", "42",
        "--use-lora", "--lora-r", "16", "--lora-alpha", "32",
        "--lora-dropout", "0.1", "--lora-targets", "q_proj,v_proj",
        "--save-every", "10000", "--log-every", "20",
        "--no-wandb",
    ]
    if smoke:
        train_cmd += ["--max-samples", "40"]

    print(f"=== [k={k}] TRAIN: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, cwd="/root/app", check=True)
    vol.commit()

    # epoch number matches --epochs above
    epoch = 1 if smoke else 4
    import glob
    qls = sorted(glob.glob(f"{out_dir}/query_latents_step*.pt"),
                 key=lambda p: int(p.split("step")[-1].split(".")[0]))
    assert qls, f"no query_latents saved in {out_dir}"
    encode_cmd = [
        "python", "-m", "src.encode_latent_library",
        "--composer-ckpt", f"{out_dir}/composer_epoch{epoch}",
        "--query-latents", qls[-1],
        "--actor-path", "Qwen/Qwen2.5-7B-Instruct",
        "--output-dir", out_dir,
        "--variant", "TRAINED",
        "--latents-per-skill", str(k),
        "--device", "cuda",
    ]
    print(f"=== [k={k}] ENCODE: {' '.join(encode_cmd)}")
    subprocess.run(encode_cmd, cwd="/root/app", check=True)
    vol.commit()
    print(f"=== [k={k}] DONE -> {out_dir}/latent_skill_library.pt")
    return f"k={k} ok ({'smoke' if smoke else 'full'})"


@app.local_entrypoint()
def main(k: int = 8, smoke: bool = False, all_ks: bool = False):
    if all_ks:
        ks = [6, 8, 16, 32]
        print(f"Launching {ks} in parallel ...")
        for r in train_and_encode.starmap([(kk, False) for kk in ks]):
            print(r)
    else:
        print(train_and_encode.remote(k, smoke))
