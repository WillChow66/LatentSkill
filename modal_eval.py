"""Modal port of the ALFWorld latent-skill eval (cluster quota exhausted).

MUST be calibrated before trusting: run the mainbatch_k8 lib (known 52.9% on
DeltaAI, eval noise ±2pp) and require agreement within ±4pp. Parser-watershed
lesson: never switch measurement environments without an anchor check.

Image = code-only stage (.modal_stage: src/ + SkillRL code, alfworld vendored);
game files (valid_seen 72MB + logic) live in the Volume at /vol/alfworld_data.

Usage:
  modal run modal_eval.py --lib libs/mainbatch_k8.pt --tag calib_mainbatch_k8   # calibration
  modal run modal_eval.py --lora-batch                                          # k=8/16/32 LoRA evals
Results -> Volume /results/*.json ; fetch: modal volume get latentskill results/<f>.json .
"""

import subprocess
import modal

app = modal.App("latentskill-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "tokenizers==0.22.0",
        "accelerate==1.11.0",
        "huggingface-hub==0.35.3",
        "ray==2.50.0",
        "textworld==1.7.0",
        "fast-downward-textworld",
        "omegaconf==2.3.0",
        "gymnasium==0.29.1",
        "pandas",
        "pyarrow",
        "safetensors",
        "sentencepiece",
        "sentence-transformers",
        "faiss-cpu",
        # vendored alfworld env imports (module-level, even in TextWorld mode)
        "torchvision==0.21.0",
        "opencv-python-headless",
        "gym",
        "h5py",
        "hydra-core",
        "networkx",
        "pillow",
        "pycocotools",
        "requests",
        "tensordict",
        "termcolor",
        "tqdm",
        "pyyaml",
    )
    .add_local_dir(".modal_stage", "/root/app")
)

vol = modal.Volume.from_name("latentskill", create_if_missing=True)

EVAL_ENV = {
    "HF_HOME": "/vol/hf_cache",
    "HF_HUB_CACHE": "/vol/hf_cache/hub",
    "ALFWORLD_DATA": "/vol/alfworld_data",
    "PYTHONPATH": "/root/app:/root/app/SkillRL:"
                  "/root/app/SkillRL/agent_system/environments/env_package/alfworld",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "RAY_DISABLE_IMPORT_WARNING": "1",
}


@app.function(image=image, gpu="L40S", cpu=8, volumes={"/vol": vol}, timeout=8 * 3600)
def eval_lib(lib: str, tag: str, episodes: int = 140):
    import os
    os.environ.update(EVAL_ENV)
    os.makedirs("/vol/results", exist_ok=True)
    lib_path = f"/vol/{lib}"
    assert os.path.exists(lib_path), f"lib not found in volume: {lib_path}"
    out = f"/vol/results/{tag}.json"

    cmd = [
        "python", "-m", "src.eval_latent_alfworld",
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--latent-library", lib_path,
        "--num-episodes", str(episodes), "--max-steps", "50",
        "--do-sample", "--temperature", "0.4", "--top-p", "1.0",
        "--history-length", "10", "--top-k", "6",
        "--output", out,
    ]
    print(f"=== [{tag}] EVAL: {' '.join(cmd)}")
    subprocess.run(cmd, cwd="/root/app", check=True)
    vol.commit()

    import json
    d = json.load(open(out))
    n = len(d["episodes"]); r = d["overall_success_rate"]
    msg = f"[{tag}] {round(r*n)}/{n} ({r*100:.1f}%) tok={d['avg_tokens']:.0f}"
    print("===", msg)
    return msg


@app.local_entrypoint()
def main(lib: str = "", tag: str = "", episodes: int = 140, lora_batch: bool = False):
    if lora_batch:
        jobs = [(f"out/composer_same7b_lora_k{k}/latent_skill_library.pt",
                 f"modal_lorabatch_k{k}", 140) for k in (8, 16, 32)]
        for r in eval_lib.starmap(jobs):
            print(r)
    else:
        assert lib and tag, "pass --lib <vol path> --tag <name>, or --lora-batch"
        print(eval_lib.remote(lib, tag, episodes))
