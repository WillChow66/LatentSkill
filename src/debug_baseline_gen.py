"""Diagnose why vanilla Qwen2.5-7B scores ~0.7% on our no-skill harness while
the SkillRL paper reports 14.8% for the same base model. Prints the actual
prompt the env builds (no-skill) and the base model's generations, so we can
see whether it emits valid <action> tags or rambles / breaks format."""
import logging
from functools import partial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval_text_alfworld import make_env_config, generate_text, SKILLRL_ROOT

logging.basicConfig(level=logging.WARNING)

def main():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()
    from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    cfg = make_env_config(top_k=6, history_length=10, use_skills=False)  # NO SKILL
    alf_cfg = str(SKILLRL_ROOT / "agent_system" / "environments" / "env_package" / "alfworld" / "configs" / "config_tw.yaml")
    raw = build_alfworld_envs(alf_cfg, 42, env_num=1, group_n=1, is_train=False,
                              env_kwargs={"eval_dataset": "eval_in_distribution"},
                              resources_per_worker={"num_cpus": 1})
    em = AlfWorldEnvironmentManager(raw, partial(alfworld_projection), cfg)

    for ep in range(2):
        obs, info = em.reset(kwargs={})
        print(f"\n{'#'*90}\nEPISODE {ep}\n{'#'*90}")
        for step in range(4):
            instr = obs['text'][0]
            if step == 0:
                print(f"\n----- FULL PROMPT (len {len(instr)} chars) -----\n{instr}\n----- END PROMPT -----")
            resp, it, ot, el = generate_text(instr, model, tok, device,
                                             do_sample=True, temperature=0.4, top_p=1.0,
                                             max_new_tokens=512)
            print(f"\n[ep{ep} step{step}] out_tok={ot}\n  RAW RESPONSE: {resp!r}")
            obs, rew, done, sinfo = em.step([resp])
            parsed = sinfo[0].get("action", sinfo[0]) if sinfo and sinfo[0] else None
            print(f"  PARSED/valid_action -> {str(parsed)[:120]}  done={done[0]}")
            if done[0]: break
    raw.close()

if __name__ == "__main__":
    main()
