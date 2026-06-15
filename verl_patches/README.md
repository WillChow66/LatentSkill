# verl / SkillRL patches (our modifications to the SkillRL fork)

`SkillRL/` is a nested git repo (excluded from this repo via .gitignore), so our
edits to it can't be tracked directly. These are copies of the 3 files we changed,
mirroring their path under `SkillRL/`. To apply: copy each back to the same path
inside your SkillRL clone.

| file (path under SkillRL/) | change |
|---|---|
| agent_system/environments/env_package/alfworld/projection.py | **Robust action parser** — accept `[action]`/`<action>`/malformed brackets (the parser-fix watershed; strict-only parser deflated format-unstable models). |
| agent_system/memory/skills_only_memory.py | latent_token_mode: `latents_per_skill` param (was hardcoded k=2 `_a/_b`); suffix matches expand_vocab `_suffix_for`. |
| agent_system/environments/env_manager.py | Plumb `latents_per_skill` into SkillsOnlyMemory construction. |

Also note (RL-stage, not copied here — larger): verl/workers/* + trainer/ppo/ray_trainer.py
carry the X2 dynamic-skill hooks; see CLAUDE.md "Modified verl Files".
