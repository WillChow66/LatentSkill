"""Garbage-latent control: build a latent library of pure Gaussian noise,
norm-matched per latent position to a reference library.

Purpose: 4th cell of the skill-representation control suite. The UNTRAINED
variant (raw 3B encoder + random ql/proj) still READS the skill text, so its
latents may carry real skill information. This control destroys the content
while preserving (a) the library schema, (b) per-position L2 norms, and
(c) the prompt scaffold at eval time. If RANDOM ≈ no-skill (~2%) while
UNTRAINED ≈ 23.6%, the UNTRAINED gap is attributable to information flowing
through the pretrained encoder, not to scaffold or "any vectors" effects.

Usage:
  python -m src.make_random_latent_library \
      --reference /path/to/latent_lib_untrained/latent_skill_library.pt \
      --output    /path/to/latent_lib_random/latent_skill_library.pt \
      --seed 123
"""

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True,
                        help="Reference latent_skill_library.pt (norms copied from here)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123,
                        help="Noise seed (deliberately != encode seed 42)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ref = torch.load(args.reference, map_location="cpu", weights_only=False)
    lib = ref["latent_library"]
    logger.info(f"Reference: {len(lib)} skills, k={ref['latents_per_skill']}, "
                f"variant={ref.get('variant')}")

    gen = torch.Generator().manual_seed(args.seed)
    random_lib = {}
    for h, lat in lib.items():
        lat = lat.float()  # (k, D_actor)
        noise = torch.randn(lat.shape, generator=gen)
        # Rescale each latent position to the reference vector's L2 norm
        ref_norms = lat.norm(dim=-1, keepdim=True)            # (k, 1)
        noise = noise / noise.norm(dim=-1, keepdim=True) * ref_norms
        random_lib[h] = noise.to(torch.bfloat16)

    # Sanity: norms match, content does not
    h0 = next(iter(lib))
    cos = torch.nn.functional.cosine_similarity(
        lib[h0].float().flatten(), random_lib[h0].float().flatten(), dim=0).item()
    logger.info(f"Sanity skill[{h0}]: ref_norm={lib[h0].float().norm():.2f} "
                f"rand_norm={random_lib[h0].float().norm():.2f} cossim={cos:.4f} (expect ~0)")

    out = dict(ref)  # copy all schema keys (skill_index, latents_per_skill, dims, ...)
    out["latent_library"] = random_lib
    out["variant"] = "RANDOM"
    out["noise_seed"] = args.seed
    out["reference_library"] = str(args.reference)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    logger.info(f"Saved RANDOM library → {out_path}")


if __name__ == "__main__":
    main()
