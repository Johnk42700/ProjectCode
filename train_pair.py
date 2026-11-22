"""Train one speaker-listener pair and print its accuracy along training. Usage: python train_pair.py steps 3000 seed 0 """

import argparse
import math

from train import Config, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=10, help="number of checkpoints")
    args = ap.parse_args()
    cfg = Config(seed=args.seed, steps=args.steps, n_evals=args.evals, progress=f"pair s{args.seed}")
    res = run_experiment(cfg)
    print("final:")
    for k, v in res["final"].items():
        if k in ("step", "speaker_updates", "listener_updates") or (isinstance(v, float) and math.isnan(v)):
            continue                      # bookkeeping, or undefined for a single pair (synchrony)
        print(f"  {k:<22} {v:.3f}")
    print(f"runtime {res['runtime_sec']:.0f}s")


if __name__ == "__main__":
    main()
