"""Self-play training entry point.

Usage:
    python scripts/train_self_play.py
    python scripts/train_self_play.py self_play.n_phases=5 self_play.timesteps_per_phase=100000
    python scripts/train_self_play.py wandb.mode=disabled seed=123
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig, OmegaConf

from training.self_play import alternating_self_play


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Self-play config:\n{OmegaConf.to_yaml(cfg)}")

    result = alternating_self_play(cfg)

    # Print summary
    history = result["history"]
    print("\n=== Self-Play Summary ===")
    for i, (cr, er, el) in enumerate(zip(
        history["capture_rate"],
        history["escape_rate"],
        history["mean_episode_length"],
    )):
        alerts = history["health_alerts"][i] if i < len(history["health_alerts"]) else []
        alert_str = f"  ALERTS: {alerts}" if alerts else ""
        print(f"Phase {i:2d}: capture={cr:.2f}  escape={er:.2f}  ep_len={el:.0f}{alert_str}")

    # Final capture rate
    final_cr = history["capture_rate"][-1] if history["capture_rate"] else 0
    print(f"\nFinal capture rate: {final_cr:.2f}")
    if abs(final_cr - 0.5) < cfg.self_play.convergence_threshold:
        print("Convergence criterion met (approximate NE)")
    else:
        print("Convergence criterion NOT met")


if __name__ == "__main__":
    main()
