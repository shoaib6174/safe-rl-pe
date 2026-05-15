import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "repro_gonultas_isler_2024"))

from evaluation.evaluate import run_full_tournament


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else "results/checkpoints"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    best_cp, best_wr, all_results = run_full_tournament(checkpoint_dir, config)
    if best_cp is not None:
        print(f"\n=== FINAL RESULT ===")
        print(f"Best checkpoint: {best_cp}")
        print(f"Win rate: {best_wr:.4f}")
