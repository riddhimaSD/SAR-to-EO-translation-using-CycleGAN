"""
evaluate.py  (root)
-------------------
CLI entry-point for evaluating a trained SAR-to-EO CycleGAN model on the test set.

Usage:
    python evaluate.py --config configs/config_part_a.yaml
    python evaluate.py --config configs/config_part_b.yaml
    python evaluate.py --config configs/config_part_c.yaml

Optional overrides:
    python evaluate.py --config configs/config_part_a.yaml --data_root /path/to/data
    python evaluate.py --config configs/config_part_c.yaml --checkpoint checkpoints/G_best.pt
"""

import argparse
import yaml

from src.evaluate import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SAR-to-EO CycleGAN on test set')
    parser.add_argument('--config',     type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--data_root',  type=str, default=None,  help='Override dataset root path')
    parser.add_argument('--checkpoint', type=str, default=None,  help='Override checkpoint path (default: G_best.pt)')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data_root is not None:
        config['data_root'] = args.data_root
    if args.checkpoint is not None:
        import os
        config['checkpoint_dir'] = os.path.dirname(args.checkpoint)

    print(f"[Config] band_config={config['band_config']} | test_samples={config.get('test_samples', 400)}")

    evaluator = Evaluator(config)
    results = evaluator.run()

    print("\n--- Evaluation Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
