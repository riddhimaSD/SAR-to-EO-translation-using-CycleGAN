"""
train.py
--------
CLI entry-point for training the SAR-to-EO CycleGAN model.

Usage:
    python train.py --config configs/config_part_a.yaml
    python train.py --config configs/config_part_b.yaml
    python train.py --config configs/config_part_c.yaml

Optional overrides:
    python train.py --config configs/config_part_a.yaml --num_epochs 20 --lr 1e-4
"""

import argparse
import yaml

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAR-to-EO CycleGAN')
    parser.add_argument('--config',     type=str, required=True,  help='Path to YAML config file')
    parser.add_argument('--num_epochs', type=int, default=None,   help='Override num_epochs from config')
    parser.add_argument('--lr',         type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--data_root',  type=str, default=None,   help='Override dataset root path')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.lr is not None:
        config['lr'] = args.lr
    if args.data_root is not None:
        config['data_root'] = args.data_root

    print(f"[Config] band_config={config['band_config']} | epochs={config['num_epochs']} | lr={config['lr']}")

    # Run training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
