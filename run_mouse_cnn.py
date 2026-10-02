#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "cmouse"), str(ROOT / "mouse_cnn")]

import network
from mousenet_complete_pool import MouseNetCompletePool


PICKLES = {
    "feedforward": "network_complete_updated_number(3,64,64).pkl",
    "recurrent": "network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl",
}


def main():
    parser = argparse.ArgumentParser(description="Run a MouseNet forward-pass smoke test.")
    parser.add_argument("--mode", choices=PICKLES, default="recurrent")
    parser.add_argument("--pickle", type=Path)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pickle_path = args.pickle or ROOT / PICKLES[args.mode]
    net = network.load_network_from_pickle(str(pickle_path))
    model = MouseNetCompletePool(net, recurrent=args.mode == "recurrent").to(args.device)
    model.eval()
    images = torch.rand(args.batch_size, 3, 64, 64, device=args.device)

    with torch.inference_mode():
        outputs, _ = model(images, n_steps=args.steps) if args.mode == "recurrent" else model(images)

    print(f"mode={args.mode} pickle={pickle_path.name} device={args.device} output_shape={tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
