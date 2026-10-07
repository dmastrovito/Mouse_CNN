#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "cmouse"), str(REPO / "cmouse" / "exps" / "cifar")]

import network
import mousenet_model

NETWORK_FILES = {
    True: REPO / "models" / "recurrent_mousenet_inputsize64_ccf_2017.pkl",
    False: REPO / "models" / "mousenet_inputsize64_ccf_2017.pkl",
}


def build_mousenet(recurrent=True, device=None, mask=3, weights=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = network.load_network_from_pickle(str(NETWORK_FILES[recurrent]))
    model = mousenet_model.mousenet(net, recurrent=recurrent, device=device, mask=mask).to(device)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MouseNet and print a summary.")
    parser.add_argument("--mode", choices=["recurrent", "feedforward"], default="recurrent")
    parser.add_argument("--weights", help="trained weights (.pth) saved by cmouse/train_mousenet.py")
    parser.add_argument("--device", help="e.g. cpu or cuda (default: GPU if available)")
    args = parser.parse_args()

    model = build_mousenet(recurrent=args.mode == "recurrent", device=args.device, weights=args.weights)
    layers = {}
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            layers[type(module).__name__] = layers.get(type(module).__name__, 0) + 1
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.mode} MouseNet: LGNd + {len(model.regions)} cortical areas, layers {layers}, "
          f"{n_params:,} trainable parameters")
