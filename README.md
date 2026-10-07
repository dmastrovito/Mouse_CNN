# Recurrent MouseNet

A recurrent version of MouseNet: a convolutional neural network whose
architecture is constrained by the anatomy of mouse visual cortex. Areas LGNd,
VISp, VISl, VISrl, VISli, VISpl, VISal and VISpor are each modelled as layers
4, 2/3 and 5.

- Feedforward connections are `Conv2d` layers whose output is added to the
  target area.
- Feedback (recurrent) connections are `ConvTranspose2d` layers whose output
  multiplicatively gates the target area's state.
- Connection weights are sparse, with fixed Gaussian masks whose size and
  density come from anatomy.

The network structure was derived from the Allen Mouse Brain Common
Coordinate Framework (CCF 2017 annotation) and the Allen voxel-scale
connectivity model.

MouseNet is described in:

> Shi J, Tripp B, Shea-Brown E, Mihalas S, Buice MA (2022). MouseNet: A biologically
> constrained convolutional neural network model for the mouse visual cortex.
> *PLOS Computational Biology* 18(9): e1010427. https://doi.org/10.1371/journal.pcbi.1010427

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.13.9, PyTorch 2.14.1 and torchvision 0.29.1.

## Training on CIFAR-10

```bash
python cmouse/train_mousenet.py                      # recurrent (the default)
python cmouse/train_mousenet.py --mode feedforward   # feedforward-only network
```

- CIFAR-10 is downloaded to `data/` on first use; images are resized to 64x64.
- Checkpoints (every 100 batches) and loss/accuracy plots are written to
  `results/<network name>_SGD_scheduledLR_BN2d_relu_run_1/`. Rerunning resumes
  from the latest checkpoint there.
- Each recurrent batch runs a random 16-19 steps, so a GPU is strongly
  recommended; the script uses one automatically if available.
- Settings are in `cmouse/exps/cifar/train_config.py`: 1000 epochs, batch size
  32, SGD with learning rate 0.01 and momentum 0.9. The learning rate is
  multiplied by 0.8 every 10 epochs, and training stops early once validation
  accuracy is at least 90% and has not improved for 10 epochs.

## Network files

| File | Connections | Use with |
|---|---|---|
| `models/recurrent_mousenet_inputsize64_ccf_2017.pkl` | 48 feedforward + 47 feedback | `recurrent=True` |
| `models/mousenet_inputsize64_ccf_2017.pkl` | 48 feedforward | `recurrent=False` |

Both take 64x64 RGB input.

## Building the model

`build_mousenet.py`, at the repository root, builds the model without the
training pipeline:

```python
import sys
sys.path.insert(0, "/path/to/Mouse_CNN")  # not needed when working in the repository root
import torch
from build_mousenet import build_mousenet

model = build_mousenet()                   # recurrent MouseNet (the default), untrained
device = next(model.parameters()).device

images = torch.rand(8, 3, 64, 64, device=device)
model.reset(8, device)                     # required before every forward pass, with the batch size
logits = model(images, n_steps=16)         # shape (8, 10)
```

`build_mousenet(recurrent=True, device=None, mask=3, weights=None)`:

- `recurrent`: `True` (default) builds the recurrent network (48 feedforward +
  47 feedback connections); `False` builds the feedforward-only network.
- `device`: defaults to a GPU if one is available, otherwise the CPU.
- `mask`: how the anatomical Gaussian masks are applied to the connection
  weights; the default (3) is what training uses.
- `weights`: path to a `.pth` file saved by `cmouse/train_mousenet.py`; loads
  the trained weights (including the masks).

The connection masks are sampled randomly each time a model is built. Set
`numpy.random.seed` and `torch.manual_seed` first if you need reproducible
untrained models; trained weights restore the masks they were trained with.

Call `model.reset(batch_size, device)` before each forward pass; it sets every
area back to its learned initial state. Training uses 16-19 steps for the
recurrent model and 6 for the feedforward model. Train the model before using
`model.eval()`: an untrained model in evaluation mode can produce NaN, because
its BatchNorm layers have not yet collected statistics.

To check the setup, `python build_mousenet.py` builds the model and prints a
summary (`--mode feedforward` and `--weights FILE` are also accepted).

## Rebuilding the networks (requires AllenSDK and Allen data)

`construct_mousenet_architecture.py`, `cmouse/anatomy.py` and `mouse_cnn/`
regenerate the network files from anatomy. This is not needed to train or use
the model, and has not been tested in this copy of the code. It requires:

- the `mcmodels` package from
  [mouse_connectivity_models](https://github.com/AllenInstitute/mouse_connectivity_models),
  which installs AllenSDK, plus `scipy`, `scikit-image` and `scikit-learn`;
- the Allen CCF 2017 annotation and voxel connectivity model data in
  `ccf_2017/` and `connectivity/` at the repository root (not included).

Run `python construct_mousenet_architecture.py` from the repository root. It
builds the recurrent network; set `recurrent = False` in the script to build
the feedforward one. Output goes to `models/`.
