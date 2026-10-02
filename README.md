# Mouse_CNN
This is the repo for the ongoing project of CNN MouseNet -- a convolutional neural network constrained by the architecture of the mouse visual cortex. This README is specific for the newer recurrent version of mousenet.


## Setup

Requires Python 3.9+ and `pip`. All model code works in a plain venv — no
conda, and has been modified so that it can run without need of using the allensdk.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Two entry points

- **`minimal_rnn/`** — a small, self-contained recurrent MouseNet
  (`RNNMouseNet`) and a CIFAR-10 trainer. Good as a tutorial / reference
  implementation.

  ```bash
  cd minimal_rnn && python cifar_trainer.py
  ```

- **`cmouse/` + `mouse_cnn/`** — the anatomy-constrained
  `MouseNetCompletePool` model. The network graph is loaded from a pickled
  file, so loading / training does **not** require the Allen SDK:

  ```python
  import sys; sys.path.extend(['cmouse', 'mouse_cnn'])
  import network
  from mousenet_complete_pool import MouseNetCompletePool
  net = network.load_network_from_pickle(
      'network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl'
  )
  model = MouseNetCompletePool(net, recurrent=True)
  ```

### Rebuilding the anatomy (optional)

If you need to regenerate the pickled network from CCF data instead of
loading a saved one:

```bash
pip install -r requirements-anatomy.txt
pip install -e mcm/mouse_connectivity_models
```

This path pulls in `allensdk` and requires the Allen CCF / voxel-model
data on disk (`ccf_201{5,6,7}/`, `voxel_model/`, `mouse_connectivity/`).
