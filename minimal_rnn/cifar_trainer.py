import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm


# Known locations to probe for an already-downloaded CIFAR-10 dataset.
# Checked in order; first one containing cifar-10-batches-py/ wins.
_HERE = Path(__file__).resolve().parent
_DATA_SEARCH_PATHS = (
    Path(os.environ["MOUSECNN_DATA_DIR"]) if os.environ.get("MOUSECNN_DATA_DIR") else None,
    Path.cwd() / "data",
    _HERE / "data",
    _HERE.parent / "data",
)


def resolve_data_root(explicit=None):
    """Return a directory to use as torchvision's CIFAR-10 ``root``.

    If ``explicit`` is given it's returned verbatim. Otherwise known
    locations are probed for an existing cifar-10-batches-py/ folder;
    if none is found, falls back to <repo>/data (created on demand) so
    torchvision can download into it.
    """
    if explicit is not None:
        return str(explicit)
    for candidate in _DATA_SEARCH_PATHS:
        if candidate is not None and (candidate / "cifar-10-batches-py").is_dir():
            return str(candidate)
    fallback = _HERE.parent / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


class Trainer:
    def __init__(self, model, bs=256, epochs=60, save_path=None, silent=False,
                 device=None, data_root=None, download=True):
        self.epochs = epochs
        self.save_path = save_path

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)

        self.data_loader_train, self.data_loader_val = self.make_loaders(
            bs,
            pin_memory=self.device.type == "cuda",
            data_root=data_root,
            download=download,
        )

        make_optimizers = getattr(self.model, 'make_optimizers', None)
        if callable(make_optimizers):
            self.optimizers, self.schedulers = make_optimizers(bs, epochs)
        else:
            self.optimizers = [torch.optim.Adam(self.model.parameters())]
            self.schedulers = [torch.optim.lr_scheduler.MultiStepLR(
                self.optimizers[-1], [2*epochs//3, 9*epochs//10], gamma=0.2
            )]

        self.criterion = torch.nn.CrossEntropyLoss()

        self.test_stats = None
        self.silent = silent

    @staticmethod
    def make_loaders(bs, pin_memory=False, num_workers=None, data_root=None, download=True):
        if num_workers is None:
            num_workers = 8 if pin_memory else 2
        root = resolve_data_root(data_root)
        train_augmentator = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        dataset_train = datasets.CIFAR10(transform=train_augmentator, root=root, train=True, download=download)
        dataset_val = datasets.CIFAR10(transform=transforms.ToTensor(), root=root, train=False, download=download)
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)
        data_loader_train = DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        data_loader_val = DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * bs),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        return data_loader_train, data_loader_val

    def train(self):
        for epoch in range(1, self.epochs + 1):
            lrs = [o.param_groups[0]['lr'] for o in self.optimizers if 'lr' in o.param_groups[0]]
            train_stats = self.train_one_epoch()
            self.test_stats = self.evaluate()
            self.test_stats = tuple(map(lambda x: sum(x) / len(x), self.test_stats))
            if not self.silent:
                print('Mean train loss during epoch %d: %.6f' % (epoch, sum(train_stats) / len(train_stats)), lrs)
                print('Val stats: ' + ('%.4f ' * len(self.test_stats)) % self.test_stats)
        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)
        return self

    def train_one_epoch(self):
        self.model.train()
        loss_values = []

        # for (samples, targets),_ in tqdm(zip(self.data_loader_train,[1,2,3,4])):
        for samples, targets in tqdm(self.data_loader_train, disable=self.silent):
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            loss_values.append(self.train_step(samples, targets))

        for sched in self.schedulers: sched.step()

        print_stats = getattr(self.model, 'print_stats', None)
        if callable(print_stats): print_stats(self.silent)

        return loss_values

    def train_step(self, samples, targets):
        outputs = self.model(samples)
        loss = self.criterion(outputs, targets)

        self.model.zero_grad()
        loss.backward()
        for opt in self.optimizers: opt.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        stats = []

        # for (images, target), _ in tqdm(zip(self.data_loader_val, [1, 2, 3, 4])):
        for images, target in tqdm(self.data_loader_val, disable=self.silent):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            new_stats = self.eval_step(images, target)

            if not stats:
                stats = [[] for _ in new_stats]

            for a,b in zip(stats, new_stats):
                a.append(b)

        print_stats = getattr(self.model, 'print_stats', None)
        if callable(print_stats): print_stats(self.silent)

        return stats

    def eval_step(self, images, target):
        output = self.model(images)
        loss = self.criterion(output, target)
        acc = (output.argmax(-1) == target).float().mean().item()
        return loss.item(), acc


if __name__ == '__main__':
    from rnn_mousenet import RNNMouseNet
    Trainer(RNNMouseNet(), bs=512).train()
