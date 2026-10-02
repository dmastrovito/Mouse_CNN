from collections import defaultdict
import torch
from torch import nn

REGIONS = ['LGNd', 'VISp4', 'VISp23', 'VISp5', 'VISrl4', 'VISrl23', 'VISrl5', 'VISli4', 'VISli23',
           'VISli5', 'VISpl4', 'VISpl23', 'VISpl5', 'VISal4', 'VISal23', 'VISal5', 'VISl4', 'VISl23',
           'VISl5', 'VISpor4', 'VISpor23', 'VISpor5']
'''
CONNECTIVITY = {  # region: (input_regions, num_channels, resolution)
    'LGNd': ([], 16, 32),
    'VISp4': (['LGNd', 'VISp23', 'VISrl4', 'VISli4', 'VISpor4', 'VISpl4', 'VISal4', 'VISl4'], 32, 16),
    'VISp23': (['VISrl4', 'VISp4', 'VISli4', 'VISp5', 'VISpor4', 'VISpl4', 'VISal4', 'VISl4'], 16, 8),
    'VISp5': (['VISp23', 'VISrl4', 'VISli4', 'VISpor4', 'VISpl4', 'VISal4', 'VISl4'], 16, 8),
    'VISrl4': (['VISp4', 'VISp23', 'VISp5', 'VISrl23', 'VISpor4'], 16, 8),
    'VISrl23': (['VISrl4', 'VISrl5', 'VISpor4'], 16, 8),
    'VISrl5': (['VISrl23', 'VISpor4'], 16, 8),
    'VISli4': (['VISp4', 'VISp23', 'VISp5', 'VISli23', 'VISpor4'], 16, 8),
    'VISli23': (['VISli4', 'VISli5', 'VISpor4'], 16, 8),
    'VISli5': (['VISli23', 'VISpor4'], 16, 8),
    'VISpl4': (['VISp4', 'VISp23', 'VISp5', 'VISpl23', 'VISpor4'], 16, 8),
    'VISpl23': (['VISpl4', 'VISpl5', 'VISpor4'], 16, 8),
    'VISpl5': (['VISpl23', 'VISpor4'], 16, 8),
    'VISal4': (['VISp4', 'VISp23', 'VISp5', 'VISal23', 'VISpor4'], 16, 8),
    'VISal23': (['VISal4', 'VISal5', 'VISpor4'], 16, 8),
    'VISal5': (['VISal23', 'VISpor4'], 16, 8),
    'VISl4': (['VISp4', 'VISp23', 'VISp5', 'VISl23', 'VISpor4'], 16, 8),
    'VISl23': (['VISl4', 'VISl5', 'VISpor4'], 16, 8),
    'VISl5': (['VISl23', 'VISpor4'], 16, 8),
    'VISpor4': (REGIONS[1:-1], 64, 8),
    'VISpor23': (['VISpor4', 'VISpor5'], 128, 4),
    'VISpor5': (['VISpor23'], 256, 2),
}
'''
CONNECTIVITY = {  # region: (input_regions, num_channels, resolution)
    'LGNd': ([], 20, 32),
    'VISp4': (['LGNd', 'VISp23', 'VISrl4', 'VISli4', 'VISpor4', 'VISpl4', 'VISal4', 'VISl4'], 106, 16),
    'VISp23': (['VISrl4', 'VISp4', 'VISli4', 'VISp5', 'VISpor4', 'VISpl4', 'VISal4', 'VISl4'], 169, 8),
    'VISp5': (['VISp23', 'VISrl4', 'VISli4', 'VISpor4', 'VISpl4', 'VISal4', 'VISl4'], 131, 8),
    'VISrl4': (['VISp4', 'VISp23', 'VISp5', 'VISrl23', 'VISpor4'], 56, 8),
    'VISrl23': (['VISrl4', 'VISrl5', 'VISpor4'], 88, 8),
    'VISrl5': (['VISrl23', 'VISpor4'], 74, 8),
    'VISli4': (['VISp4', 'VISp23', 'VISp5', 'VISli23', 'VISpor4'], 21, 8),
    'VISli23': (['VISli4', 'VISli5', 'VISpor4'], 37, 8),
    'VISli5': (['VISli23', 'VISpor4'], 45, 8),
    'VISpl4': (['VISp4', 'VISp23', 'VISp5', 'VISpl23', 'VISpor4'], 15, 8),
    'VISpl23': (['VISpl4', 'VISpl5', 'VISpor4'], 70, 8),
    'VISpl5': (['VISpl23', 'VISpor4'], 78, 8),
    'VISal4': (['VISp4', 'VISp23', 'VISp5', 'VISal23', 'VISpor4'], 37, 8),
    'VISal23': (['VISal4', 'VISal5', 'VISpor4'], 61, 8),
    'VISal5': (['VISal23', 'VISpor4'], 62, 8),
    'VISl4': (['VISp4', 'VISp23', 'VISp5', 'VISl23', 'VISpor4'], 60, 8),
    'VISl23': (['VISl4', 'VISl5', 'VISpor4'], 87, 8),
    'VISl5': (['VISl23', 'VISpor4'], 81, 8),
    'VISpor4': ([r for r in REGIONS[1:-1] if r != 'VISpor4'], 23, 8),
    'VISpor23': (['VISpor4', 'VISpor5'], 119, 4),
    'VISpor5': (['VISpor23'], 118, 2),
}

def check_symmetry():
    connectivity = defaultdict(list)
    for k in REGIONS:
        for l in CONNECTIVITY[k][0]:
            connectivity[l].append(k)
    for k in REGIONS:
        if tuple(sorted(connectivity[k])) != tuple(sorted(CONNECTIVITY[k][0])):
            print(k, tuple(sorted(connectivity[k])))
            print(k, tuple(sorted(CONNECTIVITY[k][0])))


class LGN(nn.Module):
    def __init__(self):
        super(LGN, self).__init__()
        # Output channel count comes from CONNECTIVITY so it stays in sync with downstream Connection modules.
        nc = CONNECTIVITY['LGNd'][1]
        self.conv = nn.Sequential(nn.ReLU(), nn.Conv2d(3, nc, 3, padding=1))
        self.state1 = None

    def reset(self, bs):
        pass

    def forward(self, x):
        x = self.conv(x)
        self.state1 = x


class Connection(nn.Module):
    def __init__(self, region):
        super(Connection, self).__init__()
        self.in_regions, nc, res = CONNECTIVITY[region]

        self.init_state = nn.Parameter(torch.zeros((1, nc, res, res)), requires_grad=True)
        self.state1, self.state2 = None, None

        self.decay = nn.Parameter(torch.zeros((1, nc, res, res)), requires_grad=True)

        in_dims = {k:CONNECTIVITY[k][1] for k in self.in_regions}
        in_res = {k:CONNECTIVITY[k][2] for k in self.in_regions}

        self.convs = nn.ModuleDict({
            k: (
                nn.Sequential(nn.ReLU(), nn.Conv2d(in_dims[k], nc, 3, padding=1)) if in_res[k] == res else
                nn.Sequential(nn.ReLU(), nn.Conv2d(in_dims[k], nc, 3, 2, padding=1)) if in_res[k] > res else
                nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(in_dims[k], nc, 3, 2, padding=1, output_padding=1))
            )  # For now, adjacent regions should differ in resolution by at most a factor of 2
            for k in self.in_regions
        })

    def reset(self, bs):
        self.state1, self.state2 = self.init_state.repeat(bs, 1, 1, 1), self.init_state.repeat(bs, 1, 1, 1)

    def step(self):
        self.state1 = self.state2

    def forward(self, inputs):
        x = [self.convs[k](inputs[k]) for k in self.in_regions]
        self.state2 = self.state2 * self.decay.sigmoid() + sum(x) / len(x)


class RNNMouseNet(nn.Module):
    OUTPUT_REGION = 'VISpor5'

    def __init__(self, num_classes=10):
        super(RNNMouseNet, self).__init__()
        self.connections = nn.ModuleDict({k: LGN() if k == 'LGNd' else Connection(k) for k in REGIONS})
        _, out_nc, out_res = CONNECTIVITY[self.OUTPUT_REGION]
        self.out = nn.Linear(out_nc * out_res * out_res, num_classes)

    def forward(self, x, ts=6):
        for k in REGIONS:
            self.connections[k].reset(x.shape[0])

        self.connections['LGNd'](x)
        for _ in range(ts):
            states = {k: self.connections[k].state1 for k in REGIONS}
            for k in REGIONS[1:]:  # TODO: find out if this is already lazy or can be made parallel
                self.connections[k](states)
            for k in REGIONS[1:]:
                self.connections[k].step()
        out = self.connections[self.OUTPUT_REGION].state1
        out = out.view(out.shape[0], -1)
        return self.out(out)
    
