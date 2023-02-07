from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# all types of patches
TYPES = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']


@dataclass
class Patch:
    name: str
    type: str
    data: np.ndarray

    def __post_init__(self):
        assert self.data.shape == (65, 65), 'Patch must be 65 x 65'

    def display(self):
        import matplotlib.pyplot as plt
        plt.style.use('classic')
        plt.imshow(self.data, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()


class Sequence:
    def __init__(self, seq_path: str):
        self.sequences = self.__load_seq(seq_path) # e.g. ./data/hpatches-release/v_pomegranate

    def display(self, n: int = 10):
        # display a batch of n sequences
        arr = np.vstack([np.hstack([self.sequences[j][i].data for i in range(16)]) for j in range(n)])
        import matplotlib.pyplot as plt
        plt.style.use('classic')
        plt.imshow(arr, cmap='gray')
        plt.xticks([i * 65 + 32 for i in range(16)], TYPES, fontsize=20)
        plt.yticks([])
        plt.show()

    def __load_seq(self, seq_path: str):
        seq = []
        for t in TYPES:
            name = seq_path.split('/')[-1]
            data = np.asarray(Image.open(Path(seq_path) / f'{t}.png'))
            l = data.shape[0] / 65
            seq.append([Patch(name, t, patch) for patch in np.split(data, l)]) # split into 65 x 65 patches

        return np.vstack([*zip(*seq)]) # PATCHES x TYPES

    def __len__(self): return len(self.sequences) # number of patches

    def __getitem__(self, idx: int): return self.sequences[idx] # get sequence of patches

