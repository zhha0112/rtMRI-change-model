import os

import hyperparams as hp
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import get_spectrograms


class PrepareAudioDataset(Dataset):
    def __init__(self, root_dir, file_list):
        self.root_dir = root_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav_name = self.file_list[idx].replace('avi', 'wav')
        wav_path = os.path.join(self.root_dir, wav_name)
        mel, mag = get_spectrograms(wav_path)

        np.save(wav_path[:-4] + '.pt', mel)
        np.save(wav_path[:-4] + '.mag', mag)

        sample = {'mel': mel, 'mag': mag}

        return sample