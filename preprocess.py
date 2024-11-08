import os
from collections.abc import Mapping

import hyperparams as hp
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_list_from_file


class RtMRIMelDataset(Dataset):
    def __init__(self, metadata_file, wav_root_dir, video_feauture_file):
        self.file_list = read_list_from_file(metadata_file)
        self.wav_root_dir = wav_root_dir
        self.resnet_features = torch.load(video_feauture_file, weights_only=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.wav_root_dir, self.file_list[idx].replace('avi', 'wav'))

        vid = self.resnet_features[idx]
        vid_length = vid.shape[0]

        mel = np.load(wav_name[:-4] + '.pt.npy')
        mel_input = np.concatenate([np.zeros([1, hp.num_mels], np.float32), mel[:-1, :]],
                                   axis=0)  # shift and append zero vector

        pos_vid = np.arange(1, vid_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)  # position indices

        sample = {'vid': vid, 'mel': mel, 'vid_length': vid_length, 'mel_input': mel_input, 'pos_mel': pos_mel,
                  'pos_vid': pos_vid}

        return sample


class PostDatasets(Dataset):
    def __init__(self, metadata_file, wav_root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.file_list = read_list_from_file(metadata_file)
        self.wav_root_dir = wav_root_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.wav_root_dir, self.file_list[idx].replace('avi', 'wav'))

        mel = np.load(wav_name[:-4] + '.pt.npy')
        mag = np.load(wav_name[:-4] + '.mag.npy')

        sample = {'mel': mel, 'mag': mag}

        return sample


def collate_fn_transformer(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], Mapping):
        vid = [d['vid'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        vid_length = [d['vid_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_vid = [d['pos_vid'] for d in batch]

        # sort by text length in descending order (improve padding efficiency)
        vid = [i for i, _ in sorted(zip(vid, vid_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, vid_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, vid_length), key=lambda x: x[1], reverse=True)]
        pos_vid = [i for i, _ in sorted(zip(pos_vid, vid_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, vid_length), key=lambda x: x[1], reverse=True)]
        vid_length = sorted(vid_length, reverse=True)

        # PAD sequences with largest length of the "batch"
        vid = _pad_vid(vid)
        mel = _pad_mel(mel)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_vid = _prepare_data(pos_vid).astype(np.int32)

        return torch.FloatTensor(vid), torch.FloatTensor(mel), torch.FloatTensor(mel_input), torch.LongTensor(
            pos_vid), torch.LongTensor(pos_mel), torch.LongTensor(vid_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def collate_fn_postnet(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], Mapping):
        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]

        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return torch.FloatTensor(mel), torch.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def get_dataset(metadata_file, vid_feature_file):
    vid_feature_path = os.path.join(hp.feature_path, vid_feature_file)
    wav_path = os.path.join(hp.data_path, 'wav')

    return RtMRIMelDataset(metadata_file, wav_path, vid_feature_path)


def get_post_dataset(metadata_file):
    wav_path = os.path.join(hp.data_path, 'wav')

    return PostDatasets(metadata_file, wav_path)


def _pad_vid(inputs):
    _pad = 0

    def _pad_one(x, max_len):
        vid_len = x.shape[0]
        return np.pad(x, [[0, max_len - vid_len], [0, 0]], mode='constant', constant_values=_pad)

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])


def _pad_mel(inputs):
    _pad = 0

    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=_pad)

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
