import importlib
import os
import random

import hyperparams as hp
from prepare_audio_data import PrepareAudioDataset
from prepare_vedio_data import prepare_video_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import update_hyperparams, write_list_to_file


def get_train_test_val_files(avi_folder, train_split=0.8, val_split=0.1):
    avi_files = [f for f in os.listdir(avi_folder) if f.endswith('.avi')]
    random.seed(42)
    random.shuffle(avi_files)

    total_files = len(avi_files)
    train_size = int(train_split * total_files)
    val_size = int(val_split * total_files)
    test_size = total_files - train_size - val_size

    train_files = avi_files[:train_size]
    val_files = avi_files[train_size:train_size + val_size]
    test_files = avi_files[train_size + val_size:]

    write_list_to_file(train_files, 'F:/train_files.txt')
    write_list_to_file(val_files, 'F:/val_files.txt')
    write_list_to_file(test_files, 'F:/test_files.txt')

    return train_files, val_files, test_files


if __name__ == '__main__':
    avi_folder = 'F:/avi'
    wav_folder = 'F:/wav'

    # prepare audio data
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
    audio_dataset = PrepareAudioDataset(wav_folder, wav_files)
    dataloader = DataLoader(audio_dataset, batch_size=1, drop_last=False, num_workers=8)

    pbar = tqdm(dataloader, desc='Preparing audio data')
    for d in pbar:
        pass

    # prepare video data
    update_hyperparams('vid_mean', None)
    update_hyperparams('vid_std', None)
    importlib.reload(hp)

    train_files, val_files, test_files = get_train_test_val_files(avi_folder, train_split=0.8, val_split=0.1)

    _ = prepare_video_data(avi_folder, train_files, mean=hp.vid_mean, std=hp.vid_std, file_name='resnet_features_train')
    importlib.reload(hp)
    _ = prepare_video_data(avi_folder, val_files, mean=hp.vid_mean, std=hp.vid_std, file_name='resnet_features_val')
    _ = prepare_video_data(avi_folder, test_files, mean=hp.vid_mean, std=hp.vid_std, file_name='resnet_features_test')