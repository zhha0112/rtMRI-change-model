import os
import librosa
import torch
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


def load_checkpoint(step, model_name="transformer"):
    state_dict = torch.load('./checkpoint/checkpoint_%s_%d.pth.tar' % (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def plot_spectrogram(mel, sr, filename):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    # plt.show()
    plt.savefig('images/' + filename)


def synthesis(vid, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    vid = torch.FloatTensor(vid).unsqueeze(0)
    vid = vid.cuda()
    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_vid = torch.arange(1, vid.size(1) + 1).unsqueeze(0)
    pos_vid = pos_vid.cuda()

    m = m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)

    pbar = tqdm(range(args.max_len))
    with torch.no_grad():
        for i in pbar:
            pos_mel = torch.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(vid, mel_input, pos_vid, pos_mel)
            mel_input = torch.cat([mel_input, mel_pred[:, -1:, :]], dim=1)

        plot_spectrogram(mel_input.squeeze(0).cpu().numpy(), hp.sr, filename='pred_5.png')
        mag_pred = m_post.forward(postnet_pred)

    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=110000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=78000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=200)

    args = parser.parse_args()

    video_feauture_file = os.path.join(hp.feature_path, 'resnet_features_test.pt')
    resnet_features = torch.load(video_feauture_file, weights_only=True)

    synthesis(resnet_features[5], args)