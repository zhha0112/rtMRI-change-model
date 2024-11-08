import os
import train_transformer
import train_postnet
import hyperparams as hp
from synthesis import plot_spectrogram
from utils import get_spectrograms, read_list_from_file

if __name__ == "__main__":
    # train_transformer.main()
    # train_postnet.main()

    test_file = './data/train_files.txt'
    wav_root = os.path.join(hp.data_path, 'wav')
    idx = 22

    test_file_list = read_list_from_file(test_file)
    wav_name = os.path.join(wav_root, test_file_list[idx].replace('avi', 'wav'))

    mel, mag = get_spectrograms(wav_name)
    plot_spectrogram(mel, hp.sr, filename='true_22.png')

