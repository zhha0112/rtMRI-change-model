import os
import librosa
import soundfile as sf


def resample_audio(audio_file, output_path):
    audio, sr = librosa.load(audio_file, sr=20000)
    resampled_audio = librosa.resample(audio, orig_sr=20000, target_sr=20004)

    sf.write(output_path, resampled_audio, 20004)


data_dir = 'F:/TIMIT/Data/F1/'

wav_dir = data_dir + 'wav'
avi_dir = data_dir + 'avi'

output_wav_dir = data_dir + 'resampled_wav'

os.makedirs(output_wav_dir, exist_ok=True)

try:
    for wav_file in os.listdir(wav_dir):
        if wav_file.endswith('.wav'):
            wav_path = os.path.join(wav_dir, wav_file)

            output_file_path = os.path.join(output_wav_dir, wav_file)
            resample_audio(wav_path, output_file_path)
    print("Resampling complete.")
except Exception as e:
    print(e)
    print("Error: Could not resample.")
    exit()