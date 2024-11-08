import os
import shutil

source_dir = 'F:/TIMIT_T/Data'
destination_dir = 'F:/vscode/temp_data'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for folder in os.listdir(source_dir):
    if folder.startswith('M'):
        continue
    folder_path = os.path.join(source_dir, folder)

    if os.path.isdir(folder_path):
        wav_dir = os.path.join(folder_path, 'wav')
        avi_dir = os.path.join(folder_path, 'avi')

        # Move all .wav files
        if os.path.exists(wav_dir):
            for wav_file in os.listdir(wav_dir):
                if wav_file.endswith('.wav'):
                    wav_path = os.path.join(wav_dir, wav_file)
                    shutil.move(wav_path, destination_dir)
                    print(f"Moved {wav_file} to {destination_dir}")

        # Move all .avi files
        if os.path.exists(avi_dir):
            for avi_file in os.listdir(avi_dir):
                if avi_file.endswith('.avi'):
                    avi_path = os.path.join(avi_dir, avi_file)
                    shutil.move(avi_path, destination_dir)
                    print(f"Moved {avi_file} to {destination_dir}")