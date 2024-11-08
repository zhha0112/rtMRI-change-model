import os

import cv2
from scipy.io import wavfile


def split_and_save_video(video_path, skip_duration, output_dir, video_fps=23.18):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    num_frames_to_skip = int(video_fps * skip_duration)

    out_video = cv2.VideoWriter(output_dir, fourcc, video_fps, (width, height))

    # skip the first "num_frames_to_skip" frames
    for _ in range(num_frames_to_skip):
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read enough frames to remove 1 second from the video.")
            exit()

    # save the rest of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_video.write(frame)

    cap.release()
    out_video.release()

    return num_frames_to_skip


# remove silence from audio and save each segment
def split_and_save_audio(audio_path, output_dir, base_name, sampling_rate=20004, start_time=1):
    audio_rate, audio_data = wavfile.read(audio_path)

    num_audio_samples_to_extract = int(sampling_rate * start_time)
    audio_segment = audio_data[num_audio_samples_to_extract:]

    wavfile.write(os.path.join(output_dir, base_name + '.wav'), sampling_rate, audio_segment)


data_dir = 'F:/TIMIT/Data/F1/'

wav_dir = data_dir + 'wav'
avi_dir = data_dir + 'avi'

save_path = 'F:/processed/'
output_wav_dir = save_path + 'without_noise/wav'
output_avi_dir = save_path + 'without_noise/avi'

os.makedirs(output_wav_dir, exist_ok=True)
os.makedirs(output_avi_dir, exist_ok=True)

skip_duration = 1  # skip 1s
video_fps = 23.18
audio_sampling_rate = 20004

for wav_file in os.listdir(wav_dir):
    if wav_file.endswith('.wav'):
        wav_path = os.path.join(wav_dir, wav_file)
        avi_path = os.path.join(avi_dir, wav_file.replace('.wav', '.avi'))

        if os.path.exists(avi_path):
            base_name = os.path.splitext(wav_file)[0]
            output_wav_subdir = os.path.join(output_wav_dir, base_name)
            output_avi_subdir = os.path.join(output_avi_dir, base_name)

            out_vid_path = os.path.join(output_avi_dir, base_name + '.avi')
            skipped_vid_frames = split_and_save_video(avi_path, skip_duration, out_vid_path, video_fps)

            audio_start_time = (skip_duration / video_fps) * skipped_vid_frames
            split_and_save_audio(wav_path, output_wav_dir, base_name, audio_sampling_rate, audio_start_time)

        else:
            print(f"Corresponding video file not found for {wav_file}")

print("Processing completed.")