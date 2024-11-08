import os

import cv2
from pydub import AudioSegment, silence
from scipy.io import wavfile


def get_non_silent_chunks(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    non_silent_chunks = silence.detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

    return non_silent_chunks


def split_and_save_video(video_path, output_dir, base_name, non_silence_chunks, video_fps=23.18):
    durations = [(start / 1000, end / 1000) for start, end in non_silence_chunks]  # convert to seconds
    frames = [(int(video_fps * start), int(video_fps * end)) for start, end in durations]

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    for idx, (start_frame, end_frame) in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_name = f'{base_name}_{idx + 1}.avi'
        output_video_path = os.path.join(output_dir, segment_name)
        out_video = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))

        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Cannot read frame {frame_num}.")
                break
            out_video.write(frame)

        out_video.release()

    cap.release()

    return frames


def split_and_save_audio(audio_path, output_dir, base_name, frames, audio_sampling_rate=20004, video_fps=23.18):
    delta_t = 1 / video_fps
    audio_rate, audio_data = wavfile.read(audio_path)

    for idx, (start_frame, end_frame) in enumerate(frames):
        audio_start = int((start_frame / video_fps) * audio_sampling_rate)
        audio_end = int(((end_frame / video_fps) + delta_t) * audio_sampling_rate)

        audio_segment = audio_data[audio_start:audio_end + 1]

        segment_name = f'{base_name}_{idx + 1}.wav'
        wavfile.write(os.path.join(output_dir, segment_name), audio_sampling_rate, audio_segment)


data_dir = 'F:/processed/without_noise/'

wav_dir = data_dir + 'wav'
avi_dir = data_dir + 'avi'

save_path = 'F:/'
output_wav_dir = save_path + 'wav'
output_avi_dir = save_path + 'avi'

os.makedirs(output_wav_dir, exist_ok=True)
os.makedirs(output_avi_dir, exist_ok=True)

video_fps = 23.18
audio_sampling_rate = 20004

try:
    for wav_file in os.listdir(wav_dir):
        if wav_file.endswith('.wav'):
            wav_path = os.path.join(wav_dir, wav_file)
            avi_path = os.path.join(avi_dir, wav_file.replace('.wav', '.avi'))

            if os.path.exists(avi_path):
                base_name = os.path.splitext(wav_file)[0]
                output_wav_subdir = os.path.join(output_wav_dir, base_name)
                output_avi_subdir = os.path.join(output_avi_dir, base_name)

                non_silent_chunks = get_non_silent_chunks(wav_path)
                frames = split_and_save_video(avi_path, output_avi_dir, base_name, non_silent_chunks, video_fps)
                split_and_save_audio(wav_path, output_wav_dir, base_name, frames, audio_sampling_rate, video_fps)

            else:
                print(f"Corresponding video file not found for {wav_file}")

    print("Processing completed.")
except:
    print("Error: An error occurred during processing.")