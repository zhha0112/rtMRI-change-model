# Audio
num_mels = 80
# num_freq = 1024
n_fft = 1024
sr = 20000
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.95  # tractron: 0.97
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr * frame_shift)  # samples.
win_length = 1024  # samples.
n_mels = 80  # Number of Mel banks to generate
power = 1.2  # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 512
embedding_size = 512
max_db = 100
ref_db = 25  # 20

vid_mean = [0.10323505712651736, 0.12170008119497644, 0.11400658408836856]
vid_std = [0.1364854534767809, 0.13737809329688586, 0.13725117329961473]

n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 10000  # 10000
lr = 0.001
save_step = 2000
image_step = 500
batch_size = 32

data_path = 'F:/TIMIT'
feature_path = 'F:/features'
checkpoint_path = 'F:/checkpoint'
sample_path = 'F:/samples'