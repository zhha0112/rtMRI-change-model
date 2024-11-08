import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import cv2
import numpy as np
from torchvision import transforms
from utils import update_hyperparams


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.modules = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        for module_name in self.modules:
            self.add_module(module_name, getattr(pretrained_resnet, module_name))

    def forward(self, x):
        for module_name in self.modules:
            x = getattr(self, module_name)(x)
        return x


class VideoFeatureExtractionModel(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractionModel, self).__init__()

        self.feature_extraction = ResNet18()
        self.linear = nn.Linear(25088, 512)

    def forward(self, video):
        v_b, v_s, v_c, v_w, v_h = video.shape
        video = video.view(v_b * v_s, v_c, v_w, v_h)

        video_embedding = self.feature_extraction(video)
        video_embedding = video_embedding.view(v_b, v_s, -1)
        video_embedding = self.linear(video_embedding)
        video_embedding = video_embedding.reshape(v_b, v_s, -1)

        return video_embedding


class VideoDataset(Dataset):
    def __init__(self, avi_folder, file_list, mean, std):
        self.avi_folder = avi_folder
        self.file_list = file_list
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        video_path = os.path.join(self.avi_folder, filename)
        video_frames = preprocess_video(video_path, self.mean, self.std)
        video_frames = torch.from_numpy(video_frames)

        return video_frames


def preprocess_video(video_path, mean, std, target_size=(224, 224)):
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        normalize
    ])

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Ensure the frame has 3 channels
        if frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).numpy()
        frames.append(frame)
    cap.release()
    frames = np.stack(frames, axis=0)

    return frames


def calculate_mean_std_from_videos(folder_path, video_files):
    sum_channels = np.zeros(3)
    sum_squares_channels = np.zeros(3)
    total_pixels = 0

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]

            sum_channels += frame.sum(axis=(0, 1))  # Accumulate the sum of pixel values for each channel
            sum_squares_channels += (frame ** 2).sum(axis=(0, 1))

            total_pixels += frame.shape[0] * frame.shape[1]

        cap.release()

    # mean and standard deviation for each channel
    mean_channels = sum_channels / total_pixels
    std_channels = np.sqrt(sum_squares_channels / total_pixels - mean_channels ** 2)

    return mean_channels.tolist(), std_channels.tolist()


def prepare_video_data(avi_folder, avi_files, batch_size=1, mean=None, std=None, file_name=None):
    if mean is None or std is None:
        mean, std = calculate_mean_std_from_videos(avi_folder, avi_files)
        update_hyperparams('vid_mean', mean)
        update_hyperparams('vid_std', std)

    dataset = VideoDataset(avi_folder, avi_files, mean, std)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=None, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoFeatureExtractionModel().to(device)
    model.train()

    video_embedding_list = []

    for batch in tqdm(data_loader, desc="Preparing video data"):
        video_frames = batch.to(device)
        video_embeddings = model(video_frames)

        video_embedding_list.extend(video_embeddings.cpu().detach())

    if file_name:
        torch.save(video_embedding_list, "F:/features/" + file_name + ".pt")

    return video_embedding_list
