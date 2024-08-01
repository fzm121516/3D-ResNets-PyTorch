import argparse
import os
import glob
import shutil
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from models import resnet
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop, ToTensor, ScaleValue)


def real_glob(rglob):
    from braceexpand import braceexpand
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files += glob.glob(g)
    return files


allowed_gait_types = ['nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']


def main():
    parser = argparse.ArgumentParser(description="Run model against images")
    parser.add_argument('--videos-dir', type=str, default="/home/fanzheming/zm/mygait/datasets/CASIA-B/dataset-b3",
                        help="Directory containing videos")
    parser.add_argument('--images-dir', type=str,
                        default="/home/fanzheming/zm/mygait/datasets/CASIA-B/dataset-b-frames-50-original-1280960")
    parser.add_argument("--depth", default="50", help="Which model depth")

    args = parser.parse_args()

    model_files = {
        "34": "data/saved/r3d34_K_200ep.pth",
        "50": "data/saved/r3d50_M_200ep.pth",
        "101": "data/saved/r3d101_K_200ep.pth"
    }

    model_file = model_files.get(args.depth)
    if model_file is None:
        raise ValueError(f"Unsupported model depth: {args.depth}")

    model_depth = int(args.depth)

    model = resnet.generate_model(model_depth=model_depth,
                                  n_classes=339,
                                  n_input_channels=3,
                                  shortcut_type="B",
                                  conv1_t_size=7,
                                  conv1_t_stride=1,
                                  no_max_pool=False,
                                  widen_factor=1.0)

    checkpoint = torch.load(model_file, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    arch = '{}-{}'.format("resnet", model_depth)
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True),
                         *glob.glob(os.path.join(args.videos_dir, '**', '*.mp4'), recursive=True)])
    num_video = len(video_list)
    print("Found", num_video, "videos")

    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]
    normalize = Normalize(mean, std)

    sample_size = 112

    spatial_transform = Compose([Resize(sample_size), CenterCrop(sample_size), ToTensor(), ScaleValue(1), normalize])

    overall_highest_count = 0
    total_clips = 0

    for i, video_path in enumerate(video_list):
        video_name = os.path.basename(video_path)
        image_name, _ = os.path.splitext(video_name)
        print(i + 1, '/', num_video, image_name)

        images_dir = os.path.join(
            args.images_dir,
            os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
        )
        image_list = sorted(glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True))

        parts = image_name.split('-')

        if len(parts) == 4:  # If the number of parts is 4, the filename format is correct
            gait_id = parts[0]
            gait_type = f"{parts[1]}-{parts[2]}"
            gait_view = parts[3]  # Combine the second and third parts
        else:  # If the filename format is not as expected, skip this file
            print(f"Unexpected filename format: {video_name}")
            continue

        # Check if gait_type is in allowed_gait_types
        if gait_type not in allowed_gait_types:
            print(f"Gait type {gait_type} not in allowed list, skipping.")
            continue

        # Check if gait_id is within the range 075 to 124
        try:
            gait_id_num = int(gait_id)
            if gait_id_num < 75 or gait_id_num > 124:
                print(f"Gait ID {gait_id} not in the allowed range (075-124), skipping.")
                continue
        except ValueError:
            print(f"Invalid Gait ID {gait_id}, skipping.")
            continue

        frames = image_list

        if len(frames) < 16:
            print(f"Not enough frames in {video_path}")
            continue

        highest_count = 0

        # Change step from 16 to 8 for overlapping
        for j in range(0, len(frames) - 15, 8):
            image_clips = frames[j:j + 16]

            clip = []
            for f in image_clips:
                img = Image.open(f).convert("RGB")
                img = spatial_transform(img)
                clip.append(img)

            model_clips = torch.stack([torch.stack(clip, 0).permute(1, 0, 2, 3)], 0)

            with torch.no_grad():
                outputs = model(model_clips)
                outputs = F.softmax(outputs, dim=1).cpu()

            # Get the highest confidence and its corresponding class
            confidence, predicted_class = torch.max(outputs, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

            if predicted_class == 155:
                highest_count += 1

            # Print the highest class and its confidence for this clip
            print(f"Clip: Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")

        total_clips += (len(frames) - 15) // 8
        overall_highest_count += highest_count

        if highest_count:
            print(f"Highest label 155 count for video {video_path}: {highest_count}")

    if total_clips:
        highest_percentage = (overall_highest_count / total_clips) * 100
        print(f"Percentage of clips with highest label 155: {highest_percentage:.2f}%")
    else:
        print("No clips were processed.")


if __name__ == "__main__":
    main()
