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


def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(output_dir, f"frame{count:04d}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    return sorted(glob.glob(os.path.join(output_dir, "*.jpg")))


def main():
    parser = argparse.ArgumentParser(description="Run model against images")
    parser.add_argument('--videos-dir', type=str, default="E:/CASIA_Gait_Dataset/DatasetB",
                        help="Directory containing videos")
    parser.add_argument('--images-dir', type=str, default="E:/CASIA_Gait_Dataset/DatasetB-frames-test-1280960-detect-50")
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

    overall_confidence_scores = []

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

        if len(parts) != 4:
            print(f"Unexpected filename format: {image_name}")
            continue

        frames = image_list

        if len(frames) < 16:
            print(f"Not enough frames in {video_path}")
            continue

        clip_confidences = []

        # Change step from 16 to 8 for overlapping
        for j in range(0, len(frames) - 15, 16):
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

            confidence_score = outputs[0, 155].item()
            clip_confidences.append(confidence_score)

        if clip_confidences:
            average_confidence = sum(clip_confidences) / len(clip_confidences)
            overall_confidence_scores.append(average_confidence)
            print(f"Average confidence score for video {video_path}: {average_confidence}")

    if overall_confidence_scores:
        overall_average_confidence = sum(overall_confidence_scores) / len(overall_confidence_scores)
        print(f"Overall average confidence score: {overall_average_confidence}")
    else:
        print("No confidence scores were calculated.")


if __name__ == "__main__":
    main()
