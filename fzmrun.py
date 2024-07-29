import argparse
from models import resnet
from model import load_pretrained_model
from PIL import Image
from braceexpand import braceexpand
import glob
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                ToTensor, ScaleValue)
import torch
import torch.nn.functional as F


def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return files


def extend_to_length(files, l):
    while len(files) < l:
        files = files + files
    return files[:l]


model_files = {
    "34": "data/saved/r3d34_K_200ep.pth",
    "50": "data/saved/r3d50_M_200ep.pth",
    "101": "data/saved/r3d101_K_200ep.pth"
}


def main():
    parser = argparse.ArgumentParser(description="Run model against images")
    parser.add_argument('--input-glob',
                        default='E:/CASIA_Gait_Dataset/DatasetB-frames/075/nm-01/000/075-nm-01-000-0{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16}.png',
                        help="inputs")
    parser.add_argument("--depth", default="50",
                        help="which model depth")
    args = parser.parse_args()

    model_file = model_files[args.depth]
    model_depth = int(args.depth)

    model = resnet.generate_model(model_depth=model_depth,
                                  n_classes=339,
                                  n_input_channels=3,
                                  shortcut_type="B",
                                  conv1_t_size=7,
                                  conv1_t_stride=1,
                                  no_max_pool=False,
                                  widen_factor=1.0)

    # model = load_pretrained_model(model, args.model, "resnet", 700)

    checkpoint = torch.load(model_file, map_location='cpu')
    arch = '{}-{}'.format("resnet", model_depth)
    print(arch, checkpoint['arch'])
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        # I think this only for legacy models
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    image_clips = []
    files = real_glob(args.input_glob)
    files = extend_to_length(files, 16)
    print(files)
    for f in files:
        img = Image.open(f).convert("RGB")
        image_clips.append(img)

    # print("EARLY", image_clips[0][0:4,0:4,0])

    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]
    normalize = Normalize(mean, std)

    sample_size = 112

    spatial_transform = [Resize(sample_size)]
    spatial_transform.append(CenterCrop(sample_size))
    spatial_transform.append(ToTensor())
    spatial_transform.extend([ScaleValue(1), normalize])
    spatial_transform = Compose(spatial_transform)

    # c = spatial_transform(image_clips[0])
    # c.save("raw.png")

    model_clips = []
    clip = [spatial_transform(img) for img in image_clips]
    model_clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
    model_clips = torch.stack(model_clips, 0)

    print("Final", model_clips.shape)
    print("PEEK", model_clips[0, 0, 0, 0:4, 0:4])

    with torch.no_grad():
        outputs = model(model_clips)
        print(outputs[0][0:10])
        outputs = F.softmax(outputs, dim=1).cpu()

    sorted_scores, locs = torch.topk(outputs[0], k=3)

    print(locs[0])

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': locs[i].item(),
            'score': sorted_scores[i].item()
        })

    print(video_results)


if __name__ == '__main__':
    main()
