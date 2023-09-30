import os
import tqdm
from collections import defaultdict
import cv2
import torch
import numpy as np
import pandas as pd
import fiftyone as fo
import supervision as sv

import torchvision.transforms as T

from .config import get_cfg
from .util.video import crop_box, iter_video
from .util.format_convert import fo_to_sv
from .util.step_annotations import add_step_annotations, fname_to_video_id

from detic import Detic
from IPython import embed


import torch
import clip

from PIL import Image


device = "cuda"

import ipdb
@ipdb.iex
@torch.no_grad()
def main(config_fname, fields=['ground_truth_tracker', 'detections_tracker'], file_path=None):
    cfg = get_cfg(config_fname)

    dataset_dir = cfg.DATASET.ROOT
    n_augs = cfg.DATASET.N_AUGMENTATIONS
    skip_every = cfg.DATASET.EMBED_SKIP

    # -------------------------------- Load models ------------------------------- #

    detic = Detic(['cat'])
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # ------------------------------- Load dataset ------------------------------- #

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    )

    view = dataset.view()
    if file_path:
        view = view.match(F("filepath") == os.path.abspath(filepath))
        assert len(view) == 1

    view.compute_metadata(overwrite=True)

    # ------------------------- Prepare output directory ------------------------- #

    # for s in view:
    #     print(s.filepath, {k: len([i for i in s.frames if s.frames[i][k] is not None]) for k in fields if s.frames[1].has_field(k)})
    # input()

    # Define a single transform that combines all augmentations
    if n_augs:
        image_augmentation = get_augmentor(cfg)

    # ----------------------------- Loop over videos ----------------------------- #

    for sample in tqdm.tqdm(view):
        video_name = os.path.basename(sample.filepath)
        
        counts = {k: next((i for i in sample.frames if sample.frames[i][k] is not None), None) for k in fields if sample.frames[1].has_field(k)}
        tqdm.tqdm.write(f"{video_name} field start frame: {counts}")
        for field in fields:
            if counts.get(field) is not None:
                tqdm.tqdm.write(f"Using field {field} for {video_name}")
                break
        else:
            tqdm.tqdm.write(f"\n\nSkipping {video_name} as it has no fields {fields}\n\n")
            continue

        out_dir = os.path.join(dataset_dir, 'embeddings', field)
        os.makedirs(out_dir, exist_ok=True)
        video_out_dir = os.path.join(out_dir, video_name)
        if os.path.isdir(video_out_dir):
            tqdm.tqdm.write(f"Skipping {video_out_dir} already exists")
            continue

        # ----------------------------- Loop over frames ----------------------------- #

        # track_id, embedding_type, keys
        embeddings = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        for i, frame, finfo, pbar in iter_video(sample, pbar=True):
            # if skip_every and i % skip_every: continue

            # get detection
            try:
                detections, labels = fo_to_sv(finfo[field], frame.shape)
            except Exception:
                print(f"\n\nFailed loading detection for: {video_name} frame {i}\n\n")
                import traceback
                traceback.print_exc()
                continue

            if not len(detections):
                continue

            pbar.set_description(f'{len(detections)} {labels}')

            # ---------------------------- Get clip embeddings --------------------------- #

            for xyxy, _, conf, _, track_id in detections:
                rgb = frame[:,:,::-1]
                crop: np.ndarray = crop_box(rgb, xyxy)
                if not crop.size:
                    continue
                crop = Image.fromarray(crop)
                aug_crop = Image.fromarray(crop_box(rgb, xyxy, padding=30))
                aug_crops = [aug_crop] + [
                    image_augmentation(aug_crop)
                    for i in range(n_augs)
                ]
                ims = [crop] + aug_crops
                
                x = torch.stack([preprocess(im) for im in ims]).to(device)
                z = model.encode_image(x).cpu().numpy()

                embeddings[track_id]['clip']['z'].extend(z)
                embeddings[track_id]['clip']['frame_index'].extend([i]*len(z))
                embeddings[track_id]['clip']['augmented'].extend([False] + [True]*len(aug_crops))

            # --------------------------- Get Detic embeddings --------------------------- #
            
            if skip_every and i % (skip_every*3): continue

            instances = detic(frame, boxes=[torch.as_tensor(detections.xyxy, device='cuda')])
            z_detic = instances.stage_features.cpu().numpy()  # n, 3, 512

            for track_id, z in zip(detections.tracker_id, z_detic):
                embeddings[track_id]['detic']['z'].append(z.mean(0))
                embeddings[track_id]['detic']['frame_index'].append(i)

                for stage in range(z.shape[0]):
                    embeddings[track_id][f'detic_s{stage}']['z'].append(z[stage])
                    embeddings[track_id][f'detic_s{stage}']['frame_index'].append(i)

        # ------------------------- Write embeddings to file ------------------------- #

        for track_id, ds in embeddings.items():
            for kind, data in ds.items():
                fname = f'{video_out_dir}/{kind}/{track_id}.npz'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                data = {k: np.array(x) for k, x in data.items()}
                print(fname, {k: x.shape for k, x in data.items()})
                np.savez(fname, **data, video_name=video_name)


def augment_frame(frame):
    return frame[None]  # TODO

def get_augmentor(cfg=None, flip=True):
    aug = T.Compose([
        *([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ] if flip else []),
        T.RandomRotation(degrees=(-30, 30)),
        T.AugMix(),
        # T.TrivialAugmentWide(), # this polarizes
        # T.ColorJitter(brightness=0.2, contrast=0.2),
        # T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        # T.ElasticTransform(alpha=1.0, sigma=1.),
        T.RandomPerspective(distortion_scale=0.5),
        # T.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),
    ])
    return aug

def test_aug_frame(video_path, out_path='augtest.mp4'):
    cfg = None#get_cfg(config_fname)
    aug = get_augmentor(cfg, flip=False)

    video_info = sv.VideoInfo.from_video_path(video_path=video_path)

    with sv.VideoSink(out_path, video_info) as s:
        for i, frame in tqdm.tqdm(
            enumerate(sv.get_video_frames_generator(video_path), 1),
            total=video_info.total_frames,
            desc=video_path
        ):
            s.write_frame(np.array(aug(Image.fromarray(frame[:,:,::-1])))[:,:,::-1])

if __name__ == '__main__':
    import fire
    fire.Fire(main)