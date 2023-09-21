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


import torch
import clip

from PIL import Image


device = "cuda"

import ipdb
@ipdb.iex
@torch.no_grad()
def main(config_fname, field='detections_tracker'):
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
    view.compute_metadata(overwrite=True)

    # add_step_annotations(view, cfg.DATASET.STEPS_CSV)
    steps_df = pd.read_csv(cfg.DATASET.STEPS_CSV)
    steps_df['start_frame'] = steps_df.start_frame.astype(int) + 1
    steps_df['stop_frame'] = steps_df.stop_frame.astype(int) + 1
    steps_df['step_index'] = pd.to_numeric(steps_df.narration.str.split('.').str[0], errors='coerce')
    steps_df = steps_df[~pd.isna(steps_df.step_index)]
    assert len(steps_df), "umm...."

    # ------------------------- Prepare output directory ------------------------- #

    out_dir = os.path.join(dataset_dir, 'embeddings', field)
    os.makedirs(out_dir, exist_ok=True)


    # Define a single transform that combines all augmentations
    if n_augs:
        image_augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=(-30, 30)),
            T.AugMix(),
            T.TrivialAugmentWide(),
            # T.ColorJitter(brightness=0.2, contrast=0.2),
            # T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
            # T.GaussianBlur(kernel_size=5, sigma=(0.2, 1)),
            # T.ElasticTransform(alpha=1.0, sigma=50),
            # T.Perspective(distortion_scale=0.5),
        ])


    # ----------------------------- Loop over videos ----------------------------- #

    for sample in tqdm.tqdm(view):
        video_id = fname_to_video_id(sample.filepath)
        video_name = os.path.basename(sample.filepath)
        video_out_dir = os.path.join(out_dir, video_name)
        sdf = steps_df[steps_df.video_id == video_id]
        if os.path.isdir(video_out_dir):
            tqdm.tqdm.write(f"Skipping {video_out_dir} already exists")
            continue

        # ----------------------------- Loop over frames ----------------------------- #

        # track_id, embedding_type, keys
        embeddings = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        for i, frame, finfo in iter_video(sample):
            if skip_every and i % skip_every: continue

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

            # ---------------------------- Get step annotation --------------------------- #

            step_rows = sdf[(i >= sdf.start_frame) & (i <= sdf.stop_frame)].iloc[:1]
            if not len(step_rows):
                continue
            row = step_rows.iloc[0]
            step, idx, start, end = row.narration, row.step_index, row.start_frame, row.stop_frame
            pct = (i - start) / (end - start)

            # ---------------------------- Get clip embeddings --------------------------- #

            for xyxy, _, conf, _, track_id in detections:
                rgb = frame[:,:,::-1]
                crop: np.ndarray = crop_box(rgb, xyxy)
                if not crop.size:
                    continue
                crop = Image.fromarray(crop)
                aug_crop = Image.fromarray(crop_box(rgb, xyxy, padding=10))
                aug_crops = [aug_crop] + [
                    image_augmentation(aug_crop)
                    for i in range(n_augs)
                ]
                ims = [crop] + aug_crops
                
                x = torch.stack([preprocess(im) for im in ims]).to(device)
                z = model.encode_image(x).cpu().numpy()

                embeddings[track_id]['clip']['z'].extend(z)
                embeddings[track_id]['clip']['steps'].extend([step]*len(z))
                embeddings[track_id]['clip']['index'].extend([idx]*len(z))
                embeddings[track_id]['clip']['pct'].extend([pct]*len(z))
                embeddings[track_id]['clip']['augmented'].extend([False] + [True]*len(aug_crops))

            tqdm.tqdm.write(f'{step} ({idx}) {pct:.2%} {len(detections)}')


            # # --------------------------- Get Detic embeddings --------------------------- #
            
            if skip_every and i % (skip_every*3): continue

            instances = detic(frame, boxes=[torch.as_tensor(detections.xyxy, device='cuda')])
            z_detic = instances.stage_features.cpu().numpy()  # n, 3, 512

            for track_id, z in zip(detections.tracker_id, z_detic):
                embeddings[track_id]['detic']['z'].append(z.mean(0))
                embeddings[track_id]['detic']['steps'].append(step)
                embeddings[track_id]['detic']['index'].append(idx)
                embeddings[track_id]['detic']['pct'].append(pct)

                for stage in range(z.shape[0]):
                    embeddings[track_id][f'detic_s{stage}']['z'].append(z[stage])
                    embeddings[track_id][f'detic_s{stage}']['steps'].append(step)
                    embeddings[track_id][f'detic_s{stage}']['index'].append(idx)
                    embeddings[track_id][f'detic_s{stage}']['pct'].append(pct)

        # ------------------------- Write embeddings to file ------------------------- #

        for track_id, ds in embeddings.items():
            for kind, data in ds.items():
                fname = f'{video_out_dir}/{kind}/{track_id}.npz'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                data = {k: np.array(x) for k, x in data.items()}
                print(fname, {k: x.shape for k, x in data.items()})
                np.savez(fname, **data, video_id=video_id, video_name=video_name)


def augment_frame(frame):
    return frame[None]  # TODO


if __name__ == '__main__':
    import fire
    fire.Fire(main)