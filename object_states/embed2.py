import os
import shutil
import tqdm
from collections import defaultdict
import cv2
import torch
import numpy as np
import pandas as pd
import supervision as sv
import torchvision.transforms as T
import pathtrees as pt
from object_states.util.color import *
from .config import get_cfg
from .util.video import crop_box, iter_video, iter_video2
from .util import eta_format as eta

from IPython import embed


import torch
import clip

from PIL import Image


device = "cuda"

import ipdb
@ipdb.iex
@torch.no_grad()
def main(config_fname, debug=False):
    cfg = get_cfg(config_fname)

    dataset_dir = cfg.DATASET.ROOT
    n_augs = cfg.DATASET.N_AUGMENTATIONS
    skip_every = cfg.DATASET.EMBED_SKIP

    # -------------------------------- Load models ------------------------------- #
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # ------------------------------- Load dataset ------------------------------- #

    tree = pt.tree(dataset_dir, {
        'labels': {'{name}.json': 'labels'},
        'manifest.json': 'manifest',
    })

    # ------------------------- Prepare output directory ------------------------- #

    # Define a single transform that combines all augmentations
    if n_augs:
        image_augmentation = get_augmentor(cfg)

    # ----------------------------- Loop over videos ----------------------------- #

    field = 'detections_tracker'
    manifest = eta.load(tree.manifest)
    for sample in tqdm.tqdm(manifest['index']):
        video_fname = sample['data']
        label_fname = sample['labels']
        video_name = os.path.basename(video_fname)

        out_dir = cfg.DATASET.EMBEDDING_DIR or os.path.join(dataset_dir, 'embeddings', field)
        os.makedirs(out_dir, exist_ok=True)
        video_out_dir = os.path.join(out_dir, video_name)
        if os.path.isdir(video_out_dir):
                tqdm.tqdm.write(f"{green('Skipping')} {video_out_dir} already exists")
                continue

        anns = eta.load(label_fname)
        all_track_ids = [
            o.get('index')
            for i in anns.get('frames') or []
            for o in eta.get_objects(anns, i)
        ]
        # from collections import Counter
        # count = Counter(all_track_ids)
        # track_ids = set(all_track_ids)

        # if os.path.isdir(video_out_dir):
        #     for emb_type in ['clip']:#os.listdir(video_out_dir):
        #         missing = {
        #             i for i in track_ids
        #             if not os.path.isfile(os.path.join(video_out_dir, emb_type, f'{i}.npz'))
        #         }
        #         if any(missing):
        #             print(count)
        #             input()
        #             tqdm.tqdm.write(f"{red('Deleting')} {video_out_dir} exists but is incomplete. {missing} from {track_ids}")
        #             # input("??")
        #             shutil.rmtree(video_out_dir)
        #             break
        #     else:
        #         tqdm.tqdm.write(f"{green('Skipping')} {video_out_dir} already exists")
        #         continue
        tqdm.tqdm.write(f"{blue('Doing')} {video_out_dir}")
        os.makedirs(video_out_dir, exist_ok=True)

        # ----------------------------- Loop over frames ----------------------------- #

        # track_id, embedding_type, keys
        embeddings = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        for i, frame, pbar in iter_video2(video_fname, pbar=True):
            if skip_every and i % skip_every: continue

            # get detection

            detections, labels = eta.get_sv_detections(anns, i, frame.shape)
            if not len(detections):
                continue

            pbar.set_description(f'{len(detections)} {labels}')

            # ---------------------------- Get clip embeddings --------------------------- #

            for (xyxy, _, conf, _, track_id), label in zip(detections, labels):
                if 'microwave' in label: continue
                rgb = frame[:,:,::-1]
                crop: np.ndarray = crop_box(rgb, xyxy)
                if debug:
                    Image.fromarray(crop).save('debug.png')
                    input()
                    continue
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