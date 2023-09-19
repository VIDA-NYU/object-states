import os
import tqdm
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import fiftyone as fo
import supervision as sv
from .config import get_cfg
from .util.video import crop_box, iter_video
from .util.format_convert import fo_to_sv
from .util.step_annotations import add_step_annotations, fname_to_video_id

from detic import Detic


import torch
import clip as cliplib

from PIL import Image


device = "cuda"

import ipdb
@ipdb.iex
@torch.no_grad()
def main(dataset_dir, field='detections_tracker'):
    cfg = get_cfg()

    # -------------------------------- Load models ------------------------------- #

    detic = Detic(['cat'])
    clip, clip_preprocess = cliplib.load("ViT-B/32", device=device)
    
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

    # ------------------------- Prepare output directory ------------------------- #

    out_dir = os.path.join(dataset_dir, 'embeddings', field)
    os.makedirs(out_dir, exist_ok=True)

    # track_id, embedding_type, keys
    embeddings = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

    # ----------------------------- Loop over videos ----------------------------- #

    for sample in tqdm.tqdm(view):
        video_id = fname_to_video_id(sample.filepath)
        video_name = os.path.basename(sample.filepath)
        video_out_dir = os.path.join(out_dir, video_name)
        sdf = steps_df[steps_df.video_id == video_id]

        # ----------------------------- Loop over frames ----------------------------- #


        for i, frame, finfo in iter_video(sample):
            # get detection
            detections, labels = fo_to_sv(finfo[field])

            # ---------------------------- Get step annotation --------------------------- #

            step_rows = sdf[(i >= sdf.start_frame) & (i <= sdf.stop_frame)].iloc[:1]
            if not len(step_rows):
                continue
            row = step_rows.iloc[0]
            step, start, end = row.narration, row.start_frame, row.stop_frame
            pct = (i - start) / (end - start)

            # ---------------------------- Get clip embeddings --------------------------- #

            for d in zip(detections):
                x = clip_preprocess(Image.fromarray(crop_box(frame, d.xyxy)))[None].to(device)
                z = clip.encode_image(x)[0]

                embeddings[track_id]['clip']['z'].append(z)
                embeddings[track_id]['clip']['steps'].append(step)
                embeddings[track_id]['clip']['pct'].append(pct)

            # --------------------------- Get Detic embeddings --------------------------- #

            instances = detic(frame, boxes=[torch.as_tensor(detections.xyxy, device='cuda')])
            z_detic = instances.stage_features  # n, 3, 512

            for track_id, z in zip(detections.tracker_id, z_detic.cpu().numpy()):
                embeddings[track_id]['detic']['z'].append(z.mean(1))
                embeddings[track_id]['detic']['steps'].append(step)
                embeddings[track_id]['detic']['pct'].append(pct)

                for stage in z_detic.shape[1]:
                    embeddings[track_id][f'detic_s{stage}']['z'].append(z[stage])
                    embeddings[track_id][f'detic_s{stage}']['steps'].append(step)
                    embeddings[track_id][f'detic_s{stage}']['pct'].append(pct)


    # ------------------------- Write embeddings to file ------------------------- #

    for track_id, ds in embeddings.items():
        for kind, data in embeddings.items():
            fname = f'{video_out_dir}/{kind}/{track_id}.npz'
            os.makedirs(fname, exist_ok=True)
            z = np.array(data['z'])
            steps = np.array(data['steps'])
            percent = np.array(data['percent'])
            print(fname, z.shape, steps.shape, percent.shape)
            np.savez(fname, z=z, step=steps, percent=percent)


def augment_frame(frame):
    return frame[None]  # TODO


if __name__ == '__main__':
    import fire
    fire.Fire(main)