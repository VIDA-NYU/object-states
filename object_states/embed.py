from . import _patch

import cv2
import numpy as np
import fiftyone as fo
import supervision as sv
from .util.format_convert import fo_to_sv
from .util.step_annotations import add_step_annotations

from detic import Detic


import torch
import clip as cliplib

from PIL import Image


device = "cuda"

import ipdb
@ipdb.iex
def main(dataset_dir, fields='detections_tracker'):

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

    add_step_annotations(view, steps_csv)

    # ------------------------- Prepare output directory ------------------------- #

    out_dir = os.path.join(dataset_dir, 'embeddings', fields)
    os.makedirs(out_dir, exist_ok=True)

    # track_id, embedding_type, keys
    embeddings = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

    # ----------------------------- Loop over videos ----------------------------- #

    for sample in tqdm.tqdm(view):
        video_name = os.path.basename(sample.metadata.video_path)
        video_out_dir = os.path.join(out_dir, video_name)

        # ----------------------------- Loop over frames ----------------------------- #

        for i, frame, finfo in iter_video(sample):
            # get detection
            detections, labels = fo_to_sv(finfo[field])

            step, start, end = ...
            pct = (i - start) / (end - start)

            # ---------------------------- Get clip embeddings --------------------------- #
            
            crops = crop_detections(frame, detections)
            z_clip = clip.encode_image(torch.stack([
                clip_preprocess(Image.fromarray(im)) for im in crops], dim=0, device=device))

            for track_id, z in zip(detections.tracker_id, z_clip):
                embeddings[track_id]['clip']['z'].append(z)
                embeddings[track_id]['clip']['steps'].append(step)
                embeddings[track_id]['clip']['pct'].append(pct)

            # --------------------------- Get Detic embeddings --------------------------- #

            instances = model(frame, boxes=[torch.as_tensor(detections.xyxy, device='cuda')])
            z_detic = instances.stage_features  # n, 3, 512

            for track_id, z in zip(detections.tracker_id, z_detic):
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
            np.savez(
                fname, 
                z=np.array(data['z']),
                step=np.array(data['steps']),
                percent=np.array(data['percent']),
            )


def augment_frame(frame):
    return frame[None]  # TODO


if __name__ == '__main__':
    import fire
    fire.Fire(main)