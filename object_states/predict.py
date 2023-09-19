from collections import defaultdict, Counter

import os
import cv2
import tqdm
import numpy as np
import torch
import fiftyone as fo
import supervision as sv
from torchvision.ops import masks_to_boxes

from .config import get_cfg
from .util.video import XMemSink, iter_video
from .util.format_convert import *

from xmem import XMem
from detic import Detic

device = 'cuda'

import ipdb
@ipdb.iex
def main(config_fname, field='detections', detect=False, detect_every=6, video_pattern=None):
    cfg = get_cfg(config_fname)
    
    dataset_dir = cfg.DATASET.ROOT
    video_pattern = cfg.DATASET.VIDEO_PATTERN
    
    # ------------------------------- Load dataset ------------------------------- #

    if os.path.exists(dataset_dir):
        dataset = fo.Dataset.from_videos_patt(video_pattern, name=dataset_dir.split(os.sep)[-1], overwrite=True)
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneVideoLabelsDataset,
        )

    view = dataset.view()
    view.compute_metadata(overwrite=True)

    if detect:
        assert field != 'ground_truth', 'umm..'
    track_field = f'{field}_tracker'

    # --------------------------- Load object detector --------------------------- #

    UNTRACKED_VOCAB = cfg.DATA.UNTRACKED_VOCAB
    TRACKED_VOCAB = cfg.DATA.VOCAB
    VOCAB = TRACKED_VOCAB + UNTRACKED_VOCAB
    CONFIDENCE = cfg.DETIC.CONFIDENCE

    # object detector
    detic = Detic(VOCAB, conf_threshold=CONFIDENCE, masks=True)
    print(detic.labels)

    # ---------------------------- Load object tracker --------------------------- #

    xmem = XMem(cfg.XMEM.CONFIG)
    xmem.track_detections = {}
    xmem.label_counts = defaultdict(lambda: Counter())

    # ----------------------------- Loop over videos ----------------------------- #

    for sample in tqdm.tqdm(view):
        xmem.clear_memory(reset_index=True)
        xmem.track_detections.clear()
        xmem.label_counts.clear()

        # ----------------------------- Loop over frames ----------------------------- #

        video_path = sample.filepath
        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        out_path = f'{dataset_dir}/track_render/{os.path.basename(video_path)}.mp4'

        with XMemSink(out_path, video_info) as s:
            for i, frame, finfo in iter_video(sample):
                
                # --------------------------------- detection -------------------------------- #

                if detect and i % detect_every:
                    finfo[field] = do_detect(detic, frame)

                # --------------------------------- tracking --------------------------------- #

                finfo[track_field] = do_xmem(
                    xmem, frame, finfo[field], TRACKED_VOCAB)

                # ---------------------------------- drawing --------------------------------- #

                detections, labels = fo_to_sv(finfo[track_field], frame.shape[:2])
                s.write_frame(frame, detections, labels)

                finfo.save()

    # ------------------------------ Export dataset ------------------------------ #

    view.export(
        export_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
        label_field=track_field,
    )


# ---------------------------------------------------------------------------- #
#                                Running Models                                #
# ---------------------------------------------------------------------------- #


def do_detect(model, frame):
    outputs = model(frame)
    return detectron_to_fo(outputs, model.labels, frame.shape)


def do_xmem(xmem, frame, gt, track_labels=None, width=280):
    # ------------------------------- Resize frame ------------------------------- #

    ho, wo = frame.shape[:2]
    h, w = int(width / wo * ho), int(width)
    frame = cv2.resize(frame, (w, h))

    # ----------------------- Load detections from FiftyOne ---------------------- #

    gt_mask = gt_labels = None
    if gt is not None:
        gt = [d for d in gt.detections if d.mask is not None]
        
        if track_labels is not None:  # only track certain labels
            gt = [d for d in gt if d.label in track_labels]

        if len(gt):  # get masks and labels
            gt_labels = np.array([d.label for d in gt])
            gt_mask = torch.stack([
                detection2mask(d, (wo,ho), (w, h)) 
                for d in gt
            ]).to(device)

    # ------------------------------- Track objects ------------------------------ #

    pred_mask, track_ids, input_track_ids = xmem(frame, gt_mask)
    boxes = masks_to_boxes(pred_mask)
    boxes = xyxy2xywhn(boxes, frame.shape).tolist()

    if gt_mask is not None:
        for tid, det in zip(input_track_ids, gt):
            xmem.track_detections[tid] = det
        for tid, label in zip(input_track_ids, gt_labels):
            xmem.label_counts[tid].update([label])

    # ------------------------- Convert back to FiftyOne ------------------------- #
    
    detections = []
    for tid, mask, box in zip(track_ids, pred_mask, boxes):
        det = xmem.track_detections[tid].copy()
        detections.append(det)

        det.mask = mask2detection(mask).mask
        det.bounding_box = box
        det.index = tid
        if xmem.label_counts[tid]:
            det.label = xmem.label_counts[tid].most_common()[0][0]
    
    return fo.Detections(detections=detections)
    

if __name__ == '__main__':
    import fire
    fire.Fire(main)
