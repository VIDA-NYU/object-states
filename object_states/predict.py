from . import _patch

import cv2
import numpy as np
import torch
import fiftyone as fo
import supervision as sv
from xmem import XMem
from .util.video import XMemSink, iter_video
from .util.format_convert import *


import ipdb
@ipdb.iex
def main(dataset_dir, field='detections', detect=False, detect_every=6, track_labels=None):
    
    # ------------------------------- Load dataset ------------------------------- #

    dataset = fo.Dataset.from_dir(
        dataset_dir=data_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    )

    view = dataset.view()
    view.compute_metadata(overwrite=True)

    # ---------------------------- Do object tracking ---------------------------- #

    if detect:
        assert field != 'ground_truth', 'umm..'
    track_field = f'{field}_tracker'

    xmem = XMem({})
    xmem.track_detections = {}
    xmem.label_counts = defaultdict(lambda: Counter())

    for sample in tqdm.tqdm(view):
        xmem.clear_memory(reset_index=True)
        xmem.track_detections.clear()
        xmem.label_counts.clear()

        with XMemSink(out_path, video_info) as s:
            for i, frame, finfo in iter_video(sample):
                
                # --------------------------------- detection -------------------------------- #

                if detect and i % detect_every:
                    finfo[field] = do_detect(detic, frame)

                # --------------------------------- tracking --------------------------------- #

                finfo[track_field] = do_xmem(
                    xmem, frame, finfo[field], track_labels)

                # ---------------------------------- drawing --------------------------------- #

                detections, labels = fo_to_sv(finfo[track_field], frame.shape[:2])
                s.write_frame(frame, detections, labels)

                finfo.save()

    # ------------------------------ Export dataset ------------------------------ #

    view.export(
        export_dir=data_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
        label_field=track_field,
    )


# ---------------------------------------------------------------------------- #
#                                Running Models                                #
# ---------------------------------------------------------------------------- #


def do_detect(model, frame):
    outputs = model(frame)
    return detectron_to_fo(outputs, frame.shape)


def do_xmem(xmem, frame, fo_detections, track_labels=None, width=280):
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
            track_detections[tid] = det
        for tid, label in zip(input_track_ids, gt_labels):
            xmem.label_counts[tid].update([label])

    # ------------------------- Convert back to FiftyOne ------------------------- #
    
    detections = []
    for tid, mask, box in zip(track_ids, pred_mask, boxes):
        det = track_detections[tid].copy()
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
