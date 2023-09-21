from collections import defaultdict, Counter

import os
import cv2
import tqdm
import logging
import numpy as np
import torch
import fiftyone as fo
import supervision as sv
from torchvision.ops import masks_to_boxes

from .config import get_cfg
from .util.video import XMemSink, DetectionAnnotator, iter_video
from .util.format_convert import *

from xmem import XMem
from detic import Detic

log = logging.getLogger(__name__)

device = 'cuda'

import ipdb
@ipdb.iex
@torch.no_grad()
def main(config_fname, field=None, detect=None, stop_detect_after=None, skip_every=1):
    cfg = get_cfg(config_fname)
    
    dataset_dir = cfg.DATASET.ROOT
    video_pattern = cfg.DATASET.VIDEO_PATTERN

    if not field:
        field = 'detections'
        if detect is None:
            detect = True
            log.info("No field provided. Will compute detections.")
    if detect:
        assert field != 'ground_truth', 'umm..'
        log.info(f"Putting detections in {field}")
    track_field = f'{field}_tracker'

    log.info(f"Putting tracks in {track_field}")

    # ------------------------------- Load dataset ------------------------------- #

    if not os.path.exists(f'{dataset_dir}/manifest.json'):
        log.info(f"Creating dataset from {video_pattern}")
        dataset = fo.Dataset.from_videos_patt(video_pattern, name=dataset_dir.split(os.sep)[-1], overwrite=True)
    else:
        log.info(f"Loading existing dataset from {dataset_dir}")
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneVideoLabelsDataset,
        )
    # from IPython import embed
    # embed()

    view = dataset.view()
    view.compute_metadata(overwrite=True)
    dataset.add_frame_field(track_field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

    # --------------------------- Load object detector --------------------------- #

    UNTRACKED_VOCAB = cfg.DATA.UNTRACKED_VOCAB
    TRACKED_VOCAB = cfg.DATA.VOCAB
    VOCAB = TRACKED_VOCAB + UNTRACKED_VOCAB
    CONFIDENCE = cfg.DETIC.CONFIDENCE
    detect_every = cfg.DETIC.DETECT_EVERY

    # object detector
    detic = Detic(VOCAB, conf_threshold=CONFIDENCE, masks=True, max_size=500)
    print(detic.labels)

    # ---------------------------- Load object tracker --------------------------- #

    print(cfg.XMEM.CONFIG)
    xmem = XMem(cfg.XMEM.CONFIG).cuda()
    xmem.track_detections = {}
    xmem.label_counts = defaultdict(lambda: Counter())

    # --------------------------- Check previous videos -------------------------- #

    log.info("Which videos have we already tracked things in?")
    for sample in tqdm.tqdm(view):
        if track_field in sample.frames.field_names:
            idxs = [i for i in sample.frames if sample.frames[i].get_field(track_field) is not None]
            n_detections = len(idxs)
            
            if n_detections:
                log.info(f'{sample.filepath} has {n_detections}. {min(idxs)}-{max(idxs)} out of {len(sample.frames)}')
    input('continue:')

    # ----------------------------- Loop over videos ----------------------------- #

    try:

        for sample in tqdm.tqdm(view):
            video_path = sample.filepath

            try:
                if track_field in sample.frames.field_names:
                    # det_frame = next((i for i in sample.frames if sample.frames[i].get_field(track_field) is not None), None)
                    if sample.frames.last().get_field(track_field) is not None:
                        tqdm.tqdm.write(f"\n\nSkipping {video_path}\n\n")
                        continue
            except Exception as e:
                log.exception(e)

            xmem.clear_memory(reset_index=True)
            xmem.track_detections.clear()
            xmem.label_counts.clear()

            # ----------------------------- Loop over frames ----------------------------- #

            
            video_info = sv.VideoInfo.from_video_path(video_path=video_path)
            out_dir = f'{dataset_dir}/track_render/{os.path.basename(video_path)}'
            ann = DetectionAnnotator()

            det_frame = np.zeros((video_info.height, video_info.width, 3), dtype=np.uint8)
            video_info.width *= 2
            with XMemSink(out_dir, video_info) as s:
                for i, frame, finfo in iter_video(sample):
                    if skip_every and i % skip_every and i % detect_every: continue
                    
                    # --------------------------------- detection -------------------------------- #

                    dets = None
                    if not stop_detect_after or i < stop_detect_after:
                        if detect and not i % detect_every:
                            dets = do_detect(detic, frame)
                            finfo[field] = dets
                            detections, labels = fo_to_sv(dets, frame.shape[:2])
                            det_frame = ann.annotate(frame, detections, labels)

                    # --------------------------------- tracking --------------------------------- #

                    dets = do_xmem(xmem, frame, dets, TRACKED_VOCAB)
                    finfo[track_field] = dets

                    # ---------------------------------- drawing --------------------------------- #

                    detections, labels = fo_to_sv(dets, frame.shape[:2])
                    track_frame = ann.annotate(frame.copy(), detections, labels)
                    s.tracks.write_frame(frame, detections)
                    s.write_frame(np.hstack([track_frame, det_frame]))

                    try:
                        finfo.save()
                    except Exception as e:
                        log.exception(e)
            try:
                sample.save()
            except Exception as e:
                log.exception(e)

    finally:

        # ------------------------------ Export dataset ------------------------------ #

        view.export(
            export_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneVideoLabelsDataset,
            label_field=f'frames.{track_field}',
        )


# ---------------------------------------------------------------------------- #
#                                Running Models                                #
# ---------------------------------------------------------------------------- #


def do_detect(model, frame):
    outputs = model(frame)
    log.debug(f"Detected: {model.labels[outputs['instances'].pred_classes.int().cpu().numpy()]}")
    return detectron_to_fo(outputs, model.labels, frame.shape)


def do_xmem(xmem, frame, gt, track_labels=None, width=288):
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
            ]).cuda()
            log.debug(f"Using detections: {gt_labels}")

    # ------------------------------- Track objects ------------------------------ #

    pred_mask, track_ids, input_track_ids = xmem(frame, gt_mask, only_confirmed=True)
    boxes = masks_to_boxes(pred_mask)
    boxes = xyxy2xywhn(boxes, frame.shape).tolist()
    log.debug(f"Tracks: {track_ids} input tracks: {input_track_ids}")

    if gt_mask is not None:
        for tid, det in zip(input_track_ids, gt):
            xmem.track_detections[tid] = det
        for tid, label in zip(input_track_ids, gt_labels):
            xmem.label_counts[tid].update([label])

    # ------------------------- Convert back to FiftyOne ------------------------- #
    
    detections = []
    for tid, mask, box in zip(track_ids, pred_mask.cpu(), boxes):
        det = xmem.track_detections[tid].copy()
        detections.append(det)

        det.mask = mask2detection(mask).mask
        det.bounding_box = box
        det.index = tid
        if xmem.label_counts[tid]:
            det.label = xmem.label_counts[tid].most_common()[0][0]
    
    return fo.Detections(detections=detections)
    



if __name__ == '__main__':
    import sys
    import logging
    from tqdm.contrib.logging import logging_redirect_tqdm
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    import xmem
    from xmem.inference.memory_manager import log as xmlog
    xmlog.setLevel(logging.INFO)

    with logging_redirect_tqdm():

        # from pyinstrument import Profiler
        # prof = Profiler(async_mode='disabled')
        # try:
        #     with prof:
        import fire
        fire.Fire(main)
        # finally:
        #     prof.print()
        
