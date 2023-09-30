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
from .util.vocab import prepare_vocab

from xmem import XMem
from detic import Detic
from egohos import EgoHos

log = logging.getLogger(__name__)

device = 'cuda'

import ipdb
@ipdb.iex
@torch.no_grad()
def main(config_fname, field=None, detect=None, stop_detect_after=None, skip_every=1, file_path=None):
    cfg = get_cfg(config_fname)
    print(cfg)
    
    root_dataset_dir = dataset_dir = cfg.DATASET.ROOT
    video_pattern = cfg.DATASET.VIDEO_PATTERN

    if file_path:
        dataset_dir = os.path.join(dataset_dir, os.path.basename(file_path))
        video_pattern = file_path

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

    CONFIDENCE = cfg.DETIC.CONFIDENCE
    detect_every = cfg.DETIC.DETECT_EVERY
    hoi_detect_every = 3

    untracked_prompts, UNTRACKED_VOCAB = prepare_vocab(cfg.DATA.UNTRACKED_VOCAB)
    tracked_prompts, TRACKED_VOCAB = prepare_vocab(cfg.DATA.VOCAB)
    PROMPTS = list(untracked_prompts) + list(tracked_prompts)
    VOCAB = list(UNTRACKED_VOCAB) + list(TRACKED_VOCAB)
    print("Prompts:")
    for p, v in zip(PROMPTS, VOCAB):
        print(v, ':', p)
    # input()

    # object detector
    detic = Detic(PROMPTS, conf_threshold=CONFIDENCE, masks=True, max_size=500).cuda().eval()
    detic.labels = np.array(VOCAB)
    print(detic.labels)

    # hand-object interactions
    egohos = EgoHos(mode='obj1', device=device).cuda().eval()
    egohos_classes = np.array(list(egohos.CLASSES))

    # ---------------------------- Load object tracker --------------------------- #

    print(cfg.XMEM.CONFIG)
    xmem = XMem(cfg.XMEM.CONFIG).cuda().eval()
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
    # input('continue:')

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

                    hoi_dets = None
                    # if not i % hoi_detect_every or not i % detect_every:
                    dets = None
                    if not stop_detect_after or i < stop_detect_after:
                        if detect and not i % detect_every:
                            hoi_dets = do_egohos(egohos, frame)
                            finfo['hoi'] = hoi_dets
                            dets = do_detect(detic, frame)
                            finfo[field] = dets
                            detections, labels = fo_to_sv(dets, frame.shape[:2])
                            det_frame = ann.annotate(frame, detections)

                    # --------------------------------- tracking --------------------------------- #

                    dets = do_xmem(xmem, frame, dets, hoi_dets, TRACKED_VOCAB)
                    finfo[track_field] = dets

                    # ---------------------------------- drawing --------------------------------- #

                    detections, labels = fo_to_sv(dets, frame.shape[:2])
                    track_frame = ann.annotate(frame.copy(), detections, labels)
                    s.tracks.write_frame(track_frame, detections, labels, i)
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


def do_egohos(model, frame):
    # get hoi detections
    masks, class_ids = model(frame)
    boxes = masks_to_boxes(masks).int().cpu().numpy().tolist()
    return fo.Detections(detections=[
        fo.Detection(
            mask=mask2detection(m).mask,
            bounding_box=b,
            label=model.CLASSES[cid]
        )
        for m, b, cid in zip(masks, boxes, class_ids)
    ])


def do_detect(model, frame):
    labels = model.labels
    outputs = model(frame)
    log.debug(f"Detected: {labels[outputs['instances'].pred_classes.int().cpu().numpy()]}")
    return detectron_to_fo(outputs, labels, frame.shape)


def do_xmem(xmem, frame, gt, gt_hoi, track_labels=None, width=288):
    # ------------------------------- Resize frame ------------------------------- #

    ho, wo = frame.shape[:2]
    h, w = int(width / wo * ho), int(width)
    frame = cv2.resize(frame, (w, h))

    # ----------------------- Load detections from FiftyOne ---------------------- #

    gt_mask, gt_labels, dets = get_masks_and_labels(gt, (wo,ho), (w, h), track_labels)
    hoi_mask, _, _ = get_masks_and_labels(gt_hoi, (wo,ho), (w, h), ['hand(left)', 'hand(right)'])
    if hoi_mask is not None:
        hoi_mask = hoi_mask.sum(0)

    # ------------------------------- Track objects ------------------------------ #

    pred_mask, track_ids, input_track_ids = xmem(frame, gt_mask, negative_mask=hoi_mask, only_confirmed=True)
    boxes = masks_to_boxes(pred_mask)
    boxes = xyxy2xywhn(boxes, frame.shape).tolist()
    log.debug(f"Tracks: {track_ids} input tracks: {input_track_ids}")

    if gt_mask is not None:
        for tid, det in zip(input_track_ids, dets):
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
    

def get_masks_and_labels(gt, og_shape, pred_shape, filter_labels=None):
    gt_mask = gt_labels = None
    if gt is not None:
        gt = [d for d in gt.detections if d.mask is not None]
        
        if filter_labels is not None:  # only track certain labels
            gt = [d for d in gt if d.label in filter_labels]

        if len(gt):  # get masks and labels
            gt_labels = np.array([d.label for d in gt])
            gt_mask = torch.stack([
                detection2mask(d, og_shape, pred_shape) 
                for d in gt
            ]).cuda()
    return gt_mask, gt_labels, gt



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
        
