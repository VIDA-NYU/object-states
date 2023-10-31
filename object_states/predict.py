from collections import defaultdict, Counter

import os
import glob
import cv2
import tqdm
import logging
import numpy as np
import torch
import fiftyone as fo
import supervision as sv
from torchvision.ops import masks_to_boxes

from . import _patch
from .config import get_cfg
from .util.video import XMemSink, DetectionAnnotator, iter_video
from .util.format_convert import *
from .util.vocab import prepare_vocab

from xmem import XMem
from detic import Detic
from detic.inference import asymmetric_nms as detic_asymmetric_nms, load_classifier
from object_states.util.nms import asymmetric_nms
from egohos import EgoHos

from IPython import embed

log = logging.getLogger(__name__)

device = 'cuda'

import ipdb
@ipdb.iex
@torch.no_grad()
def main(config_fname, *files_to_predict, field=None, detect=None, stop_detect_after=None, skip_every=1, file_path=None):
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

    # XXX: This does not work. it discards your predicions when its done.
    # log.info(f"Creating dataset from {video_pattern}")
    # dataset = fo.Dataset.from_videos_patt(video_pattern, name=dataset_dir.split(os.sep)[-1], overwrite=True)
    # if os.path.exists(f'{dataset_dir}/manifest.json'):
    #     log.info(f"Loading existing dataset from {dataset_dir}")
    #     dataset.merge_dir(
    #         dataset_dir=dataset_dir,
    #         dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    #     )
    dataset_name = dataset_dir.split(os.sep)[-1]
    if not os.path.exists(f'{dataset_dir}/manifest.json'):
        log.info(f"Creating dataset from {video_pattern}")
        dataset = fo.Dataset.from_videos_patt(video_pattern, name=dataset_name, overwrite=True)
    else:
        log.info(f"Loading existing dataset from {dataset_dir}")
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneVideoLabelsDataset,
            name=dataset_name,
        )
        existing_paths = {os.path.basename(s.filepath) for s in dataset}
        for f in glob.glob(video_pattern):
            if os.path.basename(f) not in existing_paths:
                print(f"Adding missing sample {f}")
                dataset.add_sample(fo.Sample(f))
    assert len(dataset), "wtf"
        
    # dataset.persist = True
    # embed()

    dataset.add_frame_field(track_field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
    dataset.compute_metadata(overwrite=True)
    view = dataset.view()
    assert len(dataset), "wtf"

    # --------------------------- Load object detector --------------------------- #

    CONFIDENCE = cfg.DETIC.CONFIDENCE
    detect_every_secs = cfg.DETIC.DETECT_EVERY_SECS
    hoi_detect_every = 3

    if isinstance(cfg.DATA.UNTRACKED_VOCAB, str):
        # PROMPTS = cfg.DATA.UNTRACKED_VOCAB
        # VOCAB = None
        tracked_prompts, TRACKED_VOCAB = prepare_vocab(cfg.DATA.VOCAB)
        _, meta, _ = load_classifier(cfg.DATA.UNTRACKED_VOCAB)
        PROMPTS = list(meta.thing_classes) + [p for p in tracked_prompts if p not in meta.thing_classes]
        VOCAB = list(meta.thing_classes) + [TRACKED_VOCAB[i] for i, p in enumerate(tracked_prompts) if p not in meta.thing_classes]
    else:
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
    if VOCAB is not None:
        detic.labels = np.array(VOCAB)
    print(detic.labels)
    assert not (set(TRACKED_VOCAB) - set(detic.labels)), f"these tracked labels dont exist: {set(TRACKED_VOCAB) - set(detic.labels)}"

    # hand-object interactions
    egohos = EgoHos(mode='obj1', device=device).cuda().eval()
    egohos_classes = np.array(list(egohos.CLASSES))

    # ---------------------------- Load object tracker --------------------------- #

    print(cfg.XMEM.CONFIG)
    xmem = XMem(cfg.XMEM.CONFIG).cuda().eval()
    xmem.track_detections = {}
    xmem.label_counts = defaultdict(lambda: Counter())
    XMEM_WIDTH = cfg.XMEM.FRAME_SIZE

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
    files_to_predict = [os.path.abspath(f) for f in files_to_predict]
    try:
        try:
            samples = [s for s in view]
            for sample in tqdm.tqdm(view):
                assert len(view), "wtf"
                video_path = sample.filepath
                single_export_dir = os.path.join(dataset_dir, 'single', os.path.basename(video_path))
                if files_to_predict and os.path.abspath(video_path) not in files_to_predict:
                    tqdm.tqdm.write(f"skipping {video_path}...")
                    continue
                if os.path.isdir(single_export_dir):
                    tqdm.tqdm.write(f"skipping {video_path} exists...")
                    continue

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
                HW = video_info.height, video_info.width
                det_frame = hoi_frame = np.zeros((*HW, 3), dtype=np.uint8)
                video_info.width *= 3
                out_dir = f'{dataset_dir}/track_render/{os.path.basename(video_path)}'
                ann = DetectionAnnotator()
                detect_every = int(detect_every_secs * video_info.fps)

                with XMemSink(out_dir, video_info) as s:
                    for i, frame, finfo in iter_video(sample):
                        if skip_every and i % skip_every and i % detect_every: continue
                        
                        # --------------------------------- detection -------------------------------- #

                        # if not i % hoi_detect_every or not i % detect_every:
                        dets = hoi_dets = None
                        if not stop_detect_after or i < stop_detect_after:
                            if detect and not i % detect_every:
                                hoi_dets = do_egohos(egohos, frame)
                                finfo['hoi'] = hoi_dets
                                detections, labels = fo_to_sv(hoi_dets, frame.shape[:2], classes=egohos.CLASSES)
                                hoi_frame = ann.annotate(frame.copy(), detections, labels)

                                dets = do_detect(detic, frame)
                                finfo[field] = dets
                                detections, labels = fo_to_sv(dets, frame.shape[:2], classes=detic.labels)
                                det_frame = ann.annotate(frame.copy(), detections, labels)

                        # --------------------------------- tracking --------------------------------- #

                        dets = do_xmem(xmem, frame, dets, hoi_dets, TRACKED_VOCAB, width=XMEM_WIDTH)
                        finfo[track_field] = dets

                        # ---------------------------------- drawing --------------------------------- #

                        detections, labels = fo_to_sv(dets, frame.shape[:2])
                        track_frame = ann.annotate(frame.copy(), detections, labels, by_track=True)
                        s.tracks.write_frame(track_frame, detections, labels, i)
                        s.write_frame(np.hstack([track_frame, det_frame, hoi_frame]))

                        try:
                            finfo.save()
                        except Exception as e:
                            log.exception(e)
                try:
                    sample.save()
                    # safety measure
                    d=fo.Dataset()
                    d.add_sample(sample)
                    os.makedirs(f'{dataset_dir}/single', exist_ok=True)
                    d.export(
                        export_dir=single_export_dir,
                        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
                        label_field=f'frames.{track_field}',
                        export_media="symlink",
                    )
                except Exception as e:
                    log.exception(e)
                # raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        finally:
            # ------------------------------ Export dataset ------------------------------ #
            print("aaaaaaaaa")
            view.export(
                export_dir=dataset_dir,
                dataset_type=fo.types.FiftyOneVideoLabelsDataset,
                label_field=f'frames.{track_field}',
                export_media="symlink",
            )
            if detect:
                view.export(
                    export_dir=os.path.join(dataset_dir, 'detections'),
                    dataset_type=fo.types.FiftyOneVideoLabelsDataset,
                    label_field=f'frames.{field}',
                    export_media="symlink",
                )
            # embed()

            # for sample in view:
            #     if sample.has_frame_field("frames.ground_truth_tracker"):
            #         results = sample.evaluate_detections(
            #             f"frames.{track_field}",
            #             gt_field="frames.ground_truth_tracker",
            #             eval_key="eval_tracker",
            #         )
    finally:
        if not os.path.isdir(f'{dataset_dir}/labels'):
            print("asdfasdfasdf")
            dataset.export(
                export_dir=dataset_dir,
                dataset_type=fo.types.FiftyOneVideoLabelsDataset,
                label_field=f'frames.{track_field}',
            )
        # embed()


# ---------------------------------------------------------------------------- #
#                                Running Models                                #
# ---------------------------------------------------------------------------- #


def do_egohos(model, frame):
    # get hoi detections
    masks, class_ids = model(frame)
    boxes = xyxy2xywhn(masks_to_boxes(masks), frame.shape).cpu().numpy().tolist()
    if len(class_ids):
        log.info(f"HOS: {model.CLASSES[class_ids]}")
    return fo.Detections(detections=[
        fo.Detection(
            mask=mask2detection(m).mask,
            bounding_box=b,
            label=model.CLASSES[cid]
        )
        for m, b, cid in zip(masks, boxes, class_ids)
    ])


def do_detect(model, frame, track_labels=None, iou_threshold=0.85):
    labels = model.labels
    outputs = model(frame)
    # selected_indices, _ = asymmetric_nms(outputs['instances'].pred_boxes.tensor, outputs['instances'].scores)
    # filtered = set(range(len(outputs['instances']))) - set(selected_indices.tolist())
    # if filtered:
    #     log.info(f"Filtered: {labels[outputs['instances'].pred_classes[list(filtered)].int().cpu().numpy()]}")
    # outputs['instances'] = outputs['instances'][selected_indices]

    instances = outputs['instances']
    instances.pred_labels = labels[instances.pred_classes.int().cpu().numpy()]
    # filtered, overlap = asymmetric_nms(instances.pred_boxes.tensor, instances.scores, iou_threshold=iou_threshold)
    # filtered_instances = instances[filtered.cpu().numpy()]
    # for i, i_ov in enumerate(overlap):
    #     if not len(i_ov): continue
    #     # get overlapping instances
    #     overlap_insts = instances[i_ov.cpu().numpy()]
    #     log.info(f"object {filtered_instances.pred_classes[i]} filtered {overlap_insts.pred_classes}")

    #     # merge overlapping detections with the same label
    #     overlap_insts = overlap_insts[overlap_insts.pred_classes == filtered_instances.pred_classes[i]]
    #     if len(overlap_insts):
    #         log.info(f"object {filtered_instances.pred_classes[i]} merged {len(overlap_insts)}")
    #         filtered_instances.pred_masks[i] |= torch.maximum(
    #             filtered_instances.pred_masks[i], 
    #             overlap_insts.pred_masks.max(0).values)
            


    # filter out objects completely inside another object
    obj_priority = torch.from_numpy(np.isin(instances.pred_labels, track_labels)).int() if track_labels is not None else None
    filtered, overlap = asymmetric_nms(instances.pred_boxes.tensor, instances.scores, obj_priority, iou_threshold=iou_threshold)
    filtered_instances = instances[filtered.cpu().numpy()]
    for i, i_ov in enumerate(overlap):
        if not len(i_ov): continue
        # get overlapping instances
        overlap_insts = instances[i_ov.cpu().numpy()]
        log.info(f"object {filtered_instances.pred_labels[i]} filtered {overlap_insts.pred_labels}")

        # merge overlapping detections with the same label
        overlap_insts = overlap_insts[overlap_insts.pred_labels == filtered_instances.pred_labels[i]]
        if len(overlap_insts):
            log.info(f"object {filtered_instances.pred_labels[i]} merged {len(overlap_insts)}")
            filtered_instances.pred_masks[i] |= torch.maximum(
                filtered_instances.pred_masks[i], 
                overlap_insts.pred_masks.max(0).values)
    outputs['instances'] = filtered_instances

    log.debug(f"Detected: {labels[outputs['instances'].pred_classes.int().cpu().numpy()]}")
    return detectron_to_fo(outputs, labels, frame.shape)


def do_xmem(xmem, frame, gt, gt_hoi, track_labels=None, width=280):
    # ------------------------------- Resize frame ------------------------------- #

    ho, wo = frame.shape[:2]
    h, w = int(width / wo * ho), int(width)
    frame = cv2.resize(frame, (w, h))

    # ----------------------- Load detections from FiftyOne ---------------------- #

    gt_mask, gt_labels, dets, neg_gt_mask = get_masks_and_labels(gt, (wo,ho), (w, h), track_labels, return_neg_mask=True)
    hoi_mask, _, _ = get_masks_and_labels(gt_hoi, (wo,ho), (w, h), ['hand(left)', 'hand(right)'])
    neg_mask = None
    if hoi_mask is not None:
        hoi_mask = hoi_mask.sum(0)
        if hoi_mask.float().sum() > 0:
            neg_mask = hoi_mask
    if neg_gt_mask is not None:
        neg_gt_mask = neg_gt_mask.sum(0)
        if neg_gt_mask.float().sum() > 0:
            neg_mask = neg_gt_mask if neg_mask is None else neg_gt_mask | neg_mask
    

    # ------------------------------- Track objects ------------------------------ #

    pred_mask, track_ids, input_track_ids = xmem(frame, gt_mask, negative_mask=neg_mask, only_confirmed=True)
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
            det.label, count = xmem.label_counts[tid].most_common()[0]
            # det.confidence = count / sum(xmem.label_counts[tid].values())
     
    return fo.Detections(detections=detections)
    

def get_masks_and_labels(gt, og_shape, pred_shape, filter_labels=None, return_neg_mask=False):
    gt_mask = gt_labels = neg_mask = None
    if gt is not None:
        gt = [d for d in gt.detections if d.mask is not None]
        neg_gt = []
        
        if filter_labels is not None:  # only track certain labels
            neg_gt = [d for d in gt if d.label not in filter_labels]
            gt = [d for d in gt if d.label in filter_labels]

        gt_conf = np.array([d.confidence for d in gt])
        if len(gt):  # get masks and labels
            gt_labels = np.array([d.label for d in gt])
            gt_mask = torch.stack([
                detection2mask(d, og_shape, pred_shape) 
                for d in gt
            ]).cuda()
        if len(neg_gt) and return_neg_mask:
            neg_mask = torch.stack([
                detection2mask(d, og_shape, pred_shape) 
                for d in neg_gt
            ]).cuda()
        log.info(f"mask labels: {gt_labels} {gt_conf}")
    if return_neg_mask:
        return gt_mask, gt_labels, gt, neg_mask
    return gt_mask, gt_labels, gt




# def asymmetric_nms(boxes, scores, iou_threshold=0.99):
#     maxi = torch.maximum
#     area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

#     # Sort boxes by their confidence scores in descending order
#     indices = torch.argsort(area, descending=True)
#     # indices = np.argsort(scores)[::-1]
#     boxes = boxes[indices]
#     scores = scores[indices]

#     selected_indices = []
#     overlap_indices = []
#     while len(boxes) > 0:
#         # Pick the box with the highest confidence score
#         b = boxes[0]
#         selected_indices.append(indices[0])

#         # Calculate IoU between the picked box and the remaining boxes
#         zero = torch.tensor([0], device=boxes.device)
#         intersection_area = (
#             torch.maximum(zero, torch.minimum(b[2], boxes[1:, 2]) - torch.maximum(b[0], boxes[1:, 0])) * 
#             torch.maximum(zero, torch.minimum(b[3], boxes[1:, 3]) - torch.maximum(b[1], boxes[1:, 1]))
#         )
#         smaller_box_area = torch.minimum(area[0], area[1:])
#         iou = intersection_area / (smaller_box_area + 1e-7)

#         # Filter out boxes with IoU above the threshold

#         overlap_indices.append(indices[torch.where(iou > iou_threshold)[0]])
#         filtered_indices = torch.where(iou <= iou_threshold)[0]
#         indices = indices[filtered_indices + 1]
#         boxes = boxes[filtered_indices + 1]
#         scores = scores[filtered_indices + 1]
#         area = area[filtered_indices + 1]

#     selected_indices = (
#         torch.stack(selected_indices) if selected_indices else 
#         torch.zeros([0], dtype=torch.int32, device=boxes.device))
#     return selected_indices, overlap_indices





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
        
