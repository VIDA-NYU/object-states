import os
import glob
import tqdm
import logging
import pathtrees as pt
import cv2
import numpy as np
import pandas as pd
import torch
import supervision as sv
from object_states.inference import Perception, util
from object_states.inference import util
from object_states.util.video import DetectionAnnotator, XMemSink, get_video_info
# from object_states.util.format_convert import detectron_to_sv
from object_states.util.data_output import json_dump
from object_states.util import eta_format as eta
from .vocab import VOCAB
from ..util.color import green, red, blue, yellow
from IPython import embed


@torch.no_grad()
def run_one(model, src, size=480, dataset_dir=None, overwrite=False, **kw):
    # out_path = out_path or f'{out_dir}/{os.path.splitext(os.path.basename(src))[0]}'
    # out_path = backup_path(out_path)
    # print(out_path)

    name = os.path.splitext(os.path.basename(src))[0]
    treeA = pt.tree(dataset_dir or f'output/{name}', {
        'labels': {'{name}.json': 'labels'},
        'labels2': {'{name}.json': 'labels2'},
        'output_json': {'{name}_{stream_name}.json': 'output_json'},
        'track_render': {'{name}': 'tracks'},
        # 'manifest.json': 'manifest',
    }, data={'name': name})
    output_json_files = {
        'track': (treeA.output_json.format(stream_name='detic-image'), []),
        'frame': (treeA.output_json.format(stream_name='detic-image-misc'), []),
    }
    # embed()

    if treeA.labels.is_file():
        if not overwrite:# and not treeA.labels2.is_dir():
            print(green("Already exists!"), treeA.labels)
            return 
    print(blue("Doing"), treeA.labels)

    eta_data = eta.eta_base()

    model.detector.xmem.clear_memory()

    try:
        ann = DetectionAnnotator()
        video_info, WH, WH2 = get_video_info(src, size, ncols=2, nrows=2)
        det_frame = hoi_frame = np.zeros((WH[1], WH[0], 3), dtype=np.uint8)

        with XMemSink(str(treeA.tracks), video_info) as s:
            pbar = tqdm.tqdm(sv.get_video_frames_generator(src), total=video_info.total_frames)
            for i, frame in enumerate(pbar):
                frame = cv2.resize(frame, WH)
                timestamp = i / video_info.fps

                # ---------------------------------- Predict --------------------------------- #

                track_detections, frame_detections, hoi_detections = model.predict(frame, timestamp)

                eta.add_frame(eta_data, i, eta.detectron2_objects(track_detections, frame.shape))
                pbar.set_description(
                    f'{len(track_detections)} '
                    f'{len(frame_detections) if frame_detections is not None else None} '
                    f'{len(hoi_detections) if hoi_detections is not None else None} ')

                # -------------------------------- Draw frames ------------------------------- #

                # Draw frame detections
                if frame_detections is not None:
                    detections, labels = detectron_to_sv(frame_detections, frame.shape[:2])
                    det_frame = ann.annotate(frame.copy(), detections, labels)

                # Draw HOI Detections
                if hoi_detections is not None:
                    detections, labels = detectron_to_sv(hoi_detections, frame.shape[:2])
                    hoi_frame = ann.annotate(frame.copy(), detections, labels)

                # Draw track detections
                detections, labels = detectron_to_sv(track_detections, frame.shape[:2])
                track_frame = ann.annotate(frame.copy(), detections, labels, by_track=True)
                # print(track_detections.pred_states)
                # if hasattr(track_detections, 'pred_states'):
                state_labels = pd.Series([
                    row.idxmax() if len(row.dropna()) else ""
                    for i, row in pd.DataFrame(list(track_detections.pred_states)).iterrows()
                ])
                # print(labels)
                keep = ~pd.isna(state_labels)
                state_detections = detections[keep]
                state_labels = state_labels[keep].tolist()
                state_frame = ann.annotate(frame.copy(), state_detections, state_labels, by_track=True)
                # else:
                #     state_frame = frame.copy()

                # -------------------------------- Write frames ------------------------------ #

                s.tracks.write_frame(track_frame, detections, labels, i)
                s.write_frame(np.vstack([
                    np.hstack([track_frame, det_frame]),
                    np.hstack([state_frame, hoi_frame])
                ]))

                # ----------------------------- Serialize outputs ---------------------------- #

                meta = { 'timestamp': timestamp, 'image_shape': frame.shape }

                # write out track predictions
                track_data = model.serialize_detections(track_detections, frame.shape)
                output_json_files['track'][1].append({ **meta, 'objects': track_data })
                
                # write out frame predictions
                frame_data = []
                if frame_detections is not None:
                    frame_data += track_data
                    frame_data += model.serialize_detections(frame_detections, frame.shape)
                if hoi_detections is not None:
                    frame_data += model.serialize_detections(hoi_detections, frame.shape)
                if frame_data:
                    output_json_files['frame'][1].append({ **meta, 'objects': frame_data })
    finally:
        # -------------------------- Write out final outputs ------------------------- #

        eta.save(eta_data, treeA.labels2.format())
        for fname, data in output_json_files.values():
            print(yellow('Writing'), fname)
            json_dump(fname, data)



def detectron_to_sv(outputs, classes=None):
    outputs = outputs.to('cpu')
    detections = sv.Detections(
        xyxy=outputs.pred_boxes.tensor.numpy(),
        mask=outputs.pred_masks.numpy() if outputs.has('pred_masks') else None,
        class_id=outputs.pred_classes.int().numpy() if outputs.has('pred_classes') else np.zeros(len(outputs), dtype=int),
        tracker_id=outputs.track_ids.int().numpy() if outputs.has('track_ids') else None,
        confidence=outputs.scores.numpy() if outputs.has('scores') else None,
    )
    labels = (
        outputs.pred_labels if outputs.has('pred_labels') else 
        np.asarray(classes)[detections.class_id] if detections.class_id is not None else 
        None)
    return detections, labels




import ipdb
@ipdb.iex
def run(*srcs, tracked_vocab=None, state_db=None, vocab=VOCAB, additional_roi_heads=None, detic_config_key=None, detect_every=0.5, conf_threshold=0.3, **kw):
    if tracked_vocab is not None:
        vocab['tracked'] = tracked_vocab

    model = Perception(
        vocabulary=vocab,
        state_db_fname=state_db,
        state_key='mod_state',
        additional_roi_heads=additional_roi_heads,
        detic_config_key=detic_config_key,
        detect_every_n_seconds=detect_every,
        conf_threshold=conf_threshold,
        filter_tracked_detections_from_frame=False,
    )
    for f in srcs:
        f = glob.glob(os.path.join(f, '*')) if os.path.isdir(f) else [f]
        for fi in f:
            run_one(model, fi, **kw)

def main(*a, profile=False, **kw):
    import sys
    import logging
    from tqdm.contrib.logging import logging_redirect_tqdm
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    from xmem.inference.memory_manager import log as xmlog
    xmlog.setLevel(logging.DEBUG)

    with logging_redirect_tqdm():
        if not profile:
            return run(*a, **kw)
        from pyinstrument import Profiler
        prof = Profiler(async_mode='disabled')
        try:
            with prof:
                return run(*a, **kw)
        finally:
            prof.print()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
