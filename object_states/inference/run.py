import os
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
from object_states.util.video import DetectionAnnotator, XMemSink, backup_path, get_video_info
from object_states.util.format_convert import detectron_to_sv
from object_states.util.data_output import json_dump
from object_states.util import eta_format as eta
from .vocab import VOCAB

@torch.no_grad()
def run_one(src, tracked_vocab=None, state_db=None, vocab=VOCAB, size=280, fps_down=1, dataset_dir=None, out_dir='outputs', out_path=None, **kw):
    if tracked_vocab is not None:
        vocab['tracked'] = tracked_vocab
    model = Perception(
        vocabulary=vocab,
        state_db_fname=state_db,
        **kw)

    out_path = out_path or f'{out_dir}/{os.path.splitext(os.path.basename(src))[0]}'
    out_path = backup_path(out_path)
    print(out_path)

    name = os.path.splitext(os.path.basename(src))[0]
    treeA = pt.tree(out_dir, {
        'labels': {'{name}.json': 'labels'},
        'output_json': {'{name}_{stream_name}.json': 'output_json'},
        'track_render': {'{name}': 'tracks'},
        # 'manifest.json': 'manifest',
    }, data={'name': name})
    output_json_files = {
        'track': (treeA.output_json.format(stream_name='detic-fast'), []),
        'frame': (treeA.output_json.format(stream_name='detic-image'), []),
    }

    eta_data = eta.eta_base()

    try:
        ann = DetectionAnnotator()
        video_info, WH, WH2 = get_video_info(src, size, fps_down, ncols=2, nrows=2)
        det_frame = hoi_frame = np.zeros((WH[1], WH[0], 3), dtype=np.uint8)

        with XMemSink(out_path, video_info) as s:
            for i, frame in enumerate(tqdm.tqdm(sv.get_video_frames_generator(src))):
                frame = cv2.resize(frame, WH)
                timestamp = i / video_info.fps

                # ---------------------------------- Predict --------------------------------- #

                track_detections, frame_detections, hoi_detections = model.predict(frame, timestamp)

                eta.add_frame(eta_data, i, eta.detectron2_objects(track_detections, frame.shape))

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
                labels = pd.Series([
                    row.idxmax() if len(row.dropna()) else ""
                    for i, row in pd.DataFrame(list(track_detections.pred_states)).iterrows()
                ])
                print(labels)
                keep = ~pd.isna(labels)
                detections = detections[keep]
                labels = labels[keep].tolist()
                state_frame = ann.annotate(frame.copy(), detections, labels, by_track=True)
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
                    frame_data += model.serialize_detections(frame_detections, frame.shape)
                if hoi_detections is not None:
                    frame_data += model.serialize_detections(hoi_detections, frame.shape)
                if frame_data:
                    output_json_files['frame'][1].append({ **meta, 'objects': frame_data })
    finally:
        # -------------------------- Write out final outputs ------------------------- #

        eta.save(eta_data, treeA.labels.format())
        for fname, data in output_json_files.values():
            json_dump(fname, data)




import ipdb
@ipdb.iex
def run(*srcs, **kw):
    for f in srcs:
        run_one(f, **kw)

def main(*a, profile=False, **kw):
    import sys
    import logging
    from tqdm.contrib.logging import logging_redirect_tqdm
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    from xmem.inference.memory_manager import log as xmlog
    xmlog.setLevel(logging.INFO)

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
