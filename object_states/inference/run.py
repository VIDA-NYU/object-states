import os
import tqdm
import logging

import cv2
import numpy as np
import pandas as pd
import torch
import supervision as sv
from object_states.inference import Perception, util
from object_states.inference import util
from object_states.util.video import DetectionAnnotator, XMemSink, backup_path
from object_states.util.format_convert import detectron_to_sv
from object_states.util.data_output import json_dump
from .vocab import VOCAB

@torch.no_grad()
def run_one(src, tracked_vocab=None, state_db=None, vocab=VOCAB, detect_every=0.5, size=280, fps_down=1, out_dir='outputs', out_path=None):
    if tracked_vocab is not None:
        vocab['tracked'] = tracked_vocab
    model = Perception(
        vocabulary=vocab,
        state_db_fname=state_db,
        detect_every_n_seconds=detect_every)

    out_path = out_path or f'{out_dir}/{os.path.splitext(os.path.basename(src))[0]}'
    out_path = backup_path(out_path)
    print(out_path)

    output_json_files = {
        'track': (f'{out_path}/detic-fast.json', []),
        'frame': (f'{out_path}/detic-image.json', []),
    }

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

        for fname, data in output_json_files.values():
            json_dump(fname, data)


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1, render_scale=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    print(f'Input Video {src}')
    print(f"Original size: {[video_info.width, video_info.height]}")
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height
    video_info.width *= render_scale
    video_info.height *= render_scale
    WH2 = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"size: {WH} {WH2} grid={ncols}x{nrows}  fps: {video_info.fps} ({fps_down or 1}x)")
    return video_info, WH, WH2


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
