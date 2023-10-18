import cv2
import torch
from ptgctl.holoframe import holoframe_load
from ptgctl.util import parse_epoch_time
from redis_record.storage_formats import get_player, get_recorder
from object_states.inference import Perception
from .vocab import VOCAB


@torch.no_grad()
def run_one(name, recording_dir, tracked_vocab=None, state_db=None, vocab=VOCAB, detect_every=0.5):
    if tracked_vocab is not None:
        vocab['tracked'] = tracked_vocab
    model = Perception(
        vocabulary=vocab,
        state_db_fname=state_db,
        detect_every_n_seconds=detect_every)


    with get_player(name, recording_dir, subset=['main']) as player, \
         get_recorder(name, recording_dir) as recorder:
        for stream_id, timestamp, message in player:
            assert stream_id == 'main'
            d = holoframe_load(message)
            frame = d['image'][:,:,::-1]
            H = 480
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (int(w*H/h), H))
            ts = parse_epoch_time(timestamp)
            track_detections, frame_detections, hoi_detections = model.predict(small_frame, ts)
            outputs = {
                "detic:image": model.serialize_detections(track_detections, frame.shape, include_mask=True),
                # "detic:image": self.perception.serialize_detections(track_detections, frame.shape),
            }

            image_params = {
                'shape': frame.shape,
                'focal': [d['focalX'], d['focalY']],
                'principal': [d['principalX'], d['principalY']],
                'cam2world': d['cam2world'].tolist(),
            }
            outputs['detic:image:for3d'] = {
                'objects': outputs['detic:image'],
                'image': image_params,
                'epoch_timestamp': ts,
                'timestamp': timestamp,
            }
            for sid, d in outputs.items():
                recorder.write(sid, ts, d)


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
