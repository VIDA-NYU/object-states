import os
import cv2
import orjson
import torch
from ptgctl.holoframe import load as holoframe_load
from ptgctl.util import parse_epoch_time, format_epoch_time
from redis_record.storage_formats import get_player, get_recorder
from object_states.inference import Perception
from .vocab import VOCAB


@torch.no_grad()
def run_one(name, recording_dir, json_recording_dir, tracked_vocab=None, state_db=None, vocab=VOCAB, detect_every=0.5, suffix=':v3'):
    if tracked_vocab is not None:
        vocab['tracked'] = tracked_vocab
    model = Perception(
        vocabulary=vocab,
        state_db_fname=state_db,
        detect_every_n_seconds=detect_every)

    skipped_3d_labels = {'person'}

    with get_player(name, recording_dir, subset=['main', 'depthlt', 'depthltCal']) as player, \
         get_recorder(recording_dir) as recorder, \
         JsonWriter(name, recording_dir) as json_recorder:
        recorder.ensure_writer(name)
        depthltCal = None
        depthltHist = dicque(maxlen=16)
        for stream_id, ts, message in player:
            tms = int(ts.split('-')[0])
            if stream_id == 'depthltCal':
                depthltCal = holoframe_load(message['d'])
            elif stream_id == 'depthlt':
                depthltHist[tms] = holoframe_load(message['d'])
            elif stream_id == 'main':
                d = holoframe_load(message['d'])
                frame = d['image'][:,:,::-1]
                H = 480
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (int(w*H/h), H))
                # ts = parse_epoch_time(timestamp)
                timestamp = format_epoch_time(ts)
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

                if len(depthltHist) and depthltCal is not None:
                    k = depthltHist.closest(tms)
                    if k is None or abs(k-tms) > 300:
                        tqdm.tqdm.write(f"Skipping 3d as depth was too far away: {k or 0:.0f} - {tms:.0f}: {(k or 0)-tms:.2f}>300")
                    else:
                        objects = get_3d_objects(objects, image_params, depthltHist[k], depthltCal)
                        objects = [d for d in objects if d['label'] not in skipped_3d_labels]
                        outputs['detic:world'] = objects
                else:
                    tqdm.tqdm.write("Skipping 3d as we have no depth data")

                for sid, d in outputs.items():
                    sid = f'{sid}{suffix}'
                    d = orjson.dumps(d, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
                    json_recorder.write(sid, timestamp, d)
                    recorder.write(sid, ts, {b'd': d})


            


def get_3d_objects(objects, image_params, depthlt, depthltCal):
    pts = pt3d.Points3D(
        image_params['shape'][:2], 
        depthlt['image'], 
        depthltCal['lut'],
        depthlt['rig2world'], 
        depthltCal['rig2cam'], 
        image_params['cam2world'],
        image_params['focal'], 
        image_params['principal'],
        generate_point_cloud=False)
    h, w = pt3d.im_shape

    # convert to 3d
    d = outputs['detic:image:for3d']
    xyxy = np.array([o['xyxyn'] for o in objects]).reshape(-1, 4)
    xyxy[:, 0] *= w 
    xyxy[:, 1] *= h 
    xyxy[:, 2] *= w 
    xyxy[:, 3] *= h 
    xyz_center, dist = pt3d.transform_center_withinbbox(xyxy)
    valid = dist < self.max_depth_dist  # make sure the points aren't too far
    log.debug('%d/%d boxes valid. dist in [%f,%f]', valid.sum(), len(valid), dist.min(initial=np.inf), dist.max(initial=0))

    for obj, xyz_center, valid, dist in zip(objects, xyz_center, valid, dist):
        obj['xyz_center'] = xyz_center
        obj['depth_map_dist'] = dist
    return objects



class dicque(OrderedDict):
    def __init__(self, *a, maxlen=0, **kw):
        self._max = maxlen
        super().__init__(*a, **kw)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if self._max > 0 and len(self) > self._max:
            self.popitem(False)

    def closest(self, tms):
        k = min(self, key=lambda t: abs(tms-t), default=None)
        return k


class JsonWriter:
    out_dir = ''
    def __init__(self, name, recording_dir=''):
        self.recording_dir = recording_dir
        self.writers = {}
        if name is not None:
            self.ensure_writer(name)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def ensure_writer(self, name):
        self.close()
        self.out_dir = os.path.join(self.recording_dir, name)
        os.makedirs(self.out_dir, exist_ok=True)

    def ensure_channel(self, sid):
        if sid not in self.writers:
            self.writers[sid] = open(os.path.join(self.out_dir, f'{sid}.json'), 'wb')
            self.writers[sid].write(b'[\n')
            self.writers[sid].i_line = 0
        return self.writers[sid]

    def close(self):
        for w in self.writers.values():
            w.write(b'\n]\n')
            w.close()
        self.writers.clear()

    def write(self, sid, t, data):
        w = self.ensure_channel(sid)
        data = self._add_timestamp_to_json(data, t)
        data = self._serialize(data)
        if w.i_line:
            w.write(b',\n')
        w.write(data)
        w.i_line += 1

    def _add_timestamp_to_json(self, d, ts):
        if ts is not None:
            try:
                if isinstance(d, bytes):
                    d = orjson.loads(d)
                if not isinstance(d, dict):
                    d = {'data': d}
                if 'timestamp' not in d:
                    d['timestamp'] = ts
            except Exception:
                print("error reading data")
                print(d)
                raise
        return d

    def _serialize(self, d):
        try:
            if not isinstance(d, bytes):
                d = orjson.dumps(d, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
        except Exception:
            print("error writing data")
            print(d)
            raise
        return d


# import ipdb
# @ipdb.iex
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
