import os
import glob
import orjson
import tqdm
import numpy as np
import cv2

import logging

log = logging.getLogger(__name__)


TYPES = {
    bool: "eta.core.data.BooleanAttribute",
    (int, float): "eta.core.data.NumericAttribute",
    str: "eta.core.data.CategoricalAttribute",
}


# ---------------------------------------------------------------------------- #
#                                ETA Primatives                                #
# ---------------------------------------------------------------------------- #

def eta_base(attrs=None):
    return {
        **_maybe_key("attrs", attrs),
        "frames": {},
    }

def attr(name, value, **kw):
    return {
        "type": next(v for t, v in TYPES.items() if isinstance(value, t)),
        "name": name,
        "value": value,
        **kw
    }

def frame(frame_number, objects, attrs=None):
    return {
        "frame_number": frame_number,
        **_maybe_key("attrs", attrs),
        **_maybe_key("objects", objects),
    }

def object(index, label, bbox, mask, confidence, shape=None, attrs=None, as_polylines=True):
    return {
        "index": index,
        "label": label,
        "polylines": binary_mask_to_polygon(mask) if as_polylines else binary_to_bounded_mask(mask, bbox),
        "confidence": confidence,
        "bounding_box": xyxy_to_box(bbox, shape or mask.shape),
        **_maybe_key("attrs", attrs),
    }


# ------------------------------ ETA Operations ------------------------------ #

def add_frame(base, frame_number, objects=None, attrs=None):
    base['frames'][str(frame_number+1)] = {
        "frame_number": frame_number+1,
        **_maybe_key("attrs", attrs),
        **_maybe_key("objects", objects),
    }
    return base


# ---------------------------------------------------------------------------- #
#                                 Object Utils                                 #
# ---------------------------------------------------------------------------- #

def _maybe_key(key, value):
    return {key: {key: value or []}} if value else {}

def nonone(d):
    return {k: v for k, v in d.items() if v is not None}

# --------------------------------- Polyline --------------------------------- #

def binary_mask_to_polygon(mask):
    mask = mask.astype(np.uint8)
    shape = np.array(mask.shape)[::-1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.asarray(contour) / shape for contour in contours]
    contours = [contour.flatten().tolist() for contour in contours]
    return contours

def polygon_to_binary_mask(polylines, shape):
    mask = np.zeros(shape[:2], np.uint8)
    shape = np.array(shape[:2])[::-1]
    polylines = [(np.asarray(p).reshape(-1, 2) * shape).astype(int) for p in polylines]
    return cv2.fillPoly(mask, polylines, color=1)

# ------------------------------- Bounded mask ------------------------------- #

def bounded_to_binary_mask(obj_mask, bbox, shape, min_size=1):
    # W, H = shape[:2]
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = box_to_xyxy(bbox, shape)
    if min(x2-x1, y2-y1) <= min_size:
        return 
    mask[y1:y2, x1:x2] = cv2.resize(
        np.asarray(obj_mask, dtype=np.uint8), 
        (max(1, x2 - x1), max(1, y2 - y1)), 
        interpolation=cv2.INTER_NEAREST)
    return mask

def binary_to_bounded_mask(mask, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return mask[y1:y2, x1:x2]

def parse_object_mask(object, shape):
    if 'mask' in object:
        return bounded_to_binary_mask(object['mask'], object['bounding_box'], shape)
    elif 'polylines' in object:
        return polygon_to_binary_mask(object['polylines'], shape)
    return None

# ------------------------------- Bounding box ------------------------------- #

def box_to_xyxy(bbox, shape):
    # W, H = shape
    H, W = shape[:2]
    x1 = int(min(max(0, bbox['top_left']['x'] * W), W-1))
    y1 = int(min(max(0, bbox['top_left']['y'] * H), H-1))
    x2 = int(min(max(0, bbox['bottom_right']['x'] * W), W-1))
    y2 = int(min(max(0, bbox['bottom_right']['y'] * H), H-1))
    return np.array([x1, y1, x2, y2])

def xyxy_to_box(xyxy, shape):
    x1, y1, x2, y2 = xyxy
    h, w = shape[:2]
    return {
        "top_left": { "x": x1 / w, "y": y1 / h },
        "bottom_right": { "x": x2 / w, "y": y2 / h },
    }

# --------------------------------- Detectron -------------------------------- #

def detectron2_objects(instances, shape, classes=None):
    instances = instances.to('cpu')
    if classes is None:
        assert instances.has('pred_labels')
        labels = instances.get('pred_labels')
    else:
        labels = np.asarray(classes)[instances.pred_classes.int().numpy()]
    return [
        object(i, label, bbox, mask, confidence, shape=shape)
        for i, (label, bbox, mask, confidence) in enumerate(zip(
            labels,
            instances.pred_boxes.tensor.numpy(),
            instances.pred_masks.int().numpy(),
            instances.scores.numpy(),
        ))
    ]

def get_frame_objects(base, i, shape, load_mask=True):
    objects = get_objects(base, i)

    boxes = []
    masks = []
    labels = []
    confidences = []
    track_ids = []
    for obj in objects:
        boxes.append(box_to_xyxy(obj['bounding_box'], shape))
        mask = None
        if load_mask:
            mask = parse_object_mask(obj, shape)
        masks.append(mask if mask is not None else np.zeros(shape[:2], dtype=bool))
        labels.append(obj.get('label'))
        confidences.append(obj.get('confidence', 1))
        track_ids.append(obj.get('index'))
    return (
        np.asarray(boxes).reshape(-1, 4),
        np.asarray(masks).reshape(-1, *shape[:2]),
        np.asarray(labels),
        np.asarray(confidences),
        np.asarray(track_ids),
    )

def get_objects(base, i):
    try:
        if isinstance(i, int):
            i = str(i+1)
        return base['frames'][i]['objects']['objects'] or []
    except KeyError:
        return []

def get_sv_detections(base, i, shape):
    import supervision as sv
    xyxy, mask, labels, confidence, track_ids = get_frame_objects(base, i, shape, load_mask=True)
    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        tracker_id=track_ids,
    ), labels

# ---------------------------------------------------------------------------- #
#                                   Manifest                                   #
# ---------------------------------------------------------------------------- #


def manifest(index, description=""):
    return {
        "type": "eta.core.datasets.LabeledVideoDataset",
        "description": description,
        "index": index or [],
        # [
        #     {
        #         "data": "data/<uuid1>.<ext>",
        #         "labels": "labels/<uuid1>.json"
        #     },
        #     {
        #         "data": "data/<uuid2>.<ext>",
        #         "labels": "labels/<uuid2>.json"
        #     },
        #     ...
        # ]
    }


def save(ann, fname, overwrite=True):
    logging.info('writing to %s', fname)
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)

    # merge with existing json
    if os.path.isfile(fname) and not overwrite:
        prev = load(fname)
        ix = prev.get('index') or []
        lookup = {d['data']: i for i, d in enumerate(ix)}
        for d in ann.get('index') or []:
            if d['data'] in lookup:
                ix[lookup[d['data']]].update(d)
            else:
                ix.append(d)
        ann['index'] = ix
        prev.update(ann)
        ann = prev
    
    # write json
    with open(fname, 'wb') as f: 
        f.write(orjson.dumps(ann, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
    logging.info('wrote to %s', fname)
    return fname


def load(fname):
    with open(fname, 'rb') as f: 
        return orjson.loads(f.read())


# --------------------------------- Filenames -------------------------------- #

def data_fname(out_dir, f, ext='.mp4'):
    return os.path.join(out_dir, 'data', os.path.splitext(os.path.basename(f))[0] + ext)

def label_fname(out_dir, f, ext='.json'):
    return os.path.join(out_dir, 'labels', os.path.splitext(os.path.basename(f))[0] + ext)

def manifest_fname(out_dir):
    return os.path.join(out_dir, 'manifest.json')


# ------------------------------ Fixing manifest ----------------------------- #

# def manifest_from_dirs(video_dir, dataset_dir):
#     fs = glob.glob(data_fname(video_dir, '*'))
#     return manifest([
#         {'data': f, 'labels': label_fname(dataset_dir, f)}
#         for f in fs
#         if os.path.isfile(label_fname(dataset_dir, f))
#     ])

def dataset_from_labels(dataset_dir, *video_dirs):
    file_list = []
    for f in tqdm.tqdm(sorted(glob.glob(label_fname(dataset_dir, '*')))):
        name = os.path.splitext(os.path.basename(f))[0]
        data_fname = next((
            f for ddir in video_dirs
            for f in glob.glob(os.path.join(ddir, f'{name}.*'))
        ), None)
        if data_fname is None:
            logging.warning(f'Could not find {name}')
        else:
            logging.info(f'Using: {name}')
            file_list.append({'data': data_fname, 'labels': f})
    return save(manifest(file_list), manifest_fname(dataset_dir), overwrite=True)


# def rewrite_manifest(video_dir, dataset_dir):
#     manifest = manifest_from_dirs(video_dir, dataset_dir)
#     fname = save(manifest, manifest_fname(dataset_dir))
#     return load(fname)


def file_tree(root_dir):
    import pathtrees as pt
    return pt.tree(root_dir, {
        'data': {
            '{video_name}.{ext}': 'video',
        },
        'labels': {
            '{video_name}.json': 'labels',
        },
        'manifest.json': 'manifest',
    })

def normalize_dataset(dataset_dir, *video_dirs, patches_field='detections_tracker'):
    dataset_from_labels(dataset_dir, *video_dirs)

    if patches_field:
        manifest = load(manifest_fname(dataset_dir))
        for m in tqdm.tqdm(manifest['index']):
            labels = load(m['labels'])
            if 'frames' not in labels:
                logging.warning("Missing data in %s", m['labels'])
                continue
            
            any_mod = False
            for i, f in labels['frames'].items():
                if 'objects' in f:
                    for o in f['objects']['objects']:
                        if o['name'] != patches_field:
                            o['name'] = patches_field
                            any_mod = True
            if any_mod:
                logging.debug('%s.%s: %s -> %s', m['labels'], i, o['name'], patches_field)
                save(labels, m['labels'])
            else:
                logging.info("No changes to %s", m['labels'])



if __name__ == '__main__':
    from tqdm.contrib.logging import logging_redirect_tqdm
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        import fire
        fire.Fire()

'''

{
    "attrs": {
        "attrs": [
            {"type": "eta.core.data.CategoricalAttribute","name": "weather","value": "rain","confidence": 0.95},
            {"type": "eta.core.data.NumericAttribute","name": "fps","value": 30.0},
            {"type": "eta.core.data.BooleanAttribute","name": "daytime","value": true}
        ]
    },
    "frames": {
        "1": {
            "frame_number": 1,
            "attrs": {
                "attrs": [
                    {"type": "eta.core.data.CategoricalAttribute","name": "scene","value": "intersection","confidence": 0.9},
                    {"type": "eta.core.data.NumericAttribute","name": "quality","value": 0.5}
                ]
            },
            "objects": {
                "objects": [
                    {
                        "label": "car",
                        "bounding_box": {"bottom_right": {"y": 1.0,"x": 1.0},"top_left": {"y": 0.0,"x": 0.0}},
                        "confidence": 0.9,
                        "index": 1,
                        "attrs": {"attrs": [{"type": "eta.core.data.CategoricalAttribute","name": "make","value": "Honda"}]}
                    }
                ]
            }
        }
    }
}

'''