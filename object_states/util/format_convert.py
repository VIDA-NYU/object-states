import cv2
import numpy as np
import torch
# import fiftyone as fo
import supervision as sv



# ---------------------------------------------------------------------------- #
#                           Converting to Supervision                          #
# ---------------------------------------------------------------------------- #


def fo_to_sv(fo_detections, shape, classes=None, mask=True): # h, w
    h, w = shape[:2]
    # get detections list
    fo_detections = getattr(fo_detections, 'detections', None) or []
    ds = [d for d in fo_detections if d.mask is not None]

    # convert to sv Detections object
    labels = np.array([d.label for d in ds], dtype=str)
    class_id = None
    if classes is not None:
        classes = list(classes)
        class_id = np.array([classes.index(d.label) for d in ds])
    detections = sv.Detections(
        xyxy=xywhn2xyxy(np.array([d.bounding_box for d in ds]).reshape(-1, 4), (h, w)),
        class_id=class_id,
        tracker_id=np.array([d.index for d in ds]),
        confidence=np.array([d.confidence for d in ds]),
        mask=np.array([d.to_segmentation(frame_size=(w, h), target=1).mask for d in ds]) if mask and len(ds) else None,
    )
    return detections, labels


def detectron_to_sv(outputs, classes=None):
    outputs = outputs.to('cpu')
    detections = sv.Detections(
        xyxy=outputs.pred_boxes.tensor.numpy(),
        mask=outputs.pred_masks.numpy() if outputs.has('pred_masks') else None,
        class_id=outputs.pred_classes.int().numpy() if outputs.has('pred_classes') else None,
        tracker_id=outputs.track_ids.int().numpy() if outputs.has('track_ids') else None,
        confidence=outputs.scores.numpy() if outputs.has('scores') else None,
    )
    labels = (
        outputs.pred_labels if outputs.has('pred_labels') else 
        np.asarray(classes)[detections.class_id] if detections.class_id is not None else 
        None)
    return detections, labels


# ---------------------------------------------------------------------------- #
#                            Converting to FiftyOne                            #
# ---------------------------------------------------------------------------- #



def detectron_to_fo(outputs, labels, shape):
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    for xyxy, score, cls_id, mask in zip(
        instances.pred_boxes, # top-left, bottom-right xyxy
        instances.scores, # confidences
        instances.pred_classes, # class_ids
        instances.pred_masks  # segmentation masks (binary masks)
    ):
        x1, y1, x2, y2 = xyxy
        fo_mask = mask.numpy()[int(y1):int(y2), int(x1):int(x2)]
        detection = fo.Detection(
            label=labels[cls_id], 
            confidence=float(score), 
            bounding_box=xyxy2xywhn(xyxy, shape), 
            mask=fo_mask)
        detections.append(detection)
    return fo.Detections(detections=detections)



# ---------------------------------------------------------------------------- #
#                               FiftyOne Helpers                               #
# ---------------------------------------------------------------------------- #



def detection2mask(d, orig_shape, new_shape):
    # the mask of a detection object is relative to the bounding box
    # convert to segmentation object
    return torch.as_tensor(cv2.resize(d.to_segmentation(frame_size=orig_shape[:2], target=1).mask, new_shape[:2]))

def mask2detection(mask):
    # the mask of a detection object needs to be relative to the bounding box
    # convert to segmentation object
    return fo.Segmentation(mask=mask.cpu().numpy()).to_detections().detections[0]


def detection2mask_alt(d, orig_shape, new_shape=None): # (w,h), (w,h)
    out = np.zeros(orig_shape[:2][::-1])
    b = xywhn2xyxy(np.array(d.bounding_box), orig_shape).astype(int)
    out[b[1]:b[3], b[0]:b[2]] = d.mask
    if new_shape is not None:
        out = cv2.resize(out, new_shape[:2])
    return torch.as_tensor(out)

# ---------------------------------------------------------------------------- #
#                                  Coordinates                                 #
# ---------------------------------------------------------------------------- #



def xyxy2xywhn(xyxy, shape):
    # xyxy = np.asarray(xyxy)
    xyxy[..., 2:] -= xyxy[..., :2]
    xyxy[..., 0] /= shape[1]
    xyxy[..., 1] /= shape[0]
    xyxy[..., 2] /= shape[1]
    xyxy[..., 3] /= shape[0]
    return xyxy

def xywhn2xyxy(xywh, shape):
    # xywh = np.asarray(xywh)
    xywh[..., 2:] += xywh[..., :2]
    xywh[..., 0] *= shape[1]
    xywh[..., 1] *= shape[0]
    xywh[..., 2] *= shape[1]
    xywh[..., 3] *= shape[0]
    return xywh
